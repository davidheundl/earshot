"""Scenarios: seeded, reproducible scripts of sources, jamming zones and sensor faults.

A scenario is a YAML file (see scenarios/). Ground truth lives here and nowhere else;
agents never see it, only the sensor models and the dashboard do.

Events change a scenario while it runs. The same op dicts come from three places: the YAML
`events:` list (applied at their `t`), live hotkeys on the dashboard (applied at the next
tick), and a recorded live session (`runs/live-*.events.jsonl`, replayable with
`earshot fast --events`). `Scenario.apply(op, t)` is the one place they take effect.

    {op: add_source, cls: drone, x: 100, y: 200}          # or path: [[x, y], ...]
        optional: to: [x, y], duration (120 s), level_db, rf: [2g4], id
    {op: jam, x: 0, y: 0}                                  optional: r (300 m), duration (30 s)
    {op: clear_jam}
    {op: fault, drone: 3, kind: stuck_bearing}             kinds: stuck_bearing | deaf | phantom
    {op: clear_faults}                                     optional: drone
    {op: set_env, background_db: 55, ego_filter: false}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml

from earshot.messages import Mode, TargetClass
from earshot.state import DroneSpec


@dataclass(frozen=True)
class Source:
    id: str
    cls: TargetClass
    start: float
    end: float
    path: list[tuple[float, float]]  # visited at constant speed between start and end
    level_db: float  # sound power level, dB(A); see sensors.acoustics.SOUND_POWER
    rf_bands: tuple[str, ...] = ()  # emitting links, keys of sensors.rf.BANDS

    def position(self, t: float) -> tuple[float, float] | None:
        if not (self.start <= t <= self.end):
            return None
        pts = np.asarray(self.path, dtype=float)
        if len(pts) == 1 or self.end == self.start:
            return float(pts[0, 0]), float(pts[0, 1])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = (t - self.start) / (self.end - self.start) * seg.sum()
        i = min(int(np.searchsorted(np.cumsum(seg), s, side="right")), len(seg) - 1)
        s0 = seg[:i].sum()
        f = 0.0 if seg[i] == 0 else (s - s0) / seg[i]
        p = pts[i] + f * (pts[i + 1] - pts[i])
        return float(p[0]), float(p[1])


@dataclass(frozen=True)
class JamZone:
    start: float
    end: float
    center: tuple[float, float]
    radius: float

    def active(self, t: float) -> bool:
        return self.start <= t <= self.end


@dataclass(frozen=True)
class Fault:
    drone: int
    start: float
    kind: str  # "stuck_bearing" | "deaf" | "phantom"
    end: float = math.inf


@dataclass
class Environment:
    background_db: float = 40.0  # dB(A); night 40, day 50-60
    ego_filter: bool = True  # subtract predicted motor noise in flight
    ego_filter_db: float = 8.0  # realistic gain 5-10 dB (section 4)


_CLASS_ALIASES = {"nuisance": "NOTHING", "fan": "NOTHING", "truck": "VEHICLE"}


def _rf_bands(v) -> tuple[str, ...]:
    if v is True:
        return ("2g4",)
    return tuple(str(b) for b in v) if v else ()


@dataclass
class Scenario:
    name: str
    seed: int
    city: str
    duration: float
    dt: float
    drones: list[DroneSpec]
    sources: list[Source]
    jamming: list[JamZone] = field(default_factory=list)
    faults: list[Fault] = field(default_factory=list)
    radio: dict = field(default_factory=dict)
    policy: str = "earshot"  # see state.SwarmConfig
    environment: Environment = field(default_factory=Environment)
    events: list[dict] = field(default_factory=list)  # pending, sorted by t
    _live_count: int = 0

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> Scenario:
        d = yaml.safe_load(Path(path).read_text())
        return cls(
            name=d["name"],
            seed=d.get("seed", 0) if seed is None else seed,
            city=d["city"],
            duration=float(d["duration"]),
            dt=float(d.get("dt", 0.1)),
            drones=[
                DroneSpec(id=int(x["id"]), x=float(x["x"]), y=float(x["y"]),
                          home=tuple(x["home"]) if "home" in x else None,
                          default=Mode[x.get("default", "listen").upper()],
                          fixed=bool(x.get("fixed", False)))
                for x in d["drones"]
            ],
            policy=d.get("policy", "earshot"),
            sources=[
                Source(
                    id=s["id"],
                    cls=TargetClass[_CLASS_ALIASES.get(s["cls"].lower(), s["cls"].upper())],
                    start=float(s["start"]), end=float(s["end"]),
                    path=[tuple(p) for p in s["path"]],
                    level_db=float(s["level_db"]), rf_bands=_rf_bands(s.get("rf")),
                )
                for s in d.get("sources", [])
            ],
            jamming=[
                JamZone(float(j["start"]), float(j["end"]), tuple(j["center"]), float(j["radius"]))
                for j in d.get("jamming", [])
            ],
            faults=[Fault(int(f["drone"]), float(f["start"]), f["kind"]) for f in d.get("faults", [])],
            radio=d.get("radio", {}),
            environment=Environment(**d.get("environment", {})),
            events=sorted(d.get("events", []), key=lambda e: e["t"]),
        )

    def active_sources(self, t: float) -> list[tuple[Source, tuple[float, float]]]:
        out = []
        for s in self.sources:
            p = s.position(t)
            if p is not None:
                out.append((s, p))
        return out

    def jam_zones(self, t: float) -> list[JamZone]:
        return [j for j in self.jamming if j.active(t)]

    def faults_for(self, drone: int, t: float) -> list[Fault]:
        return [f for f in self.faults if f.drone == drone and f.start <= t < f.end]

    # -- events ----------------------------------------------------------------------------

    def due_events(self, t: float) -> list[dict]:
        """Pop scripted events whose time has come."""
        due = [e for e in self.events if e["t"] <= t]
        self.events = [e for e in self.events if e["t"] > t]
        return due

    def apply(self, op: dict, t: float) -> str:
        """Apply one event at sim time t. Returns a one-line description for logs and UI."""
        kind = op["op"]
        if kind == "add_source":
            cls = TargetClass[_CLASS_ALIASES.get(op["cls"].lower(), op["cls"].upper())]
            path = [tuple(p) for p in op["path"]] if "path" in op else [(op["x"], op["y"])]
            if "to" in op:
                path.append(tuple(op["to"]))
            sid = op.get("id") or f"live_{cls.name.lower()}_{self._live_count}"
            self._live_count += 1
            from earshot.sensors.acoustics import DEFAULT_LEVEL  # avoid import cycle

            src = Source(sid, cls, t, t + float(op.get("duration", 120.0)), path,
                         float(op.get("level_db", DEFAULT_LEVEL[cls])),
                         _rf_bands(op.get("rf", cls == TargetClass.DRONE)))
            self.sources.append(src)
            return f"{sid}: {cls.name.lower()} at {path[0][0]:.0f}, {path[0][1]:.0f}"
        if kind == "jam":
            z = JamZone(t, t + float(op.get("duration", 30.0)), (op["x"], op["y"]),
                        float(op.get("r", 300.0)))
            self.jamming.append(z)
            return f"jam r={z.radius:.0f} m at {z.center[0]:.0f}, {z.center[1]:.0f}"
        if kind == "clear_jam":
            self.jamming = [replace(j, end=t) if j.active(t) else j for j in self.jamming]
            return "jamming cleared"
        if kind == "fault":
            self.faults.append(Fault(int(op["drone"]), t, op["kind"]))
            return f"drone {op['drone']}: {op['kind']}"
        if kind == "clear_faults":
            who = op.get("drone")
            self.faults = [replace(f, end=min(f.end, t)) if who in (None, f.drone) else f
                           for f in self.faults]
            return "faults cleared" + ("" if who is None else f" on drone {who}")
        if kind == "set_env":
            fields = {k: v for k, v in op.items() if k in Environment.__dataclass_fields__}
            self.environment = replace(self.environment, **fields)
            return "environment: " + ", ".join(f"{k}={v}" for k, v in fields.items())
        raise ValueError(f"unknown op {kind!r}")
