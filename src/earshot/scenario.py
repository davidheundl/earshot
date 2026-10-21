"""Scenarios: seeded, reproducible scripts of sources, jamming zones and sensor faults.

A scenario is a YAML file (see scenarios/). Ground truth lives here and nowhere else;
agents never see it, only the sensor models and the dashboard do. An optional `params:` map
overrides the decision thresholds (state.PolicyParams) for every drone.

`threat:` names the Threat Class the swarm hunts (default drone: the airport; vehicle: a
Reserve, where poachers drive in). Every drone knows it before take-off (SwarmConfig). A
Source's `intruder:` (default: its class is the Threat Class) is ground truth for metrics and
the Threat Log: a ranger's vehicle in a Reserve is a Vehicle but no intruder. `kind:` says
what a Source is ("elephant herd", "poacher pickup"): the camera stand-in's label for the
Threat Log and the model the Webots builder draws. `confusion:` overrides rows of the sound
stand-in's confusion table (sensors.CONFUSION), e.g. animals that sound like an engine.

Events change a scenario while it runs. The same op dicts come from three places: the YAML
`events:` list (applied at their `t`), live hotkeys on the dashboard (applied at the next
tick), and a recorded live session (`runs/live-*.events.jsonl`, replayable with
`earshot fast --events`). `Scenario.apply(op, t)` is the one place they take effect.

    {op: add_source, cls: drone, x: 100, y: 200}          # or path: [[x, y], ...]
        optional: to: [x, y], duration (120 s), level_db, rf: [2g4], id, kind, intruder
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
    intruder: bool = False  # truth: a threat the swarm should find (metrics, Threat Log)
    kind: str = ""  # what it is, free text: the camera stand-in's label, the Webots model

    def pose(self, t: float, back: float = 0.0) -> tuple[float, float, float] | None:
        """(x, y, heading_deg) at time t, `back` metres behind the head along the path
        (train wagons). None when inactive."""
        if not (self.start <= t <= self.end):
            return None
        pts = np.asarray(self.path, dtype=float)
        if len(pts) == 1:
            return float(pts[0, 0]), float(pts[0, 1]), 0.0
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = seg.sum()
        s = (t - self.start) / max(self.end - self.start, 1e-9) * total - back
        s = min(max(s, 0.0), total)
        cum = np.cumsum(seg)
        i = min(int(np.searchsorted(cum, s, side="right")), len(seg) - 1)
        s0 = cum[i] - seg[i]
        f = 0.0 if seg[i] == 0 else (s - s0) / seg[i]
        d = pts[i + 1] - pts[i]
        p = pts[i] + f * d
        return float(p[0]), float(p[1]), math.degrees(math.atan2(d[0], d[1])) % 360.0

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
    if isinstance(v, str):  # rf: 2g4 means one band, not the characters '2', 'g', '4'
        return (v,)
    return tuple(str(b) for b in v) if v else ()


# -- validation of Events: check everything before an op changes anything ------------------

FAULT_KINDS = ("stuck_bearing", "deaf", "phantom")


def _num(v, name: str):
    """A finite int or float, returned unchanged (so valid Events log exactly as before)."""
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    return v


def _float(v, name: str, positive: bool = False) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {v!r}") from None
    if not math.isfinite(f) or (positive and f <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite, got {v!r}")
    return f


def _point(p, name: str) -> tuple:
    if not isinstance(p, list | tuple) or len(p) != 2:
        raise ValueError(f"{name} must be [x, y], got {p!r}")
    return (_num(p[0], name), _num(p[1], name))


def _target_class(v) -> TargetClass:
    name = _CLASS_ALIASES.get(v.lower(), v.upper()) if isinstance(v, str) else None
    if name not in TargetClass.__members__:
        raise ValueError(f"unknown cls {v!r}")
    return TargetClass[name]


def _confusion(d: dict | None) -> dict:
    """`confusion: {nuisance: {drone: 0.05, vehicle: 0.3, train: 0, nothing: 0.65}}`: whole
    rows by true class, each row's weights over all four predicted classes."""
    out = {}
    for true, row in (d or {}).items():
        cls = _target_class(true)
        if not isinstance(row, dict) or {_target_class(k) for k in row} != set(TargetClass):
            raise ValueError(f"confusion row {true!r} must give a weight for each of "
                             f"{[c.name.lower() for c in TargetClass]}")
        out[cls] = {_target_class(k): _float(v, f"confusion {true}.{k}") for k, v in row.items()}
    return out


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
    map_cell: float = 20.0  # m, Probability Map cell
    params: dict = field(default_factory=dict)  # state.PolicyParams overrides
    audio: dict = field(default_factory=dict)  # {classifier: "mod:attr", sounds: dir}
    camera: dict = field(default_factory=dict)  # {detector: yolo, ...}: sensors/camera.py
    threat: TargetClass = TargetClass.DRONE  # the Threat Class the swarm hunts
    confusion: dict = field(default_factory=dict)  # sound stand-in rows: sensors.CONFUSION
    environment: Environment = field(default_factory=Environment)
    events: list[dict] = field(default_factory=list)  # pending, sorted by t
    _live_count: int = 0

    @classmethod
    def load(cls, path: str | Path, seed: int | None = None) -> Scenario:
        d = yaml.safe_load(Path(path).read_text())
        root = Path(path).resolve().parent.parent  # scenarios/ sits in the repo root
        city_raw = None

        def resolve(p):  # {road: i} / {rail: i}: follow the city's polyline
            nonlocal city_raw
            if not isinstance(p, dict):
                return [tuple(q) for q in p]
            if city_raw is None:
                import json

                city_raw = json.loads((root / d["city"]).read_text())
            kind = "roads" if "road" in p else "rails"
            pts = [tuple(q) for q in city_raw[kind][int(p.get("road", p.get("rail", 0)))]]
            return pts[::-1] if p.get("reverse") else pts

        threat = _target_class(d.get("threat", "drone"))
        if threat == TargetClass.NOTHING:
            raise ValueError("threat must be drone, vehicle or train")
        sc = cls(
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
            map_cell=float(d.get("map_cell", 20.0)),
            params=dict(d.get("params") or {}),
            audio=dict(d.get("audio") or {}),
            camera=dict(d.get("camera") or {}),
            threat=threat,
            confusion=_confusion(d.get("confusion")),
            sources=[
                Source(
                    id=s["id"],
                    cls=(c := TargetClass[_CLASS_ALIASES.get(s["cls"].lower(), s["cls"].upper())]),
                    start=float(s["start"]), end=float(s["end"]),
                    path=resolve(s["path"]),
                    level_db=float(s["level_db"]), rf_bands=_rf_bands(s.get("rf")),
                    intruder=bool(s.get("intruder", c == threat)), kind=str(s.get("kind", "")),
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
        bad = [f.kind for f in sc.faults if f.kind not in FAULT_KINDS]
        if bad:
            raise ValueError(f"unknown fault kind(s) {bad}, expected one of {FAULT_KINDS}")
        return sc

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
        """Pop scripted events whose time has come. Recorded events carry round(t, 3) while
        the clock drifts (0.1 added ten times is 0.9999999999999999), hence the epsilon."""
        t += 1e-6
        due = [e for e in self.events if e["t"] <= t]
        self.events = [e for e in self.events if e["t"] > t]
        return due

    def apply(self, op: dict, t: float) -> str:
        """Apply one event at sim time t. Returns a one-line description for logs and UI.

        Every field is checked before anything changes: a bad op raises ValueError (or
        KeyError for a missing field) and leaves the scenario as it was."""
        kind = op["op"]
        if kind == "add_source":
            from earshot.sensors.acoustics import DEFAULT_LEVEL  # avoid import cycle
            from earshot.sensors.rf import BANDS

            cls = _target_class(op["cls"])
            if "path" in op:
                if not isinstance(op["path"], list | tuple) or not op["path"]:
                    raise ValueError(f"path must be a non-empty list of [x, y], got {op['path']!r}")
                path = [_point(p, "path point") for p in op["path"]]
            else:
                path = [(_num(op["x"], "x"), _num(op["y"], "y"))]
            if "to" in op:
                path.append(_point(op["to"], "to"))
            duration = _float(op.get("duration", 120.0), "duration", positive=True)
            level = _float(op.get("level_db", DEFAULT_LEVEL[cls]), "level_db")
            rf = op.get("rf", cls == TargetClass.DRONE)
            if not (isinstance(rf, bool) or rf is None or (
                    isinstance(rf, list | tuple) and all(isinstance(b, str) for b in rf))):
                raise ValueError(f"rf must be true/false or a list of bands, got {rf!r}")
            bands = _rf_bands(rf)
            unknown = [b for b in bands if b not in BANDS]
            if unknown:
                raise ValueError(f"unknown rf band(s) {unknown}")
            sid = op.get("id") or f"live_{cls.name.lower()}_{self._live_count}"
            if not isinstance(sid, str):
                raise ValueError(f"id must be a string, got {sid!r}")
            if any(s.id == sid for s in self.sources):
                raise ValueError(f"source id {sid!r} already exists")
            intruder = op.get("intruder", cls == self.threat)
            if not isinstance(intruder, bool):
                raise ValueError(f"intruder must be true or false, got {intruder!r}")
            kind = op.get("kind", "")
            if not isinstance(kind, str):
                raise ValueError(f"kind must be a string, got {kind!r}")
            self._live_count += 1
            src = Source(sid, cls, t, t + duration, path, level, bands, intruder, kind)
            self.sources.append(src)
            return f"{sid}: {cls.name.lower()} at {path[0][0]:.0f}, {path[0][1]:.0f}"
        if kind == "jam":
            center = (_num(op["x"], "x"), _num(op["y"], "y"))
            z = JamZone(t, t + _float(op.get("duration", 30.0), "duration", positive=True),
                        center, _float(op.get("r", 300.0), "r", positive=True))
            self.jamming.append(z)
            return f"jam r={z.radius:.0f} m at {z.center[0]:.0f}, {z.center[1]:.0f}"
        if kind == "clear_jam":
            self.jamming = [replace(j, end=t) if j.active(t) else j for j in self.jamming]
            return "jamming cleared"
        if kind == "fault":
            drone = _num(op["drone"], "drone")
            if drone not in {d.id for d in self.drones}:
                raise ValueError(f"no drone {drone!r}")
            if op["kind"] not in FAULT_KINDS:
                raise ValueError(f"unknown fault kind {op['kind']!r}")
            self.faults.append(Fault(int(drone), t, op["kind"]))
            return f"drone {op['drone']}: {op['kind']}"
        if kind == "clear_faults":
            who = op.get("drone")
            if who is not None:
                _num(who, "drone")
            self.faults = [replace(f, end=min(f.end, t)) if who in (None, f.drone) else f
                           for f in self.faults]
            return "faults cleared" + ("" if who is None else f" on drone {who}")
        if kind == "set_env":
            fields = {k: v for k, v in op.items() if k in Environment.__dataclass_fields__}
            unknown = set(op) - set(fields) - {"op", "t", "live", "note"}
            if unknown or not fields:
                raise ValueError(f"set_env takes {sorted(Environment.__dataclass_fields__)}, "
                                 f"got {sorted(unknown) or 'nothing'}")
            for k, v in fields.items():
                if k == "ego_filter":
                    if not isinstance(v, bool):
                        raise ValueError(f"ego_filter must be true or false, got {v!r}")
                else:
                    _num(v, k)
            self.environment = replace(self.environment, **fields)
            return "environment: " + ", ".join(f"{k}={v}" for k, v in fields.items())
        raise ValueError(f"unknown op {kind!r}")
