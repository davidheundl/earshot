"""Scenarios: seeded, reproducible scripts of sources, jamming zones and sensor faults.

A scenario is a YAML file (see scenarios/). Ground truth lives here and nowhere else;
agents never see it, only the sensor models and the dashboard do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from earshot.messages import TargetClass
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
    environment: Environment = field(default_factory=Environment)

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
                          home=tuple(x["home"]) if "home" in x else None)
                for x in d["drones"]
            ],
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
        return [f for f in self.faults if f.drone == drone and t >= f.start]
