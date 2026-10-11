"""Plain data passed between the world, the backends and the agents. No behaviour here."""

from __future__ import annotations

from dataclasses import dataclass, field

from earshot.messages import Mode, Sensor


@dataclass(frozen=True)
class Command:
    """What an agent asks its body to do this tick. The backend owns the kinematics."""

    mode: Mode
    goal: tuple[float, float] | None = None  # metres; fly towards this (PATROL / VERIFY / FILL_GAP)
    perch_site: str | None = None  # LISTEN: land on this Perch Site, motors off
    camera_on: bool = False


@dataclass(frozen=True)
class DroneState:
    id: int
    t: float
    x: float
    y: float
    z: float
    heading: float  # degrees, 0 = north, clockwise
    mode: Mode
    perched_on: str | None  # Perch Site id while landed, else None
    battery: float  # 0..1
    motor_rpm: float  # 0 when perched; drives the ego-noise model
    camera_on: bool


@dataclass(frozen=True)
class SensorReading:
    """One measurement before it becomes a radio Observation."""

    sensor: Sensor
    bearing: float  # degrees; NaN when the sensor gives no direction (RF)
    bearing_sigma: float
    class_probs: tuple[float, float, float, float]  # indexed by TargetClass
    level: float  # dB (sound level or RF received power)


@dataclass(frozen=True)
class Sensed:
    """Everything a drone learns about itself and the world in one tick."""

    state: DroneState
    readings: list[SensorReading] = field(default_factory=list)
    camera_frame: object | None = None  # backend-specific image, only for the Verifier


@dataclass(frozen=True)
class DroneSpec:
    id: int
    x: float
    y: float
    home: tuple[float, float] | None = None  # centre of its initial Sector
    default: Mode = Mode.LISTEN  # LISTEN: perch in its Sector; PATROL: fly its Sector


@dataclass(frozen=True)
class SwarmConfig:
    """What every Agent knows before take-off: the swarm's homes and default Modes."""

    homes: dict[int, tuple[float, float]]
    defaults: dict[int, Mode]
    policy: str = "earshot"  # "earshot" | "nearest" (nearest drone always flies) | "no_perch"
    radio_scheme: str = "ours"  # "ours" | "flood" | "unicast3" (agent/store.py)

    @classmethod
    def from_specs(cls, specs: list[DroneSpec], policy: str = "earshot",
                   radio_scheme: str = "ours") -> SwarmConfig:
        return cls(
            homes={d.id: d.home or (d.x, d.y) for d in specs},
            defaults={d.id: d.default for d in specs},
            policy=policy,
            radio_scheme=radio_scheme,
        )
