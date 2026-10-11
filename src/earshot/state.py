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
    home: tuple[float, float] | None = None  # centre of its initial patrol sector
