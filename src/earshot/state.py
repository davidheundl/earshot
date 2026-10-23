"""Plain data passed between the world, the backends and the agents. No behaviour here."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from earshot.messages import Mode, Sensor, TargetClass


@dataclass(frozen=True)
class Command:
    """What an agent asks its body to do this tick. The backend owns the kinematics."""

    mode: Mode
    goal: tuple[float, float] | None = None  # metres; fly towards this (PATROL / VERIFY / FILL_GAP)
    perch_site: str | None = None  # LISTEN: land on this Perch Site, motors off
    camera_on: bool = False
    look_at: tuple[float, float] | None = None  # VERIFY: hold a standoff and face this point
    speed: float | None = None  # m/s asked for (Follow); None: the Mode's cruise speed


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
    range: float = 0.0  # m to what the camera saw (geolocated, or a drone's box size); 0: none.
    #                     Stays on the drone: a Follower steers by it, the radio never carries it


@dataclass(frozen=True)
class Sensed:
    """Everything a drone learns about itself and the world in one tick."""

    state: DroneState
    readings: list[SensorReading] = field(default_factory=list)
    camera_frame: object | None = None  # backend-specific image, only for the Verifier
    guide: tuple[float, float] | None = None  # demo mock (Scenario `guide: true`): where the
    #   intruder a Verifier flies to really is. The one place truth reaches an Agent; off by
    #   default, never in eval


@dataclass(frozen=True)
class DroneSpec:
    id: int
    x: float
    y: float
    home: tuple[float, float] | None = None  # centre of its initial Sector
    default: Mode = Mode.LISTEN  # LISTEN: perch in its Sector; PATROL: fly its Sector
    fixed: bool = False  # permanently perched at home, takes no tasks (a microphone array)


@dataclass(frozen=True)
class PolicyParams:
    """The decision layer's tunable numbers (agent/policy.py), the same on every drone.
    Defaults are the hand-tuned values; `earshot tune` searches them, `--params` loads them."""

    verify_min_strength: float = 4.0  # log-odds; 8 waited ~4x as long to confirm, 0 floods
    verify_min_confidence: float = 0.5
    top_suspicions: int = 2  # only the strongest Suspicions spawn tasks; ghosts rank below
    confirmed_shadow: float = 200.0  # m: Suspicions this close to a confirmed drone are side lobes
    settle: float = 3.0  # s, > typical 3-hop Claim latency, so rival bids arrive before anyone flies
    same_task_radius: float = 200.0  # m: a Claim of the same kind this close covers a drifted one
    arrive: float = 60.0  # m; above the Verifier's standoff (Kinematics.standoff: 40 m, or camera.standoff)
    dismiss_after: float = 8.0  # s at the target with no Suspicion left -> dismissed
    outbid_cooldown: float = 5.0  # s an outbid drone leaves that place alone
    verify_timeout: float = 90.0  # s at the target without confirming -> given up (Ghost held up)
    follow_for: float = 0.0  # s a Verifier stays with what it confirmed (Follow); 0: hand off at once
    follow_gap: float = 40.0  # m a Follower keeps behind its Quarry (match camera.standoff)
    follow_gain: float = 0.25  # 1/s: m/s of speed above the Quarry's per metre it is behind
    follow_max_speed: float = 25.0  # m/s a Follower sprints at most (the backend caps it too)
    max_verify_distance: float = math.inf  # m: no bid on a task further off (a big swarm has a
    #                                        nearer drone, though its Claim may be lost)

    @classmethod
    def from_dict(cls, d: dict | None) -> PolicyParams:
        d = d or {}
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown policy params: {sorted(unknown)}")
        p = cls(**d)
        return replace(p, top_suspicions=int(p.top_suspicions))


@dataclass(frozen=True)
class SwarmConfig:
    """What every Agent knows before take-off: the swarm's homes and default Modes."""

    homes: dict[int, tuple[float, float]]
    defaults: dict[int, Mode]
    policy: str = "earshot"  # "earshot" | "nearest" (nearest drone always flies) | "no_perch"
    radio_scheme: str = "ours"  # "ours" | "flood" | "unicast3" (agent/store.py)
    fixed: frozenset = frozenset()  # drones that never leave their home roof
    map_cell: float = 20.0  # m, Probability Map cell
    params: PolicyParams = field(default_factory=PolicyParams)
    threat: TargetClass = TargetClass.DRONE  # the Threat Class: what the swarm verifies
    relay_radius: float = math.inf  # m: Tracks and Claims about places further off are not relayed

    @classmethod
    def from_specs(cls, specs: list[DroneSpec], policy: str = "earshot",
                   radio_scheme: str = "ours", map_cell: float = 20.0,
                   params: PolicyParams | None = None,
                   threat: TargetClass = TargetClass.DRONE,
                   relay_radius: float = math.inf) -> SwarmConfig:
        return cls(
            homes={d.id: d.home or (d.x, d.y) for d in specs},
            defaults={d.id: d.default for d in specs},
            policy=policy,
            radio_scheme=radio_scheme,
            fixed=frozenset(d.id for d in specs if d.fixed),
            map_cell=map_cell,
            params=params or PolicyParams(),
            threat=threat,
            relay_radius=relay_radius,
        )
