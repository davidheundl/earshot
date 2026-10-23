"""2D twin: kinematic drones over the City, no rendering. Fast mode and the fallback backend.

The Webots adapter reuses this class for motion (ADR 0002: Webots drones are puppeted),
so both backends move drones identically.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from earshot.city import City
from earshot.messages import Mode
from earshot.state import Command, DroneSpec, DroneState


CRUISE = {Mode.PATROL: 8.0, Mode.LISTEN: 10.0, Mode.VERIFY: 15.0, Mode.FILL_GAP: 10.0}  # m/s


@dataclass(frozen=True)
class Kinematics:
    cruise_alt: float = 40.0  # m above ground
    speed: dict | None = None  # m/s per Mode, over CRUISE
    climb_rate: float = 4.0  # m/s
    arrive_radius: float = 2.0  # m
    hover_rpm: float = 6000.0  # hovering is near the most power-hungry state (section 4)
    cruise_rpm: float = 5700.0  # forward flight gains translational lift
    drain_flying: float = 1 / 1500  # battery fraction per s (~25 min flight)
    drain_perched: float = 1 / 36000  # ~10 h perched (assumption, section 4)
    standoff: float = 40.0  # m a Verifier keeps from the point it looks at
    max_speed: float = 25.0  # m/s the airframe flies at most, whatever a Command asks

    def v(self, mode: Mode, asked: float | None = None) -> float:
        """The speed a Command gets: what it asked for (Follow), within max_speed, else the
        Mode's cruise speed."""
        if asked is not None:
            return min(max(asked, 0.0), self.max_speed)
        return min((self.speed or {}).get(mode, CRUISE[mode]), self.max_speed)

    @classmethod
    def for_scenario(cls, scenario) -> Kinematics:
        """A Scenario's `camera: {standoff: m}` brings the Verifier closer: a real detector
        sees a drone at 30-60 m, not at the stand-in's 100 m. Its `kinematics:` block
        (`speed: {verify: m/s, ...}`, `climb_rate`, `max_speed`) flies a demo faster; what it
        leaves out keeps the defaults."""
        k = getattr(scenario, "kinematics", None) or {}
        speed = {Mode[m.upper()]: float(v) for m, v in (k.get("speed") or {}).items()}
        return cls(standoff=float(scenario.camera.get("standoff", cls.standoff)),
                   speed=speed or None,
                   climb_rate=float(k.get("climb_rate", cls.climb_rate)),
                   max_speed=float(k.get("max_speed", cls.max_speed)))


class _Body:
    def __init__(self, spec: DroneSpec, z: float):
        self.id = spec.id
        self.x, self.y, self.z = spec.x, spec.y, z
        self.heading = 0.0
        self.battery = 1.0
        self.perched_on: str | None = None
        self.rpm = 0.0
        self.cmd = Command(Mode.PATROL, goal=(spec.x, spec.y))


class Twin2D:
    def __init__(self, city: City, drones: list[DroneSpec], kin: Kinematics | None = None):
        self.city = city
        self.kin = kin or Kinematics()
        self.t = 0.0
        self._bodies = {d.id: _Body(d, float(city.ground(d.x, d.y)) + self.kin.cruise_alt)
                        for d in drones}
        self._sites = {p.id: p for p in city.perch_sites}

    # -- Backend interface ---------------------------------------------------------------

    def step(self, dt: float, commands: Mapping[int, Command]) -> dict[int, DroneState]:
        self.t += dt
        for did, cmd in commands.items():
            if did in self._bodies:
                self._bodies[did].cmd = cmd
        for b in self._bodies.values():
            self._advance(b, dt)
        return self.states()

    def camera_frame(self, drone_id: int) -> object | None:
        return None  # the twin renders nothing; sensors.camera uses ground truth instead

    def close(self) -> None:
        pass

    # -- internals -----------------------------------------------------------------------

    def states(self) -> dict[int, DroneState]:
        return {
            b.id: DroneState(
                id=b.id, t=self.t, x=b.x, y=b.y, z=b.z, heading=b.heading, mode=b.cmd.mode,
                perched_on=b.perched_on, battery=b.battery, motor_rpm=b.rpm,
                camera_on=b.cmd.camera_on,
            )
            for b in self._bodies.values()
        }

    def _advance(self, b: _Body, dt: float) -> None:
        cmd, kin = b.cmd, self.kin
        site = self._sites.get(cmd.perch_site) if cmd.mode == Mode.LISTEN else None

        if site is not None and b.perched_on == site.id:
            b.rpm = 0.0
            b.battery = max(0.0, b.battery - kin.drain_perched * dt)
            return

        cruise_z = float(self.city.ground(b.x, b.y)) + kin.cruise_alt  # follow the terrain
        if site is not None:
            goal, goal_z = (site.x, site.y), site.z
        else:
            goal, goal_z = cmd.goal or (b.x, b.y), cruise_z
        b.perched_on = None  # any other command means take off

        dx, dy = goal[0] - b.x, goal[1] - b.y
        dist = math.hypot(dx, dy)
        if cmd.look_at is not None:  # keep a standoff, camera towards the target
            dist = max(0.0, dist - kin.standoff)
        v = kin.v(cmd.mode, cmd.speed)
        step = min(dist, v * dt)
        norm = math.hypot(dx, dy)
        if step > 1e-6 and norm > 1e-6:
            b.x += dx / norm * step
            b.y += dy / norm * step
            b.heading = math.degrees(math.atan2(dx, dy)) % 360.0
        if cmd.look_at is not None:
            b.heading = math.degrees(math.atan2(cmd.look_at[0] - b.x, cmd.look_at[1] - b.y)) % 360.0
        # cruise at altitude, descend only once over the roof
        target_z = goal_z if dist <= kin.arrive_radius else max(goal_z, cruise_z)
        dz = target_z - b.z
        b.z += math.copysign(min(abs(dz), kin.climb_rate * dt), dz)

        b.rpm = kin.cruise_rpm if step > 0.1 * max(v, 1.0) * dt else kin.hover_rpm
        b.battery = max(0.0, b.battery - kin.drain_flying * dt)
        if site is not None and dist <= kin.arrive_radius and abs(b.z - site.z) < 0.5:
            b.perched_on = site.id
            b.z = site.z
            b.rpm = 0.0
