"""Quarry: where the threat a Follower follows is now, and how fast it moves.

Interface:
    q = Quarry(t, x, y)                     at the confirmed point, standing still
    q.sight(t, sx, sy, bearing, rng)        own camera hit -> bool (False: gated out)
    q.track(t, x, y)                        the Confirmed Track's point (the swarm's estimate)
    q.guide(t, x, y)                        demo mock: where it really is (Sensed.guide)
    q.at(t) -> (x, y)                       predicted position now
    q.vx, q.vy, q.speed                     velocity, m/s
    q.last_seen                             t of the newest sighting or Confirmed Track

An alpha-beta filter (constant velocity) with irregular updates: the camera reports at most
once per 1 s window and the Confirmed Track once per extraction, while the Policy commands
every tick, so between updates the Quarry coasts on its velocity.

- A ranged sighting (SensorReading.range: the thermal camera geolocated a vehicle, or a
  drone's box size gave its distance) is a point fix. A bearing only borrows its range from
  the prediction: the point on the bearing ray nearest to where the Quarry should be; range
  then comes from the swarm's Track and the camera corrects the direction. Chasing a drone
  that flies away, that range is what the camera cannot see: a bearing alone falls behind.
- The Confirmed Track updates with lower gains: at the airport it is the fused sound and RF
  peak, noisier than the camera, and it lags (belief.CONFIRM_SPEED). For a drone it is still
  the only range there is; after a ranged sighting (CAMERA_FRESH) it is not needed.
- A bearing's borrowed range is at most CAMERA_RANGE: the camera saw it, so it is that close.
- The prediction coasts at most COAST seconds on the velocity: a drone that turned is not
  chased in a straight line out of the camera's reach.
- A fix farther than GATE from the prediction is someone else: it is ignored. The gate
  widens while the Quarry coasts, so a Quarry that was lost for a while can be found again.
- A Confirmed Track still near counts as seen, even when it adds nothing new: the swarm
  still believes in the threat.

Pure: no clock, no randomness; time arrives as `t`.
"""

from __future__ import annotations

import math

from earshot.agent.belief import CAMERA_RANGE

CAMERA_GAINS = (0.6, 0.3)  # alpha (position), beta (velocity) for an own-camera fix
TRACK_GAINS = (0.3, 0.08)  # for a Confirmed Track update
GATE = 120.0  # m: a fix this far from the prediction is not the Quarry
GATE_PER_S = 15.0  # m the gate widens per second since the last update
MAX_SPEED = 40.0  # m/s the estimate is clamped to: a bad fix must not fling the Follower off
CAMERA_FRESH = 3.0  # s after a ranged sighting the Confirmed Track is not needed
COAST = 3.0  # s the prediction runs on the velocity at most
GUIDE_SMOOTH = 0.5  # s over which a guided velocity follows the true one


class Quarry:
    def __init__(self, t: float, x: float, y: float):
        self.x, self.y = x, y  # at self.t
        self.vx = self.vy = 0.0
        self.t = t
        self.last_seen = t
        self.last_sighting: float | None = None  # newest own-camera fix
        self._last_ranged: float | None = None  # newest one that carried a range
        self._track_xy: tuple[float, float] | None = None  # the last Track point used

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def at(self, t: float) -> tuple[float, float]:
        dt = min(max(0.0, t - self.t), COAST)
        return self.x + self.vx * dt, self.y + self.vy * dt

    def sight(self, t: float, sx: float, sy: float, bearing: float, rng: float) -> bool:
        b = math.radians(bearing)
        ux, uy = math.sin(b), math.cos(b)  # compass bearing: 0 = north, clockwise
        ranged = rng > 0.0
        if not ranged:
            px, py = self.at(t)
            rng = min(max(1.0, (px - sx) * ux + (py - sy) * uy), CAMERA_RANGE)
        if not self._update(t, sx + ux * rng, sy + uy * rng, CAMERA_GAINS):
            return False
        self.last_sighting = t
        if ranged:
            self._last_ranged = t
        return True

    def track(self, t: float, x: float, y: float) -> bool:
        self.last_seen = t
        if self._track_xy == (x, y):
            return False  # the same extraction again: no new information
        self._track_xy = (x, y)
        if self._last_ranged is not None and t - self._last_ranged < CAMERA_FRESH:
            return False  # the camera ranges it: the swarm's point adds only its error
        return self._update(t, x, y, TRACK_GAINS)

    def guide(self, t: float, x: float, y: float) -> None:
        """The demo mock's true position: taken as it is, velocity from the last one."""
        dt = t - self.t
        if dt > 1e-6:
            a = min(1.0, dt / GUIDE_SMOOTH)
            self.vx += a * ((x - self.x) / dt - self.vx)
            self.vy += a * ((y - self.y) / dt - self.vy)
        self.x, self.y, self.t = x, y, t
        self.last_seen = t

    def _update(self, t: float, zx: float, zy: float, gains: tuple[float, float]) -> bool:
        px, py = self.at(t)
        dt = t - self.t
        if math.hypot(zx - px, zy - py) > GATE + GATE_PER_S * max(0.0, dt):
            return False
        alpha, beta = gains
        rx, ry = zx - px, zy - py
        self.x, self.y = px + alpha * rx, py + alpha * ry
        if dt > 1e-6:
            self.vx += beta * rx / dt
            self.vy += beta * ry / dt
            v = self.speed
            if v > MAX_SPEED:
                self.vx, self.vy = self.vx * MAX_SPEED / v, self.vy * MAX_SPEED / v
        self.t = t
        self.last_seen = t
        return True
