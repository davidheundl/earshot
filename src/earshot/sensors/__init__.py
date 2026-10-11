"""Sensor models: ground truth (Scenario) + drone state -> SensorReadings.

Interface:
    SensorSuite(city, scenario, rng).sense(state, t) -> list[SensorReading]

PLACEHOLDER (pre-Oct 16): fixed hearing ranges per class and mode with Gaussian bearing noise.
To be replaced by the propagation model of concept section 9 (ISO 9613 spreading, air
absorption, -15 dB per building, ego-noise by motor_rpm, RF log-distance rings). Keep the
interface; only the internals change.
"""

from __future__ import annotations

import math

import numpy as np

from earshot.city import City
from earshot.messages import Mode, Sensor, TargetClass
from earshot.scenario import Scenario
from earshot.state import DroneState, SensorReading

# metres; (perched, flying). Concept section 5.1, night column, rounded.
_HEARING_RANGE = {
    TargetClass.DRONE: (150.0, 5.0),
    TargetClass.VEHICLE: (1300.0, 60.0),
    TargetClass.TRAIN: (3000.0, 300.0),
}
_BEARING_SIGMA = {True: 5.0, False: 25.0}  # perched / flying, degrees
_OUTLIER = {True: 0.05, False: 0.25}
_RF_RANGE = 400.0  # 2.4 GHz video link in a city
_CAMERA_RANGE = 60.0


class SensorSuite:
    def __init__(self, city: City, scenario: Scenario, rng: np.random.Generator):
        self.city = city
        self.scenario = scenario
        self.rng = rng

    def sense(self, s: DroneState, t: float) -> list[SensorReading]:
        faults = {f.kind for f in self.scenario.faults_for(s.id, t)}
        if "deaf" in faults:
            return []
        perched = s.perched_on is not None
        out: list[SensorReading] = []
        for src, (sx, sy) in self.scenario.active_sources(t):
            d = math.hypot(sx - s.x, sy - s.y)
            true_bearing = math.degrees(math.atan2(sx - s.x, sy - s.y)) % 360.0

            # sound
            rng_hear = _HEARING_RANGE[src.cls][0 if perched else 1]
            if self.city.buildings_between((s.x, s.y), (sx, sy)):
                rng_hear *= 0.3
            if d <= rng_hear:
                sigma = _BEARING_SIGMA[perched]
                if self.rng.random() < _OUTLIER[perched]:
                    b = self.rng.uniform(0, 360)
                elif "stuck_bearing" in faults:
                    b = 0.0
                else:
                    b = (true_bearing + self.rng.normal(0, sigma)) % 360.0
                out.append(SensorReading(Sensor.SOUND, b, sigma, _probs(src.cls, 0.7),
                                         level=src.level_db - 20 * math.log10(max(d, 1)) - 11))

            # RF energy (drones with a link only); no bearing, level gives a rough ring
            if src.rf and d <= _RF_RANGE:
                p = -40 - 27 * math.log10(max(d, 1)) + self.rng.normal(0, 6)
                out.append(SensorReading(Sensor.RF, float("nan"), 0.0,
                                         _probs(TargetClass.DRONE, 0.6), level=p))

            # camera (ground truth stand-in for the Webots detector)
            if s.camera_on and s.mode == Mode.VERIFY and d <= _CAMERA_RANGE:
                out.append(SensorReading(Sensor.CAMERA, true_bearing, 2.0,
                                         _probs(src.cls, 0.95), level=0.0))

        if "phantom" in faults and self.rng.random() < 0.2:
            out.append(SensorReading(Sensor.SOUND, self.rng.uniform(0, 360), 5.0,
                                     _probs(TargetClass.DRONE, 0.7), level=50.0))
        return out


def _probs(cls: TargetClass, p: float) -> tuple[float, float, float, float]:
    rest = (1.0 - p) / 3.0
    return tuple(p if c == cls else rest for c in TargetClass)  # type: ignore[return-value]
