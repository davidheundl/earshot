"""Sensor models: ground truth (Scenario) + drone state -> Detections.

Interface:
    SensorSuite(city, scenario, rng).sense(state, t) -> list[Detection]

A Detection pairs the SensorReading the drone gets with the truth behind it (source id, SNR).
World hands only the readings to Agents and keeps the truth for logs and metrics.

Sound, RF and the camera (only on the Verifier) report once per 1 s window per drone,
staggered across drones. A camera that sees nothing reports that too: "looked, no threat
within range" is what lets a Suspicion be dismissed. Physics lives in `acoustics` and `rf`;
this module only samples from it with the injected rng.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from earshot.city import City
from earshot.messages import Mode, Sensor, TargetClass
from earshot.scenario import Scenario
from earshot.sensors import acoustics as ac
from earshot.sensors import rf
from earshot.sensors import waveform as wf
from earshot.state import DroneState, SensorReading

WINDOW_S = 1.0
# emitter height above ground (m): what the sound and RF line-of-sight tests start from
SOURCE_AGL = {TargetClass.DRONE: 30.0, TargetClass.VEHICLE: 2.0, TargetClass.TRAIN: 3.0,
              TargetClass.NOTHING: 2.0}
MASKING_OFFSET_DB = wf.MASKING_OFFSET_DB
LEVEL_NOISE_DB = 2.0
CAMERA_RANGE = 100.0  # wide-angle camera from 40-60 m altitude
CAMERA_P_DETECT = 0.9

# Placeholder for the EfficientAT head: true class -> predicted class weights. The fan
# (NOTHING) is mistaken for a drone unless trained as a negative (section 5.1).
_C = TargetClass
CONFUSION = {
    _C.DRONE: {_C.DRONE: 0.85, _C.VEHICLE: 0.03, _C.TRAIN: 0.02, _C.NOTHING: 0.10},
    _C.VEHICLE: {_C.DRONE: 0.03, _C.VEHICLE: 0.80, _C.TRAIN: 0.15, _C.NOTHING: 0.02},
    _C.TRAIN: {_C.DRONE: 0.01, _C.VEHICLE: 0.14, _C.TRAIN: 0.83, _C.NOTHING: 0.02},
    _C.NOTHING: {_C.DRONE: 0.35, _C.VEHICLE: 0.05, _C.TRAIN: 0.00, _C.NOTHING: 0.60},
}


@dataclass
class Audio:
    """Waveform rendering for the sound sensor (off unless the World gets one): every sound
    window is mixed as audio, and with a classifier its beams decide the class probabilities
    instead of CONFUSION."""

    mixer: wf.Mixer
    classifier: wf.Classifier | None = None

    def classify(self, win: wf.AudioWindow, i: int) -> tuple[float, float, float, float]:
        p = np.clip(np.asarray(self.classifier.classify(self.mixer.beam(win, i), wf.SR),
                               float), 0.0, None)
        return tuple(p / p.sum()) if p.sum() > 0 else (0.25,) * 4  # type: ignore[return-value]


@dataclass(frozen=True)
class Detection:
    reading: SensorReading
    truth: str | None  # source id, None for phantom (Fault) readings
    snr_db: float


class SensorSuite:
    def __init__(self, city: City, scenario: Scenario, rng: np.random.Generator,
                 audio: Audio | None = None):
        self.city = city
        self.scenario = scenario
        self.rng = rng
        self.audio = audio
        self.windows: list[wf.AudioWindow] = []  # rendered audio, collected by the World
        self._window: dict[int, int] = {}

    def sense(self, s: DroneState, t: float, frame: dict | None = None) -> list[Detection]:
        """`frame`: the Verifier's camera frame when a renderer exists (Webots); its
        recognised objects then decide what the camera sees."""
        faults = {f.kind for f in self.scenario.faults_for(s.id, t)}
        out: list[Detection] = []
        if not self._new_window(s.id, t):
            return out
        sources = self.scenario.active_sources(t)
        if s.camera_on and s.mode == Mode.VERIFY:
            out += self._camera(s, sources, frame)
        if "deaf" not in faults:
            # buildings in the way, per source: sound and RF share the line of sight
            nbs = [self.city.obstacles_between((s.x, s.y), (sx, sy), s.z, self._z(src.cls, sx, sy))
                   for src, (sx, sy) in sources]
            out += self._sound(s, t, sources, nbs, faults)
            out += self._rf(s, sources, nbs)
            if "phantom" in faults and self.rng.random() < 0.3:
                r = SensorReading(Sensor.SOUND, self.rng.uniform(0, 360), 5.0,
                                  self._class_probs(_C.DRONE, 5.0), level=45.0)
                out.append(Detection(r, None, 5.0))
        return out

    def _z(self, cls: TargetClass, x: float, y: float) -> float:
        return float(self.city.ground(x, y)) + SOURCE_AGL[cls]

    # -- windows -------------------------------------------------------------------------

    def _new_window(self, drone: int, t: float) -> bool:
        w = math.floor((t + 0.137 * drone) / WINDOW_S)
        if self._window.get(drone) == w:
            return False
        self._window[drone] = w
        return True

    # -- sound ---------------------------------------------------------------------------

    def _sound(self, s: DroneState, t: float, sources, nbs, faults) -> list[Detection]:
        env = self.scenario.environment
        perched = s.perched_on is not None
        levels = []
        for (src, (sx, sy)), nb in zip(sources, nbs, strict=True):
            d = math.hypot(sx - s.x, sy - s.y)
            levels.append((src, sx, sy, ac.received_level(src.level_db, src.cls, d, nb)))
        ego = ac.ego_noise(s.motor_rpm, env.ego_filter, env.ego_filter_db)
        noises = []
        for i in range(len(levels)):
            others = [lv - MASKING_OFFSET_DB for j, (*_, lv) in enumerate(levels) if j != i]
            noises.append(ac.db_sum(env.background_db, ego, *others))
        snrs = [lp - nz for (*_, lp), nz in zip(levels, noises, strict=True)]
        win = None
        if self.audio is not None:
            win = self.audio.mixer.window(s.id, t, [
                wf.Part(src.id, src.cls, lp, math.hypot(sx - s.x, sy - s.y), nb,
                        math.degrees(math.atan2(sx - s.x, sy - s.y)) % 360.0, snr)
                for (src, sx, sy, lp), nb, snr in zip(levels, nbs, snrs, strict=True)
            ], ego, env.background_db)
            self.windows.append(win)

        out = []
        for i, (src, sx, sy, lp) in enumerate(levels):
            noise, snr = noises[i], snrs[i]
            if self.rng.random() >= ac.detection_probability(snr, src.cls):
                continue
            sigma = ac.bearing_sigma(src.cls, perched, snr)
            true_b = math.degrees(math.atan2(sx - s.x, sy - s.y)) % 360.0
            if "stuck_bearing" in faults:
                b = 0.0
            elif self.rng.random() < ac.outlier_rate(perched):
                b = self.rng.uniform(0, 360)
            else:
                b = (true_b + self.rng.normal(0, sigma)) % 360.0
            level = ac.db_sum(lp, noise) + self.rng.normal(0, LEVEL_NOISE_DB)
            if win is not None and self.audio.classifier is not None:
                probs = self.audio.classify(win, i)
            else:
                probs = self._class_probs(src.cls, snr)
            r = SensorReading(Sensor.SOUND, b, sigma, probs, level)
            out.append(Detection(r, src.id, snr))
        return out

    def _class_probs(self, true_cls: TargetClass, snr: float) -> tuple[float, float, float, float]:
        q = 1.0 / (1.0 + math.exp(-(snr - ac.ACOUSTICS[true_cls].snr50_db) / 3.0))
        row = np.array([CONFUSION[true_cls][c] for c in TargetClass])
        w = q * row + (1 - q) * np.full(len(row), 1 / len(row))
        pred = int(self.rng.choice(len(row), p=w / w.sum()))
        conf = 0.4 + 0.5 * q
        rest = (1 - conf) / (len(row) - 1)
        return tuple(conf if c == pred else rest for c in range(len(row)))  # type: ignore[return-value]

    # -- RF ------------------------------------------------------------------------------

    def _rf(self, s: DroneState, sources, nbs) -> list[Detection]:
        out = []
        for (src, (sx, sy)), nb in zip(sources, nbs, strict=True):
            if not src.rf_bands:
                continue
            d = math.hypot(sx - s.x, sy - s.y)
            los = nb == 0
            for key in src.rf_bands:
                band = rf.BANDS[key]
                p = rf.mean_rx_power(band, d, los) + self.rng.normal(0, rf.shadowing_sigma(los))
                if self.rng.random() >= rf.detection_probability(band, p):
                    continue
                r = SensorReading(band.sensor, float("nan"), 0.0,
                                  (0.9, 0.0333, 0.0333, 0.0334), level=p)
                out.append(Detection(r, src.id, p - band.noise_floor_dbm))
        return out

    # -- camera (ground-truth stand-in for the Webots detector, with an error rate) ------

    def _camera(self, s: DroneState, sources, frame: dict | None = None) -> list[Detection]:
        out = []
        seen = None
        if frame is not None and "objects" in frame:
            seen = {m for m in frame["objects"] if m in TargetClass.__members__}
        for src, (sx, sy) in sources:
            d = math.hypot(sx - s.x, sy - s.y)
            if d > CAMERA_RANGE:
                continue
            if seen is not None:
                if src.cls.name not in seen:
                    continue  # rendered and recognised, or not at all
            elif self.rng.random() >= CAMERA_P_DETECT:
                continue
            b = math.degrees(math.atan2(sx - s.x, sy - s.y)) % 360.0
            probs = tuple(0.95 if c == src.cls else 0.05 / 3 for c in TargetClass)
            out.append(Detection(SensorReading(Sensor.CAMERA, b, 2.0, probs, 0.0), src.id, 99.0))
        if not out:
            nothing = (0.0, 0.0, 0.0, 1.0)
            out.append(Detection(SensorReading(Sensor.CAMERA, float("nan"), 0.0, nothing, 0.0),
                                 None, 0.0))
        return out
