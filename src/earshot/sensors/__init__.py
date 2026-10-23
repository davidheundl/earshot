"""Sensor models: ground truth (Scenario) + drone state -> Detections.

Interface:
    SensorSuite(city, scenario, rng).sense(state, t) -> list[Detection]

A Detection pairs the SensorReading the drone gets with the truth behind it (source id, SNR).
World hands only the readings to Agents and keeps the truth for logs and metrics.

Sound, RF and the camera (only on the Verifier) report once per 1 s window per drone,
staggered across drones. A camera that sees nothing reports that too: "looked, no threat
within range" is what lets a Suspicion be dismissed; a recognised Nuisance (a fan, an
elephant) is no threat, so it only names what was seen instead. Nor is a Source of the
Threat Class that is no intruder: the stand-in assumes the Verifier's classifier knows the
reserve's own marked vehicles (rangers, game drives). The COCO detector does not. Physics lives in `acoustics`
and `rf`; this module only samples from it with the injected rng.

A Shot (a rifle fired, Scenario.shots) reaches each drone once, distance / SPEED_OF_SOUND
after it was fired, as a SHOT reading: one sharp bearing whose class probabilities sit on the
Threat Class, as an RF band's sit on its emitter's (in a Reserve only poachers fire at night).
Shots draw from their own random stream, so a run is the same with or without them until the
first one is heard.

A camera Detection's `label` says what the camera recognised (the stand-in: the Source's
`kind`; the detector: its class names). It stays in the world process, for the Threat Log.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np

from earshot.city import City
from earshot.messages import Mode, Sensor, TargetClass
from earshot.scenario import Scenario
from earshot.sensors import acoustics as ac
from earshot.sensors import rf
from earshot.sensors import waveform as wf
from earshot.sensors.camera import YoloDetector
from earshot.state import DroneState, SensorReading

WINDOW_S = 1.0
# emitter height above ground (m): what the sound and RF line-of-sight tests start from
SOURCE_AGL = {TargetClass.DRONE: 30.0, TargetClass.VEHICLE: 2.0, TargetClass.TRAIN: 3.0,
              TargetClass.NOTHING: 2.0}
SHOT_AGL = 1.5  # m: a rifle fired standing or from a pickup's bed
MASKING_OFFSET_DB = wf.MASKING_OFFSET_DB
LEVEL_NOISE_DB = 2.0
CAMERA_RANGE = 100.0  # wide-angle camera from 40-60 m altitude
CAMERA_P_DETECT = 0.9
CAMERA_HOVER_M = 5.0  # m moved in a window: a Verifier this still is looking, not flying in
CAMERA_MATCH_DEG = 5.0  # a detector bearing this close to a Source is that Source (truth only)

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
        return wf.class_probs(self.classifier.classify(self.mixer.beam(win, i), wf.SR))


@dataclass(frozen=True)
class Detection:
    reading: SensorReading
    truth: str | None  # source id, None for phantom (Fault) readings
    snr_db: float
    label: str | None = None  # camera: what it recognised (world side only, never on the radio)


def _label(src) -> str:
    return src.kind or src.cls.name.lower()


def _probs_on(cls: TargetClass) -> tuple[float, float, float, float]:
    """0.9 on one class (an RF band's emitter, a Shot's Threat Class), the rest spread as for
    the drone links."""
    rest = iter((0.0333, 0.0333, 0.0334))
    return tuple(0.9 if c == cls else next(rest) for c in TargetClass)  # type: ignore[return-value]


class SensorSuite:
    def __init__(self, city: City, scenario: Scenario, rng: np.random.Generator,
                 audio: Audio | None = None, camera: YoloDetector | None = None):
        self.city = city
        self.scenario = scenario
        self.rng = rng
        self.audio = audio
        self.camera = camera  # a trained detector on rendered frames; else the stand-in
        self.confusion = {**CONFUSION, **scenario.confusion}  # the Scenario may override rows
        # camera.range: how far the stand-in sees (a thermal camera sees a vehicle's engine
        # much further than a quadcopter). Then it also geolocates ground targets, from its
        # height and the depression angle: a hit's `level` is the range, a "nothing"'s the
        # radius it cleared. 0: CAMERA_RANGE, bearings only, as at the airport.
        self.camera_reach = float(scenario.camera.get("range", 0.0))
        self.windows: list[wf.AudioWindow] = []  # rendered audio, collected by the World
        self._window: dict[int, int] = {}
        self._cam_drone: int | None = None  # the Verifier the detector's view belongs to
        self._cam_pos: tuple[float, float] | None = None  # where it was at the last verdict
        self._shot_rng = rng.spawn(1)[0]  # Shots apart: they shift no other reading's draws
        self._heard: set[tuple[int, int]] = set()  # (drone, Shot index) whose sound has passed

    def sense(self, s: DroneState, t: float, frame: dict | None = None) -> list[Detection]:
        """`frame`: the Verifier's camera frame when a renderer exists (Webots); its
        recognised objects then decide what the camera sees."""
        faults = {f.kind for f in self.scenario.faults_for(s.id, t)}
        out: list[Detection] = []
        if self.camera is not None:
            self._feed_detector(s, frame)
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
        out += self._shots(s, t, faults)
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
            brg = math.degrees(math.atan2(sx - s.x, sy - s.y)) % 360.0
            levels.append((src, d, brg, ac.received_level(src.level_db, src.cls, d, nb)))
        ego = ac.ego_noise(s.motor_rpm, env.ego_filter, env.ego_filter_db)
        noises = []
        for i in range(len(levels)):
            others = [lv - MASKING_OFFSET_DB for j, (*_, lv) in enumerate(levels) if j != i]
            noises.append(ac.db_sum(env.background_db, ego, *others))
        snrs = [lp - nz for (*_, lp), nz in zip(levels, noises, strict=True)]
        win = None
        if self.audio is not None:
            win = self.audio.mixer.window(s.id, t, [
                wf.Part(src.id, src.cls, lp, d, nb, brg, snr)
                for (src, d, brg, lp), nb, snr in zip(levels, nbs, snrs, strict=True)
            ], ego, env.background_db)
            self.windows.append(win)

        out = []
        for i, (src, _, true_b, lp) in enumerate(levels):
            noise, snr = noises[i], snrs[i]
            if self.rng.random() >= ac.detection_probability(snr, src.cls):
                continue
            sigma = ac.bearing_sigma(src.cls, perched, snr)
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
        row = np.array([self.confusion[true_cls][c] for c in TargetClass])
        w = q * row + (1 - q) * np.full(len(row), 1 / len(row))
        pred = int(self.rng.choice(len(row), p=w / w.sum()))
        conf = 0.4 + 0.5 * q
        rest = (1 - conf) / (len(row) - 1)
        return tuple(conf if c == pred else rest for c in range(len(row)))  # type: ignore[return-value]

    def _shots(self, s: DroneState, t: float, faults: set[str]) -> list[Detection]:
        """Each Shot once per drone, in the first window after its sound arrives. An onset
        detector compares the impulse with the steady floor (background and the drone's own
        motors), not with the other Sources' sum over the window; kopjes and buildings block
        it with SHOT's loss."""
        env = self.scenario.environment
        perched = s.perched_on is not None
        rng = self._shot_rng
        out = []
        for i, shot in enumerate(self.scenario.shots):
            if (s.id, i) in self._heard:
                continue
            d = math.hypot(shot.x - s.x, shot.y - s.y)
            if t < shot.t + d / ac.SPEED_OF_SOUND:
                continue  # still on its way
            self._heard.add((s.id, i))
            if "deaf" in faults:
                continue
            nb = self.city.obstacles_between((s.x, s.y), (shot.x, shot.y), s.z,
                                             float(self.city.ground(shot.x, shot.y)) + SHOT_AGL)
            lp = ac.received_level(shot.level_db, ac.SHOT, d, nb)
            noise = ac.db_sum(env.background_db,
                              ac.ego_noise(s.motor_rpm, env.ego_filter, env.ego_filter_db))
            snr = lp - noise
            if rng.random() >= ac.detection_probability(snr, ac.SHOT):
                continue
            sigma = ac.bearing_sigma(ac.SHOT, perched, snr)
            if "stuck_bearing" in faults:
                b = 0.0
            elif rng.random() < ac.outlier_rate(perched):
                b = rng.uniform(0, 360)
            else:
                b = (math.degrees(math.atan2(shot.x - s.x, shot.y - s.y)) + rng.normal(0, sigma)) % 360.0
            level = ac.db_sum(lp, noise) + rng.normal(0, LEVEL_NOISE_DB)
            r = SensorReading(Sensor.SHOT, b, sigma, _probs_on(self.scenario.threat), level)
            out.append(Detection(r, shot.by or None, snr))
        return out

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
                if band.duty < 1.0 and self.rng.random() >= band.duty:
                    continue  # push-to-talk: nothing on the air this window
                p = rf.mean_rx_power(band, d, los) + self.rng.normal(0, rf.shadowing_sigma(los))
                if self.rng.random() >= rf.detection_probability(band, p):
                    continue
                r = SensorReading(band.sensor, float("nan"), 0.0, _probs_on(band.emitter),
                                  level=p)
                out.append(Detection(r, src.id, p - band.noise_floor_dbm))
        return out

    # -- camera (ground-truth stand-in for the Webots detector, with an error rate) ------

    def _feed_detector(self, s: DroneState, frame: dict | None) -> None:
        """Every rendered tick of a Verifier in VERIFY: the detector sees it. A sortie ends
        (camera off, another Mode) or another drone takes the rig: its view starts afresh,
        so nothing it saw on one sortie can verify the next."""
        verifying = s.camera_on and s.mode == Mode.VERIFY
        if not verifying:
            if self._cam_drone == s.id:
                self.camera.reset()
                self._cam_drone = self._cam_pos = None
            return
        if frame is None or "bgra" not in frame:
            return
        if self._cam_drone != s.id:
            self.camera.reset()
            self._cam_drone, self._cam_pos = s.id, None
        try:
            self.camera.observe(frame)  # every rendered tick; the verdict comes per window
        except Exception as e:  # noqa: BLE001 - a broken model must not stop the World
            print(f"[sensors] camera detector failed ({e!r}); ground-truth stand-in from now")
            self.camera = None

    def _camera(self, s: DroneState, sources, frame: dict | None = None) -> list[Detection]:
        if self.camera is not None:
            # the detector or nothing: a Verifier without the rig's images (another drone
            # has it) stays inconclusive rather than falling back to ground truth
            return self._detected(s, sources) if self._cam_drone == s.id else []
        out = []
        harmless = []  # recognised Nuisances: no threat, only named in the "nothing" reading
        seen = None
        if frame is not None and "objects" in frame:
            seen = {m for m in frame["objects"] if m in TargetClass.__members__}
        for src, (sx, sy) in sources:
            d = math.hypot(sx - s.x, sy - s.y)
            if d > (self.camera_reach or CAMERA_RANGE):
                continue
            if seen is not None:
                if src.cls.name not in seen:
                    continue  # rendered and recognised, or not at all
            elif self.rng.random() >= CAMERA_P_DETECT:
                continue
            if src.cls == TargetClass.NOTHING or (src.cls == self.scenario.threat
                                                  and not src.intruder):
                harmless.append(src)
                continue
            b = math.degrees(math.atan2(sx - s.x, sy - s.y)) % 360.0
            probs = _camera_probs(src.cls)
            ranged = d if self.camera_reach and src.cls != TargetClass.DRONE else 0.0
            # a drone's range from its box size: right up to its unknown real size (one scale
            # per Source, no random draw, so every other number of the run stays the same)
            size = 0.85 + 0.3 * (zlib.crc32(src.id.encode()) % 1000) / 1000.0
            rng = ranged or (d * size if src.cls == TargetClass.DRONE else 0.0)
            out.append(Detection(SensorReading(Sensor.CAMERA, b, 2.0, probs, ranged, rng),
                                 src.id, 99.0, _label(src)))
        if not out:
            out.append(_nothing(self.camera_reach,
                                ", ".join(sorted({_label(h) for h in harmless})) or None))
        return out

    def _detected(self, s: DroneState, sources) -> list[Detection]:
        """The trained detector's verdict for this window: verified tracks as bearings of
        their class (the detector's `classes`), geolocated as the stand-in's are when the
        Scenario's camera ranges ground targets (`camera.range`), "nothing" with the radius it
        really cleared, or no reading when inconclusive. Truth (logs and metrics only) is the
        active Source nearest in bearing, within 5 deg."""
        moved = (math.inf if self._cam_pos is None
                 else math.hypot(s.x - self._cam_pos[0], s.y - self._cam_pos[1]))
        self._cam_pos = (s.x, s.y)
        # an empty view only counts towards "nothing" from where the Verifier holds still: on
        # the way in it has not yet looked at the place
        v = self.camera.verdict(hovering=moved <= CAMERA_HOVER_M)
        out = []
        for b, sigma, _, label, ground, box in v.hits:
            truth, best = None, CAMERA_MATCH_DEG
            for src, (sx, sy) in sources:
                d = math.hypot(sx - s.x, sy - s.y)
                err = abs((math.degrees(math.atan2(sx - s.x, sy - s.y)) - b + 180.0) % 360.0
                          - 180.0)
                if d <= (self.camera_reach or CAMERA_RANGE) and err <= best:
                    truth, best = src.id, err
            cls = self.camera.class_of(label)
            probs = _camera_probs(cls)
            # a ground target geolocated (-> level, the Belief's point); a drone ranged by its
            # box size (-> range, the Follower's only)
            ranged = ground if self.camera_reach and cls != TargetClass.DRONE else 0.0
            rng = ranged or (box if cls == TargetClass.DRONE else 0.0)
            out.append(Detection(SensorReading(Sensor.CAMERA, b, sigma, probs, ranged, rng), truth,
                                 99.0, label))
        if not out and v.clear_radius is not None:
            out.append(_nothing(v.clear_radius, ", ".join(v.others) or None))
        return out



def _camera_probs(cls: TargetClass) -> tuple[float, ...]:
    """A camera sighting's class probabilities (not _probs_on's split: other bytes)."""
    return tuple(0.95 if c == cls else 0.05 / 3 for c in TargetClass)


def _nothing(radius: float, label: str | None) -> Detection:
    """The camera's "nothing here" out to `radius`, naming the harmless things it saw."""
    return Detection(SensorReading(Sensor.CAMERA, float("nan"), 0.0, (0.0, 0.0, 0.0, 1.0), radius),
                     None, 0.0, label)


def build_perception(scenario: Scenario, *, classifier: str | None = None,
                     sounds: str | None = None, audio: bool = False,
                     camera: str | None = None, renders: bool = False,
                     log=print) -> tuple[Audio | None, YoloDetector | None]:
    """The trained models a run uses: the Scenario's `audio:` and `camera:`, overridden by the
    command line (`classifier` "none" and `camera` "truth" turn them off). The camera detector
    needs rendered images (`renders`: Webots). A model that cannot load (no torch or
    ultralytics, missing weights) is reported and replaced by its stand-in, never fatal."""
    clf_spec = classifier if classifier is not None else scenario.audio.get("classifier")
    clf_spec = None if clf_spec in (None, "", "none") else clf_spec
    clf = None
    if clf_spec:
        try:
            clf = wf.load_classifier(clf_spec)
        except Exception as e:  # noqa: BLE001 - a head for other classes too: never a dead World
            log(f"[sensors] sound classifier {clf_spec} unavailable ({e}); CONFUSION stand-in")
    snd = None
    if audio or clf is not None:
        from pathlib import Path

        root = sounds or scenario.audio.get("sounds", "data/sounds")
        snd = Audio(wf.Mixer(wf.ClipBank(Path(root))), clf)

    spec = dict(scenario.camera)
    if camera is not None:
        spec["detector"] = camera
    det = None
    if not renders and camera == "yolo":
        log("[sensors] --camera yolo needs rendered images (--backend webots); stand-in")
    if renders and spec.get("detector") == "yolo":
        from earshot.sensors.camera import load_detector

        det = load_detector(spec)
        try:
            det.warm(*spec.get("resolution", (1280, 768)))
        except Exception as e:  # noqa: BLE001 - any failure: the stand-in, not a dead World
            log(f"[sensors] camera detector unavailable ({e}); ground-truth stand-in")
            det = None
    return snd, det
