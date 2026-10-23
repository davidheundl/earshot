"""World: one Backend + the Scenario's ground truth + the sensor models, advanced together.

Interface:
    world = World(scenario, city, backend)
    sensed = world.step(commands)     -> {drone_id: Sensed}, advances by scenario.dt
    world.detections                  -> this step's readings with their truth (logs, metrics)
    world.snapshot(telemetry)         -> JSON-able dict for the dashboard / run log
    world.audio                       -> this step's sound windows as audio (with `audio=`);
                                         snapshot() then carries each drone's sound spectrum

The World is the only module that sees both truth and drones; Sensed carries readings only,
so Agents cannot see ground truth (ADR 0001). Runners own the clock.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from earshot.backend import Backend
from earshot.city import City
from earshot.messages import Mode
from earshot.scenario import Scenario
from earshot.sensors import Audio, SensorSuite
from earshot.sensors.camera import YoloDetector
from earshot.sensors.waveform import spectrum
from earshot.state import Command, DroneState, Sensed

GUIDE_RADIUS = 600.0  # m around a Verifier's goal its Guide looks for the intruder


class World:
    def __init__(self, scenario: Scenario, city: City, backend: Backend,
                 audio: Audio | None = None, camera: YoloDetector | None = None):
        self.scenario = scenario
        self.city = city
        self.backend = backend
        self.t = 0.0
        self.states: dict[int, DroneState] = {}
        self.detections: list[dict] = []
        self.camera: tuple[int, dict] | None = None  # (drone, frame) of the latest Verifier view
        self._preview: dict | None = None  # last dashboard camera preview, kept when it ends
        self.applied: list[dict] = []  # every event that took effect, with its sim time
        self._pending: list[dict] = []
        self._recent: list[dict] = []  # last few seconds of sound bearings, for the dashboard
        self.sensors = SensorSuite(city, scenario, np.random.default_rng(scenario.seed + 1),
                                   audio, camera)
        if camera is not None and hasattr(backend, "detector_frames"):
            backend.detector_frames = True  # the detector reads every Verifier image
        self.audio: list = []  # sensors.waveform.AudioWindow rendered this step
        self._spectra: dict[int, list[int]] = {}  # drone -> its newest window's spectrum

    def control(self, op: dict) -> None:
        """Queue a live event (dashboard hotkey); it takes effect at the next step."""
        self._pending.append(op)

    def _apply_events(self) -> None:
        live = [dict(op, t=self.t, live=True) for op in self._pending]
        due = self.scenario.due_events(self.t) + live
        self._pending = []
        for op in due:
            try:
                note = self.scenario.apply(op, self.t)
            except Exception as e:  # noqa: BLE001 - one bad Event must never stop the world
                note = f"rejected {op.get('op')}: {e}"
            self.applied.append({**op, "t": round(self.t, 3), "note": note})

    def _guide(self, cmd: Command | None) -> tuple[float, float] | None:
        """Demo mock (Scenario `guide: true`): the true position of the intruder nearest to
        where a Verifier flies, within GUIDE_RADIUS; a Verifier sent to a Ghost far from any
        intruder gets nothing and is dismissed as usual."""
        if not self.scenario.guide or cmd is None or cmd.mode != Mode.VERIFY or cmd.goal is None:
            return None
        near = [(math.hypot(p[0] - cmd.goal[0], p[1] - cmd.goal[1]), p)
                for src, p in self.scenario.active_sources(self.t) if src.intruder]
        d, p = min(near, default=(math.inf, None), key=lambda e: e[0])
        return (float(p[0]), float(p[1])) if d <= GUIDE_RADIUS else None

    def step(self, commands: Mapping[int, Command]) -> dict[int, Sensed]:
        self._apply_events()
        self.states = self.backend.step(self.scenario.dt, commands)
        self.t = next(iter(self.states.values())).t if self.states else self.t + self.scenario.dt
        self.detections = []
        self.sensors.windows = self.audio = []
        sensed = {}
        for did, s in self.states.items():
            frame = self.backend.camera_frame(did) if s.camera_on else None
            if frame is not None:
                self.camera = (did, frame)
            dets = self.sensors.sense(s, self.t, frame)
            for det in dets:
                r = det.reading
                self.detections.append({
                    "t": round(self.t, 3), "drone": did, "x": s.x, "y": s.y,
                    "mode": s.mode.name, "perched": s.perched_on is not None,
                    "sensor": r.sensor.name, "bearing": r.bearing, "sigma": r.bearing_sigma,
                    "level": r.level, "probs": list(r.class_probs),
                    "truth": det.truth, "snr": det.snr_db,
                    **({"label": det.label} if det.label else {}),
                })
            sensed[did] = Sensed(state=s, readings=[d.reading for d in dets], camera_frame=frame,
                                 guide=self._guide(commands.get(did)))
        if self.sensors.camera is None and getattr(self.backend, "detector_frames", False):
            self.backend.detector_frames = False  # the detector broke: the stand-in's view
        for w in self.audio:
            self._spectra[w.drone] = spectrum(w.mix)
        self._recent = [d for d in self._recent if self.t - d["t"] < 3.0] + [
            d for d in self.detections if d["sensor"] in ("SOUND", "SHOT")
        ]
        return sensed

    def positions(self) -> dict[int, tuple[float, float, float]]:
        return {d: (s.x, s.y, s.z) for d, s in self.states.items()}

    def camera_preview(self) -> dict | None:
        """The latest Verifier frame, halved and as RGBA bytes, for the dashboard. The image is
        fetched here, not every step, when the backend renders on demand (`camera_image`)."""
        if self.camera is None:
            return None
        did, f = self.camera
        s = self.states.get(did)
        if "bgra" not in f and s is not None and s.camera_on and hasattr(self.backend,
                                                                           "camera_image"):
            f["bgra"] = self.backend.camera_image()
        if not f.get("bgra"):
            return self._preview
        img = np.frombuffer(f["bgra"], dtype=np.uint8).reshape(f["h"], f["w"], 4)[::2, ::2]
        rgba = img[..., [2, 1, 0, 3]].copy()
        rgba[..., 3] = 255
        self._preview = {"drone": did, "w": rgba.shape[1], "h": rgba.shape[0],
                         "rgba": rgba.tobytes(), "objects": f.get("objects", [])}
        return self._preview

    def snapshot(self, telemetry: Mapping[int, dict] | None = None) -> dict:
        telemetry = telemetry or {}
        return {
            "t": self.t,
            "drones": [
                {
                    "id": s.id, "x": s.x, "y": s.y, "z": s.z, "heading": s.heading,
                    "agl": round(s.z - float(self.city.ground(s.x, s.y)), 1),
                    "mode": s.mode.name, "perched_on": s.perched_on, "battery": s.battery,
                    "camera_on": s.camera_on, "telemetry": telemetry.get(s.id, {}),
                    **({"spectrum": self._spectra[s.id]} if s.id in self._spectra else {}),
                }
                for s in self.states.values()
            ],
            "truth": [
                {"id": src.id, "cls": src.cls.name, "x": p[0], "y": p[1],
                 **({"kind": src.kind} if src.kind else {})}
                for src, p in self.scenario.active_sources(self.t)
            ],
            "bearings": [
                {"drone": d["drone"], "x": d["x"], "y": d["y"], "bearing": d["bearing"],
                 "sigma": d["sigma"], "age": round(self.t - d["t"], 2)}
                for d in self._recent
            ],
            "readings": [  # what the drones measured this step (no ground truth); RF has no bearing
                {**{k: d[k] for k in ("t", "drone", "sensor", "sigma", "level", "probs", "mode")},
                 "bearing": None if d["bearing"] != d["bearing"] else round(d["bearing"], 1)}
                for d in self.detections
            ],
            "environment": vars(self.scenario.environment),
            "events": self.applied[-5:],
            "jamming": [
                {"x": j.center[0], "y": j.center[1], "r": j.radius}
                for j in self.scenario.jam_zones(self.t)
            ],
        }
