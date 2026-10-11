"""World: one Backend + the Scenario's ground truth + the sensor models, advanced together.

Interface:
    world = World(scenario, city, backend)
    sensed = world.step(commands)     -> {drone_id: Sensed}, advances by scenario.dt
    world.detections                  -> this step's readings with their truth (logs, metrics)
    world.snapshot(telemetry)         -> JSON-able dict for the dashboard / run log

The World is the only module that sees both truth and drones; Sensed carries readings only,
so Agents cannot see ground truth (ADR 0001). Runners own the clock.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from earshot.backend import Backend
from earshot.city import City
from earshot.scenario import Scenario
from earshot.sensors import SensorSuite
from earshot.state import Command, DroneState, Sensed


class World:
    def __init__(self, scenario: Scenario, city: City, backend: Backend):
        self.scenario = scenario
        self.city = city
        self.backend = backend
        self.t = 0.0
        self.states: dict[int, DroneState] = {}
        self.detections: list[dict] = []
        self._recent: list[dict] = []  # last few seconds of sound bearings, for the dashboard
        self.sensors = SensorSuite(city, scenario, np.random.default_rng(scenario.seed + 1))

    def step(self, commands: Mapping[int, Command]) -> dict[int, Sensed]:
        self.states = self.backend.step(self.scenario.dt, commands)
        self.t = next(iter(self.states.values())).t if self.states else self.t + self.scenario.dt
        self.detections = []
        sensed = {}
        for did, s in self.states.items():
            dets = self.sensors.sense(s, self.t)
            for det in dets:
                r = det.reading
                self.detections.append({
                    "t": round(self.t, 3), "drone": did, "x": s.x, "y": s.y,
                    "mode": s.mode.name, "perched": s.perched_on is not None,
                    "sensor": r.sensor.name, "bearing": r.bearing, "sigma": r.bearing_sigma,
                    "level": r.level, "probs": list(r.class_probs),
                    "truth": det.truth, "snr": det.snr_db,
                })
            sensed[did] = Sensed(
                state=s,
                readings=[d.reading for d in dets],
                camera_frame=self.backend.camera_frame(did) if s.camera_on else None,
            )
        self._recent = [d for d in self._recent if self.t - d["t"] < 3.0] + [
            d for d in self.detections if d["sensor"] == "SOUND"
        ]
        return sensed

    def positions(self) -> dict[int, tuple[float, float, float]]:
        return {d: (s.x, s.y, s.z) for d, s in self.states.items()}

    def snapshot(self, telemetry: Mapping[int, dict] | None = None) -> dict:
        telemetry = telemetry or {}
        return {
            "t": self.t,
            "drones": [
                {
                    "id": s.id, "x": s.x, "y": s.y, "z": s.z, "heading": s.heading,
                    "mode": s.mode.name, "perched_on": s.perched_on, "battery": s.battery,
                    "camera_on": s.camera_on, "telemetry": telemetry.get(s.id, {}),
                }
                for s in self.states.values()
            ],
            "truth": [
                {"id": src.id, "cls": src.cls.name, "x": p[0], "y": p[1]}
                for src, p in self.scenario.active_sources(self.t)
            ],
            "bearings": [
                {"drone": d["drone"], "x": d["x"], "y": d["y"], "bearing": d["bearing"],
                 "sigma": d["sigma"], "age": round(self.t - d["t"], 2)}
                for d in self._recent
            ],
            "environment": vars(self.scenario.environment),
            "jamming": [
                {"x": j.center[0], "y": j.center[1], "r": j.radius}
                for j in self.scenario.jam_zones(self.t)
            ],
        }
