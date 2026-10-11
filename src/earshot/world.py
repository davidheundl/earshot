"""World: one Backend + the Scenario's ground truth + the sensor models, advanced together.

Interface:
    world = World(scenario, city, backend)
    sensed = world.step(commands)     -> {drone_id: Sensed}, advances by scenario.dt
    world.snapshot(telemetry)         -> JSON-able dict for the dashboard / run log

The World is the only module that sees both truth and drones. Runners own the clock.
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
        self.sensors = SensorSuite(city, scenario, np.random.default_rng(scenario.seed + 1))

    def step(self, commands: Mapping[int, Command]) -> dict[int, Sensed]:
        self.states = self.backend.step(self.scenario.dt, commands)
        self.t = next(iter(self.states.values())).t if self.states else self.t + self.scenario.dt
        return {
            did: Sensed(
                state=s,
                readings=self.sensors.sense(s, self.t),
                camera_frame=self.backend.camera_frame(did) if s.camera_on else None,
            )
            for did, s in self.states.items()
        }

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
            "jamming": [
                {"x": j.center[0], "y": j.center[1], "r": j.radius}
                for j in self.scenario.jam_zones(self.t)
            ],
        }
