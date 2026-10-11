"""Webots adapter: the 2D twin moves the drones, Webots shows them and renders one camera.

ADR 0002. Drones are plain Pose nodes (DEF DRONE_<id>) with no physics and no controller; a
single Supervisor robot (DEF RIG, controller <extern>) moves them every basic time step. A
Supervisor cannot read devices on other robots, so the RIG itself carries the one Camera and
flies along with whichever drone has camera_on (the Verifier). Sources are puppeted the same
way (DEF SRC_<id>), parked below ground while inactive.

Must run inside a Webots extern-controller environment (see `earshot live --backend webots`).
Frames: City x = east, y = north, metres  ==  Webots ENU world (z up).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from earshot.backend.twin2d import Twin2D
from earshot.city import City
from earshot.scenario import Scenario
from earshot.state import Command, DroneState

PARKED = [0.0, 0.0, -100.0]
SOURCE_Z = {"DRONE": 30.0, "VEHICLE": 1.5, "TRAIN": 2.0}


def _yaw(heading_deg: float) -> float:
    """Compass heading (0 = north, clockwise) -> rotation about +z from +x (counter-clockwise)."""
    return math.radians(90.0 - heading_deg)


class WebotsBackend:
    def __init__(self, city: City, scenario: Scenario):
        from controller import Supervisor  # only importable inside the Webots env

        self.sup = Supervisor()
        self.basic_ms = int(self.sup.getBasicTimeStep())
        self.scenario = scenario
        self.twin = Twin2D(city, scenario.drones)
        self._drones = {}
        for d in scenario.drones:
            node = self.sup.getFromDef(f"DRONE_{d.id}")
            if node is None:
                raise RuntimeError(f"world has no DEF DRONE_{d.id}; regenerate it")
            self._drones[d.id] = (node.getField("translation"), node.getField("rotation"))
        self._sources = {}
        for s in scenario.sources:
            node = self.sup.getFromDef(f"SRC_{s.id}")
            if node is not None:
                self._sources[s.id] = node.getField("translation")
        rig = self.sup.getSelf()
        self._rig = (rig.getField("translation"), rig.getField("rotation"))
        self.camera = self.sup.getDevice("camera")
        self._verifier: int | None = None

    # -- Backend interface ---------------------------------------------------------------

    def step(self, dt: float, commands: Mapping[int, Command]) -> dict[int, DroneState]:
        n = max(1, round(dt * 1000 / self.basic_ms))
        states = {}
        for i in range(n):
            states = self.twin.step(dt / n, commands if i == 0 else {})
            self._push(states)
            if self.sup.step(self.basic_ms) == -1:
                raise SystemExit("Webots quit")
        return states

    def camera_frame(self, drone_id: int) -> object | None:
        if drone_id != self._verifier:
            return None
        c = self.camera
        frame = {"w": c.getWidth(), "h": c.getHeight(), "bgra": c.getImage()}
        if c.hasRecognition() and c.hasRecognitionSegmentation():
            frame["seg"] = c.getRecognitionSegmentationImage()
            frame["objects"] = [o.getModel() for o in c.getRecognitionObjects()]
        return frame

    def close(self) -> None:
        pass

    # -- internals -----------------------------------------------------------------------

    def _push(self, states: dict[int, DroneState]) -> None:
        for did, s in states.items():
            tf, rf = self._drones[did]
            tf.setSFVec3f([s.x, s.y, s.z])
            rf.setSFRotation([0.0, 0.0, 1.0, _yaw(s.heading)])

        t = self.twin.t
        for src in self.scenario.sources:
            field = self._sources.get(src.id)
            if field is None:
                continue
            p = src.position(t)
            field.setSFVec3f(PARKED if p is None else [p[0], p[1], SOURCE_Z[src.cls.name]])

        verifier = next((d for d, s in states.items() if s.camera_on), None)
        if verifier != self._verifier:
            if verifier is None:
                self.camera.disable()
                if self.camera.hasRecognition():
                    self.camera.recognitionDisable()
            elif self._verifier is None:
                self.camera.enable(self.basic_ms)
                if self.camera.hasRecognition():
                    self.camera.recognitionEnable(self.basic_ms)
                    self.camera.enableRecognitionSegmentation()
            self._verifier = verifier
        rt, rr = self._rig
        if verifier is None:
            rt.setSFVec3f(PARKED)
        else:
            s = states[verifier]
            rt.setSFVec3f([s.x, s.y, s.z - 0.3])
            rr.setSFRotation([0.0, 0.0, 1.0, _yaw(s.heading)])
