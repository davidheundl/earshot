"""Gazebo adapter: the 2D twin moves the drones, Gazebo Sim shows them and renders the cameras.

ADR 0008 (the Gazebo twin of ADR 0002). Drones, Sources and cameras are static models
(DRONE_<id>, SRC_<id>[_w<k>], CAMV, CAM_<k>) in a world written by
tools/build_gazebo_world.py. Nothing in Gazebo flies on its own: every basic step this adapter
sends the poses of everything that moves, and Gazebo steps once.

Gazebo's Python bindings exist only for Homebrew's Python (3.12+), not for our 3.11, so a
separate process, `gz_bridge.py`, embeds the Gazebo server and talks to this adapter over one
ZeroMQ REQ/REP socket (msgpack). The bridge steps Gazebo in lockstep with us and paces each
basic step to its wall-clock deadline, like WebotsBackend in realtime mode. The transport is
injectable (`link`), so tests run without Gazebo.

Cameras: the Verifier camera (CAMV) is a bounding-box camera; Sources carry a Label per
class (LABELS), so its boxes stand in for a detector, as Webots' recogniser did. A Gazebo
sensor renders only while someone subscribes to it, so the bridge subscribes exactly to the
cameras that are on: CAMV while there is a Verifier, the view cameras while a game screen
asks for them.

Frames: City x = east, y = north, metres  ==  Gazebo ENU world (z up); cameras look along +x.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping

from earshot import config
from earshot.backend.twin2d import Kinematics, Twin2D
from earshot.backend.webots import (
    CHASE,
    CHASE_PERCHED,
    LIVE_SLOTS,
    PARKED,
    PILOT,
    SOURCE_Z,
    TRAIN_WAGONS,
    VIEW_KEEPALIVE_S,
    WAGON_PITCH,
    def_names,
    live_slot,
)
from earshot.city import City
from earshot.scenario import Scenario
from earshot.state import Command, DroneState

__all__ = ["BASIC_MS", "LABELS", "LIVE_SLOTS", "N_VIEWS", "TRAIN_WAGONS", "GazeboBackend",
           "live_slot"]

BASIC_MS = 20  # Gazebo's max_step_size: one pose update per 20 ms, as Webots' basicTimeStep
LABELS = {"DRONE": 1, "VEHICLE": 2, "TRAIN": 3, "NOTHING": 4}  # Label system value per class
CLASS_OF_LABEL = {v: k for k, v in LABELS.items()}
N_VIEWS = 4  # view cameras CAM_0 (main, larger) .. CAM_3
VIEW_SIZE = ((854, 480), (384, 216), (384, 216), (384, 216))
VIEW_FOV = 1.15  # rad, horizontal
VERIFY_SIZE = (400, 240)
VERIFY_FOV = 1.0
RECOGNITION_RANGE = 100.0  # m; Webots' Recognition maxRange: the box camera's far clip
# Hz. Gazebo's label render costs with the scene's triangles (13 ms in the synthetic city,
# 92 ms at Munich), so the boxes refresh every other tick; the sensors read the newest
VERIFY_RATE = 5
CAMV_PITCH = 14.0  # degrees down from the Verifier's heading


def _yaw(heading_deg: float) -> float:
    """Compass heading (0 = north, clockwise) -> rotation about +z from +x (counter-clockwise)."""
    return math.radians(90.0 - heading_deg)


def pose(x: float, y: float, z: float, heading: float = 0.0, pitch_down: float = 0.0) -> list:
    """[x, y, z, roll, pitch, yaw] for the bridge. A positive pitch about +y tilts +x down."""
    return [x, y, z, 0.0, math.radians(pitch_down), _yaw(heading)]


class BridgeLink:
    """REQ socket to gz_bridge.py. `call` waits for the reply; a silent bridge ends the run."""

    def __init__(self, endpoint: str = config.GAZEBO_BRIDGE, timeout_s: float = 10.0):
        import zmq

        from earshot.runners.wire import pack, unpack

        self._pack, self._unpack, self._zmq = pack, unpack, zmq
        self.ctx = zmq.Context.instance()
        self.ctx.setsockopt(zmq.LINGER, 0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(endpoint)
        self.timeout_ms = int(timeout_s * 1000)
        self.waiting = False  # a request without its reply: REQ can send nothing more

    def call(self, msg: dict, timeout_s: float | None = None) -> dict:
        self.sock.send(self._pack(msg))
        self.waiting = True
        wait = self.timeout_ms if timeout_s is None else int(timeout_s * 1000)
        if not self.sock.poll(wait):
            raise SystemExit(f"Gazebo bridge did not answer {msg.get('op')!r} within {wait} ms")
        out = self._unpack(self.sock.recv())
        self.waiting = False
        return out

    def close(self) -> None:
        self.sock.close()


class GazeboBackend:
    def __init__(self, city: City, scenario: Scenario, link=None, startup_s: float = 120.0):
        self.link = link if link is not None else BridgeLink()
        hello = self.link.call({"op": "hello"}, timeout_s=startup_s)  # the world may be big
        missing = [f"DRONE_{d.id}" for d in scenario.drones
                   if f"DRONE_{d.id}" not in set(hello.get("models", []))]
        if missing:
            raise RuntimeError(f"Gazebo world lacks {missing}; regenerate it")
        self.basic_ms = int(hello.get("basic_ms", BASIC_MS))
        self.scenario = scenario
        self.twin = Twin2D(city, scenario.drones, Kinematics.for_scenario(scenario))
        self._models = set(hello.get("models", []))
        self._verifier: int | None = None
        self._objects: list[str] = []  # Source classes the Verifier camera boxes, last step
        ids = [d.id for d in scenario.drones]
        self.views: list[list] = [[ids[k % len(ids)], "chase" if k == 0 else "pilot"]
                                  for k in range(N_VIEWS)]
        self._views_asked: float | None = None
        self._view_yaw: dict[int, float] = {}
        self._view_pitch: dict[int, float] = {}
        self._names: dict[str, str] = {}  # Source id -> model name, see def_names
        self._placed: dict[str, list] = {}  # model -> last pose sent (send only changes)

    # -- views (the game screen's drone cameras) -------------------------------------------

    def set_views(self, views: list) -> None:
        """Same contract as WebotsBackend.set_views: [[drone_id, "pilot" | "chase"], ...];
        every call is also a keepalive."""
        self._views_asked = time.monotonic()
        if not isinstance(views, list | tuple):
            return
        drones = {d.id for d in self.scenario.drones}
        for k, v in enumerate(views[:N_VIEWS]):
            if not (isinstance(v, list | tuple) and len(v) == 2):
                continue
            did, mode = v
            if isinstance(did, int) and did in drones and mode in ("pilot", "chase"):
                self.views[k] = [did, mode]

    def _views_on(self) -> bool:
        return (self._views_asked is not None
                and time.monotonic() - self._views_asked < VIEW_KEEPALIVE_S)

    def view_frames(self, which: list[int], quality: int = 72) -> dict[int, dict]:
        """JPEG frames of the given view cameras: {k: {drone, mode, w, h, fov, pitch, jpeg}}."""
        if not self._views_on():
            return {}
        rep = self.link.call({"op": "frames", "which": list(which), "quality": quality})
        out = {}
        for k, f in rep.get("frames", {}).items():
            k = int(k)
            out[k] = {"drone": self.views[k][0], "mode": self.views[k][1], "w": f["w"],
                      "h": f["h"], "fov": VIEW_FOV, "pitch": self._view_pitch.get(k, 0.0),
                      "jpeg": f["jpeg"]}
        return out

    def _view_poses(self, states: dict[int, DroneState], out: dict) -> None:
        for k in range(N_VIEWS):
            did, mode = self.views[k]
            s = states.get(did)
            if s is None:
                continue
            prev = self._view_yaw.get(k, s.heading)  # swing, don't snap
            diff = (s.heading - prev + 180.0) % 360.0 - 180.0
            yaw = (prev + 0.06 * diff) % 360.0
            self._view_yaw[k] = yaw
            fx, fy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
            if mode == "pilot":
                p = [s.x + PILOT["ahead"] * fx, s.y + PILOT["ahead"] * fy, s.z + PILOT["up"]]
                pitch = PILOT["pitch"]
            else:
                c = CHASE_PERCHED if s.perched_on else CHASE
                p = [s.x - c["back"] * fx, s.y - c["back"] * fy, s.z + c["up"]]
                pitch = c["pitch"]
            self._view_pitch[k] = pitch
            out[f"CAM_{k}"] = pose(*p, heading=yaw, pitch_down=pitch)

    # -- Backend interface ---------------------------------------------------------------

    def step(self, dt: float, commands: Mapping[int, Command]) -> dict[int, DroneState]:
        n = max(1, round(dt * 1000 / self.basic_ms))
        frames, states = [], {}
        for i in range(n):
            states = self.twin.step(dt / n, commands if i == 0 else {})
            frames.append(self._poses(states))
        views_on = self._views_on()
        sensors = (["verify", "preview"] if self._verifier is not None else []) + (
            [f"view{k}" for k in range(N_VIEWS)] if views_on else [])
        rep = self.link.call({"op": "step", "frames": frames, "sensors": sensors})
        if rep.get("quit"):
            raise SystemExit("Gazebo quit")
        self._objects = [CLASS_OF_LABEL[lb] for lb in rep.get("labels", []) if lb in CLASS_OF_LABEL]
        return states

    def camera_frame(self, drone_id: int) -> object | None:
        """{"w", "h", "objects"}: what the Verifier camera boxes (one class name per box)."""
        if drone_id != self._verifier:
            return None
        return {"w": VERIFY_SIZE[0], "h": VERIFY_SIZE[1], "objects": list(self._objects)}

    def camera_image(self) -> bytes | None:
        """The Verifier camera's current image as BGRA (what world.camera_preview reads)."""
        if self._verifier is None:
            return None
        rep = self.link.call({"op": "preview"})
        rgb = rep.get("rgb")
        if not rgb or (rep["w"], rep["h"]) != VERIFY_SIZE:
            return None
        import numpy as np

        a = np.frombuffer(rgb, np.uint8).reshape(rep["h"], rep["w"], 3)
        bgra = np.empty((rep["h"], rep["w"], 4), np.uint8)
        bgra[..., :3] = a[..., ::-1]
        bgra[..., 3] = 255
        return bgra.tobytes()

    def close(self) -> None:
        """Ask the bridge to exit, unless a stop signal cut a request off mid-reply (the
        launcher then stops the bridge itself)."""
        if not getattr(self.link, "waiting", False):
            try:
                self.link.call({"op": "close"}, timeout_s=2.0)
            except SystemExit:
                pass
        self.link.close()

    # -- internals -----------------------------------------------------------------------

    def _poses(self, states: dict[int, DroneState]) -> dict[str, list]:
        """Poses of everything that moved since the last basic step: {model: pose}."""
        out: dict[str, list] = {}
        for did, s in states.items():
            out[f"DRONE_{did}"] = pose(s.x, s.y, s.z, s.heading)
        t = self.twin.t
        ground = self.twin.city.ground
        if len(self._names) != len(self.scenario.sources):  # a Source was added
            self._names = def_names(self.scenario.sources)
        placed: dict[str, list] = {}  # one pose per model; an active source wins
        for src in self.scenario.sources:
            base = self._names[src.id]
            cls = src.cls.name
            names = [f"{base}_w{k}" for k in range(TRAIN_WAGONS)] if cls == "TRAIN" else [base]
            for k, name in enumerate(names):
                p = src.pose(t, back=k * WAGON_PITCH)
                if p is None:
                    placed.setdefault(name, pose(*PARKED))
                    continue
                x, y, heading = p
                placed[name] = pose(x, y, float(ground(x, y)) + SOURCE_Z[cls], heading)
        out.update(placed)

        self._verifier = next((d for d, s in states.items() if s.camera_on), None)
        if self._verifier is None:
            out["CAMV"] = pose(*PARKED)
        else:
            s = states[self._verifier]
            out["CAMV"] = pose(s.x, s.y, s.z - 0.3, s.heading, CAMV_PITCH)
        if self._views_on():
            self._view_poses(states, out)
        changed = {}
        for name, p in out.items():
            if name in self._models and self._placed.get(name) != p:
                changed[name] = p
                self._placed[name] = p
        return changed

