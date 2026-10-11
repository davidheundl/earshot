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
# visual model origin above ground: Mavic at flight height, truck wheel axis, wagon on the rails
SOURCE_Z = {"DRONE": 30.0, "VEHICLE": 0.5, "TRAIN": 0.45, "NOTHING": 0.7}
TRAIN_WAGONS = 9  # locomotive + 8 container wagons, nodes SRC_<id>_w<k>
WAGON_PITCH = 21.0  # m between wagon centres
LIVE_SLOTS = 2  # spare SRC nodes per class for sources added by hotkey


def live_slot(cls_name: str, k: int) -> str:
    return f"live_{cls_name.lower()}_slot{k}"


def _def_name(src) -> str:
    """Live sources (id live_<cls>_<n>) borrow spare node slot n % LIVE_SLOTS."""
    tail = src.id.rsplit("_", 1)[-1]
    if src.id.startswith("live_") and tail.isdigit():
        return f"SRC_{live_slot(src.cls.name, int(tail) % LIVE_SLOTS)}"
    return f"SRC_{src.id}"


def _yaw(heading_deg: float) -> float:
    """Compass heading (0 = north, clockwise) -> rotation about +z from +x (counter-clockwise)."""
    return math.radians(90.0 - heading_deg)


def look_rotation(heading_deg: float, pitch_down_deg: float) -> list[float]:
    """Axis-angle that points a camera (looking along its +x) at compass `heading`, tilted
    down by `pitch`: yaw about z, then pitch about the camera's own y."""
    yaw, pitch = _yaw(heading_deg), math.radians(pitch_down_deg)
    w1, z1 = math.cos(yaw / 2), math.sin(yaw / 2)
    w2, y2 = math.cos(pitch / 2), math.sin(pitch / 2)
    w, x, y, z = w1 * w2, -z1 * y2, w1 * y2, z1 * w2
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    return [x / n, y / n, z / n, 2 * math.acos(max(-1.0, min(1.0, w)))]


N_VIEWS = 4  # view cameras on the RIG: CAM_0 (main, larger) .. CAM_3
VIEW_PERIOD_MS = (100, 200, 200, 200)  # main view 10 Hz, tiles 5 Hz
PILOT = {"ahead": 1.8, "up": -0.2, "pitch": 16.0}  # just outside the marker ring
CHASE = {"back": 14.0, "up": 6.0, "pitch": 22.0}
CHASE_PERCHED = {"back": 32.0, "up": 20.0, "pitch": 30.0}  # see the roof and what it listens to


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
        self._fields: dict[str, object | None] = {}  # DEF name -> translation field (cached)
        # all cameras ride on child Solids of the RIG, which stays at the origin, so a child's
        # local pose is its world pose (a Supervisor can only read its own robot's devices)
        self._camv = self._node("CAMV")
        self.camera = self.sup.getDevice("camera")
        self._verifier: int | None = None
        ids = [d.id for d in scenario.drones]
        self.views: list[list] = [[ids[k % len(ids)], "chase" if k == 0 else "pilot"]
                                  for k in range(N_VIEWS)]
        self._view_cams = []
        for k in range(N_VIEWS):
            cam = self.sup.getDevice(f"view{k}")
            node = self._node(f"CAM_{k}")
            if cam is None or node is None:
                break  # an older world without view cameras
            cam.enable(VIEW_PERIOD_MS[k])
            self._view_cams.append((cam, node))
        self._view_yaw: dict[int, float] = {}
        self._def_class = {f"SRC_{s.id}": s.cls.name for s in scenario.sources}
        for cls in ("DRONE", "VEHICLE", "TRAIN", "NOTHING"):
            for k in range(LIVE_SLOTS):
                self._def_class[f"SRC_{live_slot(cls, k)}"] = cls

    # -- views (the game screen's drone cameras) -------------------------------------------

    def set_views(self, views: list) -> None:
        """[[drone_id, "pilot" | "chase"], ...] for CAM_0 (main) .. CAM_3."""
        for k, (did, mode) in enumerate(views[: len(self._view_cams)]):
            if did in self._drones and mode in ("pilot", "chase"):
                self.views[k] = [did, mode]

    def view_frames(self, which: list[int], quality: int = 72) -> dict[int, dict]:
        """JPEG frames of the given view cameras: {k: {drone, mode, w, h, jpeg}}."""
        import io

        from PIL import Image

        out = {}
        for k in which:
            if k >= len(self._view_cams):
                continue
            cam, _ = self._view_cams[k]
            raw = cam.getImage()
            if not raw:
                continue
            w, h = cam.getWidth(), cam.getHeight()
            img = Image.frombuffer("RGBA", (w, h), raw, "raw", "BGRA", 0, 1).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=quality)
            out[k] = {"drone": self.views[k][0], "mode": self.views[k][1], "w": w, "h": h,
                      "jpeg": buf.getvalue()}
        return out

    def _place_views(self, states: dict[int, DroneState]) -> None:
        for k, (cam, node) in enumerate(self._view_cams):
            did, mode = self.views[k]
            s = states.get(did)
            if s is None:
                continue
            # smooth the heading so a chase camera swings instead of snapping
            prev = self._view_yaw.get(k, s.heading)
            diff = (s.heading - prev + 180.0) % 360.0 - 180.0
            yaw = (prev + 0.06 * diff) % 360.0
            self._view_yaw[k] = yaw
            fx, fy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
            if mode == "pilot":
                pos = [s.x + PILOT["ahead"] * fx, s.y + PILOT["ahead"] * fy, s.z + PILOT["up"]]
                rot = look_rotation(yaw, PILOT["pitch"])
            else:
                c = CHASE_PERCHED if s.perched_on else CHASE
                pos = [s.x - c["back"] * fx, s.y - c["back"] * fy, s.z + c["up"]]
                rot = look_rotation(yaw, c["pitch"])
            node[0].setSFVec3f(pos)
            node[1].setSFRotation(rot)

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
        if c.hasRecognition():
            # the Webots recogniser is the stand-in detector: it reports each Solid's model,
            # which the world generator sets to the Source's class
            frame["objects"] = self._recognised()
            if c.hasRecognitionSegmentation():
                frame["seg"] = c.getRecognitionSegmentationImage()
        return frame

    def close(self) -> None:
        pass

    # -- internals -----------------------------------------------------------------------

    def _node(self, def_name: str):
        """(translation, rotation) fields of a DEF node, cached; None if the world lacks it."""
        if def_name not in self._fields:
            node = self.sup.getFromDef(def_name)
            self._fields[def_name] = (node.getField("translation"), node.getField("rotation")) \
                if node else None
        return self._fields[def_name]

    def _recognised(self) -> list[str]:
        """Class names of the Sources in view: object -> its node -> nearest SRC_ ancestor."""
        out = []
        for o in self.camera.getRecognitionObjects():
            node = self.sup.getFromId(o.getId())
            while node is not None and not (node.getDef() or "").startswith("SRC_"):
                node = node.getParentNode()
            if node is None:
                continue
            cls = self._def_class.get(node.getDef().split("_w")[0])
            if cls:
                out.append(cls)
        return out

    def _push(self, states: dict[int, DroneState]) -> None:
        for did, s in states.items():
            tf, rf = self._drones[did]
            tf.setSFVec3f([s.x, s.y, s.z])
            rf.setSFRotation([0.0, 0.0, 1.0, _yaw(s.heading)])

        t = self.twin.t
        placed: dict[str, tuple] = {}  # one pose per node; an active source wins
        ground = self.twin.city.ground
        for src in self.scenario.sources:
            base = _def_name(src)
            cls = src.cls.name
            names = [f"{base}_w{k}" for k in range(TRAIN_WAGONS)] if cls == "TRAIN" else [base]
            for k, name in enumerate(names):
                pose = src.pose(t, back=k * WAGON_PITCH)
                if pose is None:
                    placed.setdefault(name, (PARKED, 0.0))
                    continue
                x, y, heading = pose
                placed[name] = ([x, y, float(ground(x, y)) + SOURCE_Z[cls]], heading)
        for name, (pos, heading) in placed.items():
            node = self._node(name)
            if node is not None:
                node[0].setSFVec3f(pos)
                node[1].setSFRotation([0.0, 0.0, 1.0, _yaw(heading)])

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
        if self._camv is not None:
            if verifier is None:
                self._camv[0].setSFVec3f(PARKED)
            else:
                s = states[verifier]
                self._camv[0].setSFVec3f([s.x, s.y, s.z - 0.3])
                self._camv[1].setSFRotation(look_rotation(s.heading, 14.0))
        self._place_views(states)
