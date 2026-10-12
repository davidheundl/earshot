"""Webots adapter: the 2D twin moves the drones, Webots shows them and renders one camera.

ADR 0002. Drones are plain Pose nodes (DEF DRONE_<id>) with no physics and no controller; a
single Supervisor robot (DEF RIG, controller <extern>) moves them every basic time step. A
Supervisor cannot read devices on other robots, so the RIG itself carries the one Camera and
flies along with whichever drone has camera_on (the Verifier). Sources are puppeted the same
way (DEF SRC_<id>), parked below ground while inactive.

Pacing: this adapter sleeps each basic step to its wall-clock deadline. With Webots in fast
mode (`earshot live --no-webots-view`) our own per-tick work (sensors, publishing, JPEG) then
fits into the slack; in realtime mode Webots spends the full step itself, our work comes on
top, and the pacing sleeps no longer. The 3D window keeps realtime mode, so it animates evenly.

Must run inside a Webots extern-controller environment (see `earshot live --backend webots`).
Frames: City x = east, y = north, metres  ==  Webots ENU world (z up).
"""

from __future__ import annotations

import math
import re
import time
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
CATCH_UP_S = 0.25  # further behind the wall clock than this: start over instead of sprinting


def live_slot(cls_name: str, k: int) -> str:
    return f"live_{cls_name.lower()}_slot{k}"


def def_names(sources) -> dict[str, str]:
    """Source id -> DEF name. Live sources (id live_<cls>_<n>) borrow their class's spare
    node slots in turn: the k-th live source of a class takes slot k % LIVE_SLOTS (`n` counts
    across all classes, so it cannot pick the slot). Sources are only ever appended, so an
    assignment never changes once made."""
    names, per_class = {}, {}
    for src in sources:
        tail = src.id.rsplit("_", 1)[-1]
        if src.id.startswith("live_") and tail.isdigit():
            k = per_class.get(src.cls.name, 0)
            per_class[src.cls.name] = k + 1
            names[src.id] = f"SRC_{live_slot(src.cls.name, k % LIVE_SLOTS)}"
        else:
            names[src.id] = f"SRC_{src.id}"
    return names


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
VIEW_KEEPALIVE_S = 5.0  # view cameras render only while a screen asked for them this recently
VERIFY_PERIOD_MS = 100  # the sensors read the Verifier camera once per 0.1 s tick
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
            self._view_cams.append((cam, node))  # enabled while a screen shows them
        self._views_asked: float | None = None  # wall clock of the last set_views
        self._views_on = False
        self._view_yaw: dict[int, float] = {}
        self._due: float | None = None  # wall-clock deadline of the next basic step
        self._names: dict[str, str] = {}  # Source id -> DEF name, see def_names
        self._seen: dict[int, str | None] = {}  # recognised object id -> its Source class
        self._def_class = {f"SRC_{s.id}": s.cls.name for s in scenario.sources}
        for cls in ("DRONE", "VEHICLE", "TRAIN", "NOTHING"):
            for k in range(LIVE_SLOTS):
                self._def_class[f"SRC_{live_slot(cls, k)}"] = cls

    # -- views (the game screen's drone cameras) -------------------------------------------

    def set_views(self, views: list) -> None:
        """[[drone_id, "pilot" | "chase"], ...] for CAM_0 (main) .. CAM_3. Malformed entries
        are ignored. Every call is also a screen's keepalive (None: keepalive only); without
        one for VIEW_KEEPALIVE_S (no game screen open, or it shows the evidence view) the
        view cameras stop rendering."""
        self._views_asked = time.monotonic()
        if not isinstance(views, list | tuple):
            return
        for k, v in enumerate(views[: len(self._view_cams)]):
            if not (isinstance(v, list | tuple) and len(v) == 2):
                continue
            did, mode = v
            if isinstance(did, int) and did in self._drones and mode in ("pilot", "chase"):
                self.views[k] = [did, mode]

    def view_frames(self, which: list[int], quality: int = 72) -> dict[int, dict]:
        """JPEG frames of the given view cameras: {k: {drone, mode, w, h, jpeg}}."""
        import io

        from PIL import Image

        out = {}
        for k in which:
            if not self._views_on or k >= len(self._view_cams):
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
        on = (self._views_asked is not None
              and time.monotonic() - self._views_asked < VIEW_KEEPALIVE_S)
        if on != self._views_on:
            self._views_on = on
            for k, (cam, _) in enumerate(self._view_cams):
                if on:
                    cam.enable(VIEW_PERIOD_MS[k])
                else:
                    cam.disable()
        if not on:
            return
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
            self._pace()
        return states

    def camera_frame(self, drone_id: int) -> object | None:
        """{"w", "h", "objects"}: what the Verifier recognises. The image is copied only on
        request (`camera_image`, the dashboard preview); the sensors need just the objects."""
        if drone_id != self._verifier:
            return None
        c = self.camera
        frame = {"w": c.getWidth(), "h": c.getHeight()}
        if c.hasRecognition():
            # the Webots recogniser is the stand-in detector: it reports each Solid's model,
            # which the world generator sets to the Source's class
            frame["objects"] = self._recognised()
        return frame

    def camera_image(self) -> bytes | None:
        """The Verifier camera's current BGRA image."""
        return self.camera.getImage() if self._verifier is not None else None

    def close(self) -> None:
        pass

    # -- internals -----------------------------------------------------------------------

    def _pace(self) -> None:
        """Sleep until this basic step's wall-clock deadline (real time). A Webots already in
        realtime mode has used the time up, so this then returns at once."""
        now = time.monotonic()
        if self._due is None or now - self._due > CATCH_UP_S:
            self._due = now
        self._due += self.basic_ms / 1000
        time.sleep(max(0.0, self._due - time.monotonic()))

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
            oid = o.getId()
            if oid not in self._seen:  # node lookup and parent walk once per object
                node = self.sup.getFromId(oid)
                while node is not None and not (node.getDef() or "").startswith("SRC_"):
                    node = node.getParentNode()
                self._seen[oid] = (None if node is None
                                   else self._def_class.get(re.sub(r"_w\d+$", "", node.getDef())))
            cls = self._seen[oid]
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
        if len(self._names) != len(self.scenario.sources):  # a Source was added
            self._names = def_names(self.scenario.sources)
        for src in self.scenario.sources:
            base = self._names[src.id]
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
                self.camera.enable(VERIFY_PERIOD_MS)
                if self.camera.hasRecognition():
                    self.camera.recognitionEnable(VERIFY_PERIOD_MS)
            self._verifier = verifier
        if self._camv is not None:
            if verifier is None:
                self._camv[0].setSFVec3f(PARKED)
            else:
                s = states[verifier]
                self._camv[0].setSFVec3f([s.x, s.y, s.z - 0.3])
                self._camv[1].setSFRotation(look_rotation(s.heading, 14.0))
        self._place_views(states)
