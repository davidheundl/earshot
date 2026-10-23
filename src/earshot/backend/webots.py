"""Webots adapter: the 2D twin moves the drones, Webots shows them and renders one camera.

ADR 0002. Drones are plain Pose nodes (DEF DRONE_<id>) with no physics and no controller; a
single Supervisor robot (DEF RIG, controller <extern>) moves them every basic time step. A
Supervisor cannot read devices on other robots, so the RIG itself carries the one Camera and
flies along with whichever drone has camera_on (the Verifier). Sources are puppeted the same
way (DEF SRC_<id>), parked below ground while inactive.

Pacing: this adapter sleeps each basic step to its wall-clock deadline. The last basic step
of a tick is left open (`stepBegin`) when `step` returns and closed (`stepEnd`) at the start of
the next call, so Webots computes it while the world process does sensors, publishing and
views; in realtime mode, where Webots spends the full step itself, that work then no longer
comes on top. Nothing may ask Webots anything while a step is open (no getFromDef, getField,
getImage, ...), so the camera images and recognised objects the world reads are copied just
before the step opens (`_grab`) and served from that copy.

If Webots quits mid-run (window closed, crash), the run goes on: this adapter says so once
and keeps stepping its own twin at wall-clock pace, without camera images.

Must run inside a Webots extern-controller environment (see `earshot live --backend webots`).
Frames: City x = east, y = north, metres  ==  Webots ENU world (z up).
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Mapping

from earshot.backend.twin2d import Kinematics, Twin2D
from earshot.city import City
from earshot.scenario import Scenario, _target_class
from earshot.state import Command, DroneState

PARKED = [0.0, 0.0, -100.0]
# visual model origin above ground: drone at flight height, truck wheel axis, wagon on the rails
SOURCE_Z = {"DRONE": 30.0, "VEHICLE": 0.5, "TRAIN": 0.45, "NOTHING": 0.7}
TRAIN_WAGONS = 9  # locomotive + 8 container wagons, nodes SRC_<id>_w<k>
WAGON_PITCH = 21.0  # m between wagon centres
# spare SRC nodes per class for Sources added by hotkey: Webots refuses more (no stealing an
# earlier one's node); Gazebo (which imports LIVE_SLOTS) keeps 2, as each visual costs it render
WEBOTS_LIVE_SLOTS = 4
LIVE_SLOTS = 2
CATCH_UP_S = 0.25  # further behind the wall clock than this: start over instead of sprinting


def live_slot(cls_name: str, k: int) -> str:
    return f"live_{cls_name.lower()}_slot{k}"


def def_names(sources, slots: int = LIVE_SLOTS, wrap: bool = True) -> dict[str, str]:
    """Source id -> DEF name. Live sources (id live_<cls>_<n>) take their class's `slots`
    spare nodes in turn: the k-th live source of a class takes slot k (`n` counts across all
    classes, so it cannot pick the slot). Beyond them, with `wrap` it takes slot k % slots
    (an earlier one's node), else it gets no name (heard, not drawn). Sources are only ever
    appended, so an assignment never changes once made."""
    names, per_class = {}, {}
    for src in sources:
        if is_live(src.id):
            k = per_class.get(src.cls.name, 0)
            per_class[src.cls.name] = k + 1
            if wrap or k < slots:
                names[src.id] = f"SRC_{live_slot(src.cls.name, k % slots)}"
        else:
            names[src.id] = f"SRC_{src.id}"
    return names


def is_live(source_id: str) -> bool:
    """Added by a live Event without an explicit id (scenario.apply names it live_<cls>_<n>)."""
    return source_id.startswith("live_") and source_id.rsplit("_", 1)[-1].isdigit()


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


N_VIEWS = 1  # view cameras on the RIG: CAM_0 only, the main view (each tile costs render time)
VIEW_PERIOD_MS = (100, 200, 200, 200)  # main view 10 Hz, tiles 5 Hz (with N_VIEWS 4)
VIEW_KEEPALIVE_S = 5.0  # view cameras render only while a screen asked for them this recently
VERIFY_PERIOD_MS = 100  # the sensors read the Verifier camera once per 0.1 s tick
TARGET_DROP_M = 10.0  # a Verifier cruises 10 m above an intruder (40 m vs. 30 m above ground)
PILOT = {"ahead": 0.6, "up": -0.15, "pitch": 10.0}  # just ahead of the props
CHASE = {"back": 3.2, "up": 1.1, "pitch": 12.0}  # close behind: the drone fills the lower third
CHASE_PERCHED = {"back": 9.0, "up": 4.5, "pitch": 24.0}  # see the roof and what it listens to


def _image(cam) -> bytes | None:
    """A copy of the camera's image; None before its first one (the API reads a NULL buffer)."""
    try:
        return cam.getImage()
    except ValueError:
        return None


class _Encoder:
    """JPEG-encodes view images on a thread of its own, off the world's tick. One job waits at
    most: a newer image of the same camera replaces one not yet encoded. Pillow releases the
    GIL while it converts and encodes."""

    def __init__(self):
        self._cv = threading.Condition()
        self._jobs: dict[int, tuple] = {}  # camera k -> (grabbed view, quality), waiting
        self._done: dict[int, dict] = {}  # camera k -> frame, not yet taken
        threading.Thread(target=self._run, name="view-jpeg", daemon=True).start()

    def submit(self, views: dict[int, tuple], quality: int) -> None:
        if views:
            with self._cv:
                self._jobs.update({k: (v, quality) for k, v in views.items()})
                self._cv.notify()

    def take(self) -> dict[int, dict]:
        with self._cv:
            done, self._done = self._done, {}
        return done

    def _run(self) -> None:
        import io

        from PIL import Image

        while True:
            with self._cv:
                while not self._jobs:
                    self._cv.wait()
                jobs, self._jobs = self._jobs, {}
            for k, ((raw, w, h, meta), quality) in jobs.items():
                img = Image.frombuffer("RGBA", (w, h), raw, "raw", "BGRA", 0, 1).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=quality)
                frame = {**meta, "w": w, "h": h, "jpeg": buf.getvalue()}
                with self._cv:
                    self._done[k] = frame


class WebotsBackend:
    timing = None  # earshot live --profile: a runners.profile.TickProfile
    _encoder: _Encoder | None = None  # started with the first view frames

    def __init__(self, city: City, scenario: Scenario):
        from controller import Supervisor  # only importable inside the Webots env

        self.sup = Supervisor()
        self.basic_ms = int(self.sup.getBasicTimeStep())
        self.scenario = scenario
        self.twin = Twin2D(city, scenario.drones, Kinematics.for_scenario(scenario))
        self._drones = {}
        for d in scenario.drones:
            node = self.sup.getFromDef(f"DRONE_{d.id}")
            if node is None:
                raise RuntimeError(f"world has no DEF DRONE_{d.id}; regenerate it")
            self._drones[d.id] = (node.getField("translation"), node.getField("rotation"))
        self._fields: dict[str, object | None] = {}  # DEF name -> translation field (cached)
        self._written: dict[int, list[float]] = {}  # id(field) -> the value last written
        # all cameras ride on child Solids of the RIG, which stays at the origin, so a child's
        # local pose is its world pose (a Supervisor can only read its own robot's devices)
        self._camv = self._node("CAMV")
        self.camera = self.sup.getDevice("camera")
        self._verifier: int | None = None
        self._verifier_heading = 0.0
        self._rig_at = (0.0, 0.0, 0.0)  # where the Verifier camera is (x, y, z)
        self.detector_frames = False  # set by the World when a trained detector reads images
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
        self._view_pitch: dict[int, float] = {}  # degrees down, so a HUD can draw the horizon
        self._view_shown: dict[int, tuple] = {}  # (yaw, pitch) the last rendered image had
        self._due: float | None = None  # wall-clock deadline of the next basic step
        self._quit = False  # Webots went away: the twin alone carries on
        self._open = False  # a basic step is running in Webots (stepBegin without stepEnd)
        self._grabbed: dict = {}  # camera data copied before the step opened, see _grab
        self._grabs = 0
        self._last_image: bytes | None = None  # the Verifier's, see _verify_image
        self._names: dict[str, str] = {}  # Source id -> DEF name, see def_names
        self._seen: dict[int, str | None] = {}  # recognised object id -> its Source class
        self._def_class = {f"SRC_{s.id}": s.cls.name for s in scenario.sources}
        for cls in ("DRONE", "VEHICLE", "TRAIN", "NOTHING"):
            for k in range(WEBOTS_LIVE_SLOTS):
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
        """JPEG frames of the given view cameras: {k: {drone, mode, w, h, fov, yaw, pitch,
        boxes, jpeg}}; `boxes` are the Sources the camera recognises in that image, as
        {cls, x, y, w, h} (pixels, x and y the top-left corner).
        The images grabbed this tick go to the encoder thread; returned are the frames it
        finished since the last call (a tick late, none while it is busy)."""
        if self._quit:
            return {}
        views = self._grabbed.get("views", {})
        if self._encoder is None:
            self._encoder = _Encoder()
        self._encoder.submit({k: views[k] for k in which if k in views}, quality)
        return self._encoder.take()

    def _place_views(self, states: dict[int, DroneState]) -> None:
        on = (self._views_asked is not None
              and time.monotonic() - self._views_asked < VIEW_KEEPALIVE_S)
        if on != self._views_on:
            self._views_on = on
            for k, (cam, _) in enumerate(self._view_cams):
                if on:
                    cam.enable(VIEW_PERIOD_MS[k])
                    if cam.hasRecognition():  # the boxes; an older world has none
                        cam.recognitionEnable(VIEW_PERIOD_MS[k])
                else:
                    cam.disable()
                    if cam.hasRecognition():
                        cam.recognitionDisable()
        if not on:
            return
        # the image read before the next step was rendered with the pose set before this one
        self._view_shown = {k: (self._view_yaw[k], self._view_pitch[k]) for k in self._view_yaw}
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
                pitch = PILOT["pitch"]
            else:
                c = CHASE_PERCHED if s.perched_on else CHASE
                pos = [s.x - c["back"] * fx, s.y - c["back"] * fy, s.z + c["up"]]
                pitch = c["pitch"]
            rot = look_rotation(yaw, pitch)
            self._view_pitch[k] = pitch
            self._set_vec(node[0], pos)
            self._set_rot(node[1], rot)

    # -- Backend interface ---------------------------------------------------------------

    def step(self, dt: float, commands: Mapping[int, Command]) -> dict[int, DroneState]:
        n = max(1, round(dt * 1000 / self.basic_ms))
        states = {}
        for i in range(n):
            self._finish()
            t0 = time.perf_counter()
            states = self.twin.step(dt / n, commands if i == 0 else {})
            t1 = time.perf_counter()
            if self._quit:
                continue
            self._push(states)
            t2 = time.perf_counter()
            if i == n - 1:
                self._grab()  # the world reads these while the last step runs
            t3 = time.perf_counter()
            self._open = self.sup.stepBegin(self.basic_ms) != -1
            if self.timing is not None:
                self.timing.add("wb.twin", t1 - t0)
                self.timing.add("wb.push", t2 - t1)
                self.timing.add("wb.grab", t3 - t2)
                self.timing.add("wb.begin", time.perf_counter() - t3)
            if not self._open:
                self._gone()
        return states

    def _finish(self) -> None:
        """Close the open basic step (wait for Webots to finish it), then sleep to its
        wall-clock deadline. Each basic step, open or not (Webots gone), gets one pace."""
        if self._open:
            t0 = time.perf_counter()
            self._open = False
            ended = self.sup.stepEnd()
            if self.timing is not None:
                self.timing.add("wb.end", time.perf_counter() - t0)
            if ended == -1:
                self._gone()
        self._pace()

    def _gone(self) -> None:
        self._quit = True  # never touch the Supervisor again
        self._verifier = None
        self._grabbed = {}
        print("[webots] Webots quit; the run goes on in the 2D twin, without camera images",
              flush=True)

    def _grab(self) -> None:
        """Copy what the world reads from Webots during the open step: the Verifier camera's
        image and recognised classes, the view cameras' raw images."""
        g: dict = {}
        c = self.camera
        if self._verifier is not None:
            g["verify"] = {"w": c.getWidth(), "h": c.getHeight(), "fov": c.getFov(),
                           "bgra": self._verify_image(),
                           "objects": self._recognised() if c.hasRecognition() else None,
                           "yaw": self._verifier_heading,
                           "pitch": math.radians(self._verify_pitch()),
                           "height": self._rig_at[2] - float(self.twin.city.ground(*self._rig_at[:2]))}
        if self._views_on:
            g["views"] = {}
            for k, (cam, _) in enumerate(self._view_cams):
                raw = _image(cam)
                if raw:
                    did, mode = self.views[k]
                    yaw, pitch = self._view_shown.get(k, (None, 0.0))
                    g["views"][k] = (raw, cam.getWidth(), cam.getHeight(), {
                        "drone": did, "mode": mode, "fov": cam.getFov(), "yaw": yaw,
                        "pitch": pitch,  # for the HUD
                        "boxes": self._boxes(cam) if cam.hasRecognition() else []})
        self._grabbed = g

    def _verify_image(self) -> bytes | None:
        """The Verifier image: every tick for a trained detector, else once a second for the
        dashboard preview (a 1280x768 copy costs ~3 ms)."""
        self._grabs += 1
        if self.detector_frames or self._grabs % 10 == 1:
            self._last_image = _image(self.camera)
        return self._last_image

    def refuses(self, op: dict) -> str | None:
        """Why a live Event cannot be shown, else None: an added Source needs a free spare
        node of its class (WEBOTS_LIVE_SLOTS), or it would be invisible in the 3D view."""
        if op.get("op") != "add_source" or op.get("id"):
            return None
        try:
            cls = _target_class(op.get("cls")).name
        except (ValueError, AttributeError):
            return None  # scenario.apply rejects it with a better message
        taken = sum(1 for s in self.scenario.sources if s.cls.name == cls and is_live(s.id))
        if taken >= WEBOTS_LIVE_SLOTS:
            return (f"all {WEBOTS_LIVE_SLOTS} spare Webots nodes for {cls.lower()} Sources are "
                    "taken (WEBOTS_LIVE_SLOTS in backend/webots.py)")
        return None

    def camera_frame(self, drone_id: int) -> object | None:
        """{"w", "h", "objects"}: what the Verifier recognises. With `detector_frames` (a trained
        detector reads the images) also {"bgra", "fov" (rad), "yaw" (deg), "pitch" (rad),
        "height" (m the camera is above the ground below it)}; else the image is handed over
        only on request (`camera_image`, the dashboard preview). Served from the copy `_grab`
        took before the current step opened."""
        g = self._grabbed.get("verify")
        if self._quit or drone_id != self._verifier or g is None:
            return None
        frame = {"w": g["w"], "h": g["h"]}
        if self.detector_frames:
            frame.update(bgra=g["bgra"], fov=g["fov"], yaw=g["yaw"], pitch=g["pitch"],
                         height=g["height"])
        if g["objects"] is not None:
            # the Webots recogniser is the stand-in detector: it reports each Solid's model,
            # which the world generator sets to the Source's class
            frame["objects"] = g["objects"]
        return frame

    def camera_image(self) -> bytes | None:
        """The Verifier camera's BGRA image, as copied before the current step opened."""
        g = self._grabbed.get("verify")
        return g["bgra"] if g is not None and not self._quit else None

    def close(self) -> None:
        if self._open and not self._quit:
            self._open = False
            self.sup.stepEnd()

    # -- internals -----------------------------------------------------------------------

    def _pace(self) -> None:
        """Sleep until this basic step's wall-clock deadline (real time). A Webots already in
        realtime mode has used the time up, so this then returns at once."""
        now = time.monotonic()
        if self._due is None or now - self._due > CATCH_UP_S:
            self._due = now
        self._due += self.basic_ms / 1000
        slack = max(0.0, self._due - time.monotonic())
        if self.timing is not None:
            self.timing.add("wb.pace", slack)
        time.sleep(slack)

    def _verify_pitch(self) -> float:
        """Degrees the Verifier camera looks down, at the point it keeps its standoff from. A
        trained detector (`detector_frames`) needs the target in its pixels: the camera aims at
        the hunted class's height there, a drone 10 m below (14 deg at 40 m), a vehicle on the
        ground (18 deg at 120 m; at 30 m, 53 deg). The stand-in's recogniser sees out to
        camera.range, so it keeps the drone's tilt, which reaches the horizon."""
        drop = TARGET_DROP_M
        if self.detector_frames:
            drop = self.twin.kin.cruise_alt - SOURCE_Z[self.scenario.threat.name]
        return math.degrees(math.atan2(drop, self.twin.kin.standoff))

    def _node(self, def_name: str):
        """(translation, rotation) fields of a DEF node, cached; None if the world lacks it."""
        if def_name not in self._fields:
            node = self.sup.getFromDef(def_name)
            self._fields[def_name] = (node.getField("translation"), node.getField("rotation")) \
                if node else None
        return self._fields[def_name]

    def _set_vec(self, field, value: list[float]) -> None:
        """Write a translation only when it changed: a parked Source or a perched drone once."""
        if self._written.get(id(field)) != value:
            self._written[id(field)] = value
            field.setSFVec3f(value)

    def _set_rot(self, field, value: list[float]) -> None:
        if self._written.get(id(field)) != value:
            self._written[id(field)] = value
            field.setSFRotation(value)

    def _source_class(self, oid: int) -> str | None:
        """The Source class of a recognised object: its node -> nearest SRC_ ancestor."""
        if oid not in self._seen:  # node lookup and parent walk once per object
            node = self.sup.getFromId(oid)
            while node is not None and not (node.getDef() or "").startswith("SRC_"):
                node = node.getParentNode()
            self._seen[oid] = (None if node is None
                               else self._def_class.get(re.sub(r"_w\d+$", "", node.getDef())))
        return self._seen[oid]

    def _recognised(self) -> list[str]:
        """Class names of the Sources the Verifier camera sees."""
        return [cls for o in self.camera.getRecognitionObjects()
                if (cls := self._source_class(o.getId()))]

    def _boxes(self, cam) -> list[dict]:
        """The Sources a view camera recognises, as boxes on its image: what Webots frames in
        red in its own camera window."""
        out = []
        for o in cam.getRecognitionObjects():
            cls = self._source_class(o.getId())
            if cls:
                (x, y), (w, h) = o.getPositionOnImage(), o.getSizeOnImage()  # x, y: the centre
                out.append({"cls": cls, "x": x - w // 2, "y": y - h // 2, "w": w, "h": h})
        return out

    def _push(self, states: dict[int, DroneState]) -> None:
        for did, s in states.items():
            tf, rf = self._drones[did]
            self._set_vec(tf, [s.x, s.y, s.z])
            self._set_rot(rf, [0.0, 0.0, 1.0, _yaw(s.heading)])

        t = self.twin.t
        placed: dict[str, tuple] = {}  # one pose per node; an active source wins
        ground = self.twin.city.ground
        if len(self._names) != len(self.scenario.sources):  # a Source was added
            self._names = def_names(self.scenario.sources, WEBOTS_LIVE_SLOTS, wrap=False)
        for src in self.scenario.sources:
            base = self._names.get(src.id)
            if base is None:  # a live Source beyond the spare nodes: heard, not drawn
                continue
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
                self._set_vec(node[0], pos)
                self._set_rot(node[1], [0.0, 0.0, 1.0, _yaw(heading)])

        # one rig: it stays with its Verifier until that camera goes off, so a second
        # Verifier arriving cannot take it mid-look (that one stays inconclusive)
        verifier = (self._verifier if self._verifier in states and states[self._verifier].camera_on
                    else next((d for d, s in states.items() if s.camera_on), None))
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
            self._last_image, self._grabs = None, 0  # no frame of the previous look
        if self._camv is not None:
            if verifier is None:
                self._set_vec(self._camv[0], PARKED)
            else:
                s = states[verifier]
                self._verifier_heading = s.heading
                self._rig_at = (s.x, s.y, s.z - 0.3)
                self._set_vec(self._camv[0], list(self._rig_at))
                self._set_rot(self._camv[1], look_rotation(s.heading, self._verify_pitch()))
        self._place_views(states)
