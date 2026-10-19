"""Gazebo bridge: embeds the Gazebo Sim server and steps it for GazeboBackend (ADR 0008).

    gazebo/.venv/bin/python src/earshot/backend/gz_bridge.py gazebo/worlds/demo.sdf

Runs under Homebrew's Python (the only one with Gazebo's bindings), in the venv that
`earshot live --backend gazebo` makes (gazebo/requirements.txt), so it imports nothing from
earshot. Needs GZ_PARTITION and GZ_IP in its environment (`cli.gazebo_env()`); `gz sim -g`
with the same environment shows the world.

One REP socket, msgpack, one request at a time:
    {"op": "hello"}                       -> {"models": [...], "basic_ms": 20}
    {"op": "step", "frames": [{model: [x, y, z, roll, pitch, yaw]}, ...],
     "sensors": ["verify", "preview", "view0", ...]}
                                          -> {"labels": [...], "t": sim seconds}
        one Gazebo iteration per frame (its poses applied first), each paced to its
        wall-clock deadline; "sensors" are the cameras to keep rendering (subscribed)
    {"op": "frames", "which": [k, ...], "quality": q} -> {"frames": {k: {w, h, jpeg}}}
    {"op": "preview"}                     -> {"w", "h", "rgb"}  (the Verifier camera)
    {"op": "close"}                       -> {} and exit
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time

import msgpack
import zmq
from gz.math7 import Pose3d
from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
from gz.msgs10.image_pb2 import Image
from gz.sim8 import Model, TestFixture, World, world_entity
from gz.transport13 import Node

CATCH_UP_S = 0.25  # further behind the wall clock than this: start over instead of sprinting
TOPICS = {  # sensor name -> (topic, message type); names fixed by tools/build_gazebo_world.py
    "verify": ("/earshot/verify/boxes", AnnotatedAxisAligned2DBox_V),
    "preview": ("/earshot/verify/image", Image),
    **{f"view{k}": (f"/earshot/view{k}/image", Image) for k in range(4)},
}


class Bridge:
    def __init__(self, world: str, basic_ms: int, pace: bool):
        self.basic_ms, self.pace = basic_ms, pace
        self.pending: dict[str, list] = {}  # poses for the next iteration
        self.models: dict[str, Model] = {}
        self.names: list[str] = []
        self.latest: dict[str, object] = {}  # sensor -> newest message
        self.lock = threading.Lock()
        self.node = Node()
        self.on: set[str] = set()
        self.sim_t = 0.0
        self._due: float | None = None
        fx = TestFixture(world)
        fx.on_pre_update(self._pre)
        fx.finalize()
        self.server = fx.server()
        self.server.run(True, 1, False)  # loads the world; _pre lists its models
        self._warm_up()

    def _warm_up(self) -> None:
        """Render every camera once: Gazebo starts its renderer on the first subscription,
        which takes seconds, and a Verifier would see nothing meanwhile."""
        self.sensors(set(TOPICS))
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            self.server.run(True, 5, False)
            with self.lock:
                if set(self.latest) == set(TOPICS):
                    break
        self.sensors(set())

    def _pre(self, info, ecm) -> None:
        self.sim_t = info.sim_time.total_seconds()
        if not self.models:
            w = World(world_entity(ecm))
            for e in w.models(ecm):
                m = Model(e)
                self.models[m.name(ecm)] = m
            self.names = sorted(self.models)
        for name, p in self.pending.items():
            m = self.models.get(name)
            if m is not None:
                m.set_world_pose_cmd(ecm, Pose3d(*p))
        self.pending = {}

    def _keep(self, name: str):
        def cb(msg):
            with self.lock:
                self.latest[name] = msg
        return cb

    def sensors(self, want: set[str]) -> None:
        """Subscribe to exactly the wanted cameras: Gazebo renders only those."""
        for name in self.on - want:
            self.node.unsubscribe(TOPICS[name][0])
            with self.lock:
                self.latest.pop(name, None)
        for name in want - self.on:
            topic, typ = TOPICS[name]
            self.node.subscribe(typ, topic, self._keep(name))
        self.on = set(want)

    def step(self, frames: list[dict]) -> None:
        for f in frames:
            self.pending.update(f)
            self.server.run(True, 1, False)
            if self.pace:
                now = time.monotonic()
                if self._due is None or now - self._due > CATCH_UP_S:
                    self._due = now
                self._due += self.basic_ms / 1000
                time.sleep(max(0.0, self._due - time.monotonic()))

    def labels(self) -> list[int]:
        with self.lock:
            msg = self.latest.get("verify")
        return [b.label for b in msg.annotated_box] if msg is not None else []

    def image(self, name: str):
        with self.lock:
            return self.latest.get(name)


def jpeg(msg, quality: int) -> bytes:
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.frombytes("RGB", (msg.width, msg.height), msg.data).save(buf, "JPEG",
                                                                      quality=quality)
    return buf.getvalue()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5554")
    ap.add_argument("--basic-ms", type=int, default=20)
    ap.add_argument("--no-pace", action="store_true", help="step as fast as asked")
    a = ap.parse_args(argv)
    if "GZ_PARTITION" not in os.environ:
        print("[gz] warning: GZ_PARTITION unset, other Gazebos on the network can join",
              file=sys.stderr, flush=True)

    ctx = zmq.Context.instance()
    ctx.setsockopt(zmq.LINGER, 0)
    rep = ctx.socket(zmq.REP)
    rep.bind(a.endpoint)  # bound before the world loads, so the backend's hello just waits
    t0 = time.monotonic()
    bridge = Bridge(a.world, a.basic_ms, not a.no_pace)
    print(f"[gz] {a.world}: {len(bridge.names)} models, loaded in "
          f"{time.monotonic() - t0:.1f} s", flush=True)
    try:
        while True:
            msg = msgpack.unpackb(rep.recv(), raw=False, strict_map_key=False)
            op = msg.get("op")
            out: dict = {}
            if op == "hello":
                out = {"models": bridge.names, "basic_ms": a.basic_ms}
            elif op == "step":
                bridge.sensors(set(msg.get("sensors", [])) & set(TOPICS))
                bridge.step(msg.get("frames", []))
                out = {"labels": bridge.labels(), "t": bridge.sim_t}
            elif op == "frames":
                frames = {}
                for k in msg.get("which", []):
                    img = bridge.image(f"view{k}")
                    if img is not None:
                        frames[k] = {"w": img.width, "h": img.height,
                                     "jpeg": jpeg(img, msg.get("quality", 72))}
                out = {"frames": frames}
            elif op == "preview":
                img = bridge.image("preview")
                if img is not None:
                    out = {"w": img.width, "h": img.height, "rgb": bytes(img.data)}
            elif op == "close":
                rep.send(msgpack.packb({}))
                time.sleep(0.2)  # let the reply leave before os._exit drops it (LINGER 0)
                break
            else:
                out = {"error": f"unknown op {op!r}"}
            rep.send(msgpack.packb(out, use_bin_type=True))
    except KeyboardInterrupt:
        pass
    finally:
        print("[gz] stopped", flush=True)
        os._exit(0)  # the embedded server's threads would otherwise hold the exit


if __name__ == "__main__":
    main()
