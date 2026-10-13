"""Gazebo backend (ADR 0008): poses go to the bridge, only the cameras that are on render,
the box camera's labels become Source classes, and the generated world names every model the
backend moves. Runs without Gazebo (a fake bridge); the SDF check needs `gz`, the bridge test
needs EARSHOT_GZ_TEST=1 and the bridge venv."""

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from earshot.backend.gazebo import LABELS, N_VIEWS, VERIFY_SIZE, GazeboBackend, pose
from earshot.backend.webots import def_names
from earshot.city import City
from earshot.messages import Mode
from earshot.scenario import Scenario
from earshot.state import Command

ROOT = Path(__file__).resolve().parents[1]


def _scenario():
    return Scenario.load(ROOT / "scenarios/demo.yaml", seed=42)


def _models(sc) -> list[str]:
    names = [f"DRONE_{d.id}" for d in sc.drones] + ["CAMV"] + [f"CAM_{k}" for k in range(N_VIEWS)]
    for src_id, name in def_names(sc.sources).items():
        cls = next(s.cls.name for s in sc.sources if s.id == src_id)
        names += [f"{name}_w{k}" for k in range(9)] if cls == "TRAIN" else [name]
    return names


class FakeBridge:
    def __init__(self, models, labels=()):
        self.models, self.labels, self.calls = models, list(labels), []

    def call(self, msg, timeout_s=None):
        self.calls.append(msg)
        op = msg["op"]
        if op == "hello":
            return {"models": self.models, "basic_ms": 20}
        if op == "step":
            return {"labels": self.labels if "verify" in msg["sensors"] else [], "t": 0.0}
        if op == "frames":
            return {"frames": {k: {"w": 384, "h": 216, "jpeg": b"jpg"} for k in msg["which"]}}
        if op == "preview":
            w, h = VERIFY_SIZE
            return {"w": w, "h": h, "rgb": bytes([10, 20, 30]) * (w * h)}
        return {}

    def close(self):
        pass

    def steps(self):
        return [c for c in self.calls if c["op"] == "step"]


def _backend(labels=()):
    sc = _scenario()
    link = FakeBridge(_models(sc), labels)
    return GazeboBackend(City.load(ROOT / sc.city), sc, link=link), link, sc


def test_one_bridge_call_per_tick_with_a_pose_per_basic_step():
    be, link, sc = _backend()
    states = be.step(0.1, {})
    step = link.steps()[-1]
    assert len(step["frames"]) == 5  # 100 ms tick, 20 ms basic step
    first = step["frames"][0]
    for d in sc.drones:
        x, y = first[f"DRONE_{d.id}"][:2]
        s = states[d.id]
        assert (round(x, 3), round(y, 3)) == (round(s.x, 3), round(s.y, 3))
    assert first["CAMV"][2] < -50  # no Verifier: the camera is parked below ground
    assert step["sensors"] == []  # nothing renders


def test_only_changes_are_sent():
    be, link, _ = _backend()
    be.step(0.1, {})
    be.step(0.1, {})
    last = link.steps()[-1]["frames"][-1]
    assert "CAMV" not in last  # parked and unchanged
    assert not any(k.startswith("SRC_") and v[2] < -50 for k, v in last.items())


def test_verifier_turns_its_camera_on_and_gets_classes():
    be, link, _ = _backend(labels=[LABELS["DRONE"], LABELS["TRAIN"], 99])
    target = (-340.0, -540.0)
    cmd = {0: Command(Mode.VERIFY, goal=target, camera_on=True, look_at=target)}
    be.step(0.1, cmd)
    assert be.camera_frame(1) is None  # not the Verifier
    be.step(0.1, {})
    assert set(link.steps()[-1]["sensors"]) == {"verify", "preview"}
    frame = be.camera_frame(0)
    assert frame["objects"] == ["DRONE", "TRAIN"]  # unknown label ignored
    camv = link.steps()[-1]["frames"][0]["CAMV"]
    assert camv[2] > 0 and math.degrees(camv[4]) == pytest.approx(14.0)


def test_preview_is_bgra():
    be, _, _ = _backend()
    be.step(0.1, {0: Command(Mode.VERIFY, goal=(0.0, 0.0), camera_on=True, look_at=(0.0, 0.0))})
    img = np.frombuffer(be.camera_image(), np.uint8).reshape(VERIFY_SIZE[1], VERIFY_SIZE[0], 4)
    assert tuple(img[0, 0]) == (30, 20, 10, 255)


def test_view_cameras_render_only_while_a_screen_asks():
    be, link, _ = _backend()
    be.step(0.1, {})
    assert be.view_frames([0]) == {}
    be.set_views([[2, "pilot"], [3, "chase"], ["x", "pilot"], [1, "sideways"]])
    assert be.views[0] == [2, "pilot"] and be.views[2][1] == "pilot"  # bad entries ignored
    be.step(0.1, {})
    step = link.steps()[-1]
    assert {f"view{k}" for k in range(N_VIEWS)} <= set(step["sensors"])
    assert "CAM_0" in step["frames"][0]
    frames = be.view_frames([0, 1])
    assert frames[0]["drone"] == 2 and frames[0]["mode"] == "pilot" and frames[0]["jpeg"] == b"jpg"


def test_missing_drone_model_is_reported():
    sc = _scenario()
    with pytest.raises(RuntimeError, match="DRONE_0"):
        GazeboBackend(City.load(ROOT / sc.city), sc, link=FakeBridge(["CAMV"]))


def test_pose_points_x_along_the_heading_and_pitches_down():
    p = pose(1, 2, 3, heading=90.0, pitch_down=10.0)  # east
    assert p[:3] == [1, 2, 3] and p[5] == pytest.approx(0.0) and p[4] == pytest.approx(math.radians(10))
    assert pose(0, 0, 0, heading=0.0)[5] == pytest.approx(math.pi / 2)  # north = +y


@pytest.fixture(scope="module")
def demo_world(tmp_path_factory):
    out = tmp_path_factory.mktemp("gz") / "demo.sdf"
    subprocess.run([sys.executable, str(ROOT / "tools/build_gazebo_world.py"),
                    str(ROOT / "scenarios/demo.yaml"), str(out)], check=True, cwd=ROOT,
                   capture_output=True)
    return out


def test_world_names_every_model_the_backend_moves(demo_world):
    sdf = demo_world.read_text()
    for name in _models(_scenario()):
        assert f'<model name="{name}">' in sdf
    for cls, label in LABELS.items():
        assert f"<label>{label}</label>" in sdf, cls
    assert 'type="boundingbox_camera"' in sdf and "fuel" not in sdf.lower()
    assert sdf.count("<visual ") < 80  # the label render costs per visual


@pytest.mark.skipif(shutil.which("gz") is None, reason="Gazebo not installed")
def test_world_is_valid_sdf(demo_world):
    r = subprocess.run(["gz", "sdf", "-k", str(demo_world)], capture_output=True, text=True,
                       check=False)
    assert r.returncode == 0 and "Valid" in r.stdout, r.stdout + r.stderr


@pytest.mark.skipif(os.environ.get("EARSHOT_GZ_TEST") != "1"
                    or not (ROOT / "gazebo/.venv/bin/python").exists(),
                    reason="set EARSHOT_GZ_TEST=1 (needs Gazebo and the bridge venv)")
def test_bridge_steps_gazebo_and_sees_the_intruder(demo_world):
    from earshot.backend.gazebo import BridgeLink

    env = dict(os.environ, GZ_IP="127.0.0.1", GZ_PARTITION="earshot_test")
    endpoint = "tcp://127.0.0.1:5564"
    proc = subprocess.Popen([str(ROOT / "gazebo/.venv/bin/python"),
                             str(ROOT / "src/earshot/backend/gz_bridge.py"), str(demo_world),
                             "--no-pace", "--endpoint", endpoint], env=env)
    try:
        sc = _scenario()
        be = GazeboBackend(City.load(ROOT / sc.city), sc, link=BridgeLink(endpoint))
        drone1 = next(s for s in sc.sources if s.id == "drone1")
        seen = set()
        for k in range(900):
            t = k * 0.1
            cmd = {}
            if t >= 60:
                x, y, _ = drone1.pose(t)
                cmd = {0: Command(Mode.VERIFY, goal=(x, y), camera_on=True, look_at=(x, y))}
            be.step(0.1, cmd)
            seen.update((be.camera_frame(0) or {}).get("objects", []))
        assert seen == {"DRONE"}
        be.close()
    finally:
        proc.wait(timeout=10)
