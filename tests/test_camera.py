"""The Verifier camera detector: bearing geometry, multi-frame verification, "nothing" only
after repeated empty looks, and the real model on Nikolas' recorded Webots frames."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from earshot.sensors.camera import DEFAULT_WEIGHTS, YoloDetector, box_bearing, load_detector

HIGHRES = Path(__file__).resolve().parents[1] / "data/cv/verifier-highres-20261015"


PITCH = math.radians(14)


def frame(w=1280, h=768, yaw=90.0, pitch=PITCH):
    return {"bgra": bytes(w * h * 4), "w": w, "h": h, "fov": 1.0, "yaw": yaw, "pitch": pitch}


def test_box_in_the_centre_points_where_the_camera_looks():
    f = frame(yaw=37.0)
    b, sigma = box_bearing(640, 384, 40, f)
    assert b == pytest.approx(37.0)
    assert sigma >= 2.0


def test_box_on_the_right_edge_is_half_the_field_of_view_clockwise():
    f = frame(yaw=0.0, pitch=0.0)
    b, _ = box_bearing(1280, 384, 10, f)
    assert b == pytest.approx(math.degrees(0.5), abs=0.01)


def test_pitch_corrects_corner_bearings():
    f = frame(yaw=0.0)
    flat, _ = box_bearing(1200, 700, 10, {**f, "pitch": 0.0})
    pitched, _ = box_bearing(1200, 700, 10, f)
    assert pitched != pytest.approx(flat, abs=0.5)


class FakeModel:
    """model.track(...) returning scripted boxes: [(track id or None, score, (x1, y1, x2, y2))]."""

    def __init__(self, script):
        self.names = {0: "quadcopter", 1: "fixed-wing"}
        self.script = list(script)

    def track(self, img, **kw):
        boxes = [SimpleNamespace(cls=[0], conf=[s], id=None if i is None else [i],
                                 xyxy=[SimpleNamespace(tolist=lambda b=b: list(b))])
                 for i, s, b in self.script.pop(0)]
        return [SimpleNamespace(boxes=boxes, speed={"inference": 5.0})]


BOX = (620, 360, 660, 400)


def test_a_track_is_verified_after_three_consecutive_frames():
    det = YoloDetector(model=FakeModel([[(1, 0.8, BOX)]] * 3), device="cpu")
    det.observe(frame())
    det.observe(frame())
    assert det.verdict().hits == []  # two frames are not enough
    det.observe(frame())
    v = det.verdict()
    assert len(v.hits) == 1 and v.hits[0][0] == pytest.approx(90.0, abs=1.0)
    assert v.clear_radius is None


def test_a_missed_frame_resets_the_streak():
    det = YoloDetector(model=FakeModel([[(1, 0.8, BOX)], [(1, 0.8, BOX)], [], [(1, 0.8, BOX)]]),
                       device="cpu")
    for _ in range(4):
        det.observe(frame())
    assert det.verdict().hits == []


def test_weak_boxes_neither_verify_nor_clear():
    det = YoloDetector(model=FakeModel([[(1, 0.3, BOX)]] * 9), device="cpu")
    for _ in range(3):
        for _ in range(3):
            det.observe(frame())
        v = det.verdict()
        assert v.hits == [] and v.clear_radius is None  # inconclusive: something is there


def test_nothing_only_after_repeated_empty_windows():
    det = YoloDetector(model=FakeModel([[]] * 3), device="cpu", clear_radius=40.0)
    verdicts = []
    for _ in range(3):
        det.observe(frame())
        verdicts.append(det.verdict().clear_radius)
    assert verdicts == [None, None, 40.0]


def test_scenario_spec():
    assert load_detector(None) is None
    assert load_detector({"detector": "truth"}) is None
    det = load_detector({"detector": "yolo", "verify_conf": 0.5, "standoff": 20})
    assert det.verify_conf == 0.5


@pytest.mark.skipif(not DEFAULT_WEIGHTS.exists(), reason="no drone-yolo weights")
def test_real_model_verifies_a_recorded_drone_and_clears_an_empty_view():
    pytest.importorskip("ultralytics")
    pytest.importorskip("lap")
    from PIL import Image

    labels = json.loads((HIGHRES / "labels.json").read_text())
    present = next(r for r in labels if r["source_present"] and r["horizontal_distance_m"] == 15)
    empty = next(r for r in labels if not r["source_present"])

    def as_frame(r):
        rgb = np.asarray(Image.open(HIGHRES / r["image"]).convert("RGB"))
        bgra = np.concatenate([rgb[..., ::-1], np.full(rgb.shape[:2] + (1,), 255, np.uint8)],
                              axis=2)
        return {"bgra": bgra.tobytes(), "w": rgb.shape[1], "h": rgb.shape[0], "fov": 1.0,
                "yaw": 0.0, "pitch": math.radians(14)}

    det = YoloDetector(verify_conf=0.5, device="cpu")
    f = as_frame(present)
    for _ in range(3):
        det.observe(f)
    v = det.verdict()
    x1, y1, x2, y2 = present["objects"][0]["bbox_xyxy"]
    truth, _ = box_bearing((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, f)
    assert v.hits, "the drone at 15 m was not verified"
    assert abs((v.hits[0][0] - truth + 180) % 360 - 180) < 2.0

    det.reset()
    e = as_frame(empty)
    radii = []
    for _ in range(3):
        det.observe(e)
        radii.append(det.verdict().clear_radius)
    assert radii[-1] == det.clear_radius
