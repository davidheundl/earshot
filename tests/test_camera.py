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


# -- through the SensorSuite: what the World hands the Agents -----------------------------

def _suite(script):
    from earshot.city import City
    from earshot.scenario import Scenario
    from earshot.sensors import SensorSuite

    root = Path(__file__).resolve().parents[1]
    sc = Scenario.load(root / "scenarios/demo.yaml")
    det = YoloDetector(model=FakeModel(script), device="cpu")
    return SensorSuite(City.load(root / sc.city), sc, np.random.default_rng(0), camera=det), det


def _verifier(t, x=0.0, mode=None, camera=True, drone=3):
    from earshot.messages import Mode
    from earshot.state import DroneState

    return DroneState(drone, t, x, 0.0, 40.0, 90.0, mode or Mode.VERIFY, None, 1.0, 5700.0,
                      camera)


def _camera_readings(dets):
    from earshot.messages import Sensor

    return [d.reading for d in dets if d.reading.sensor == Sensor.CAMERA]


def test_a_second_sortie_starts_from_a_clean_view():
    from earshot.messages import Mode

    suite, _ = _suite([[(1, 0.8, BOX)]] * 3 + [[]] * 50)
    hits = []
    for k in range(3):  # first sortie: three frames of one track, verified
        hits += _camera_readings(suite.sense(_verifier(10.0 + k * 0.1), 10.0 + k * 0.1, frame()))
    suite.sense(_verifier(20.0, mode=Mode.LISTEN, camera=False), 20.0, None)  # back on its roof
    later = []
    for k in range(20):  # second sortie, nothing in view
        t = 40.0 + k * 0.1
        later += _camera_readings(suite.sense(_verifier(t), t, frame()))
    assert all(r.class_probs[0] < 0.5 for r in later)  # no stale drone from the first sortie


def test_no_nothing_while_flying_in():
    suite, _ = _suite([[]] * 200)
    readings = []
    for k in range(60):  # 6 s at 15 m/s: empty views, but on the move
        t = 50.0 + k * 0.1
        readings += _camera_readings(suite.sense(_verifier(t, x=15.0 * k * 0.1), t, frame()))
    assert readings == []
    for k in range(40):  # holding still: three empty windows, then "nothing" for 40 m
        t = 56.0 + k * 0.1
        readings += _camera_readings(suite.sense(_verifier(t, x=90.0), t, frame()))
    assert readings and readings[-1].class_probs[3] == 1.0 and readings[-1].level == 40.0


def test_a_verifier_without_the_rig_stays_inconclusive():
    suite, _ = _suite([[]] * 200)
    for k in range(30):  # drone 3 has the rig
        t = 50.0 + k * 0.1
        suite.sense(_verifier(t), t, frame())
    other = []
    for k in range(30):  # drone 4 verifies too, without images: no ground-truth fallback
        t = 50.0 + k * 0.1
        other += _camera_readings(suite.sense(_verifier(t, drone=4), t, None))
    assert other == []


def test_a_detector_that_breaks_hands_over_to_the_stand_in():
    class Broken(FakeModel):
        def track(self, img, **kw):
            raise RuntimeError("MPS out of memory")

    suite, _ = _suite([])
    suite.camera.model = Broken([])
    suite.sense(_verifier(1.0), 1.0, frame())
    assert suite.camera is None
