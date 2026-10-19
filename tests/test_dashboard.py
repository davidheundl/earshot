"""Dashboard fan-out: each state is serialized once, big parts only when they changed."""

import asyncio
import json
import math

from earshot.dashboard.server import QUEUE, Hub, offer


def state(t, p=b"\x01\x02", camera=..., readings=()):
    msg = {"t": t, "drones": [
        {"id": 0, "telemetry": {"accepted": 1, "heatmap": {"w": 2, "h": 1, "p": p, "cls": b"\x00\x00"}}},
        {"id": 1, "telemetry": {"accepted": 2}},
    ], "readings": [{"t": t, "drone": 0, "sensor": "SOUND", "bearing": b} for b in readings]}
    if camera is not ...:
        msg["camera"] = camera
    return msg


def camera_hub():
    hub = Hub()
    hub.want_camera = True  # a classic page is connected
    return hub


def heat(msg, drone=0):
    return msg["state"]["drones"][drone]["telemetry"].get("heatmap", "absent")


def test_heatmap_and_camera_only_when_changed():
    hub = camera_hub()
    assert hub.delta_text() is None
    hub.ingest_state(state(0.1, camera={"rgba": b"\xff", "w": 1, "h": 1}, readings=[10]))
    first = json.loads(hub.delta_text())
    assert heat(first)["p"] == "AQI=" and first["camera"]["rgba"] == "/w=="
    assert [r["bearing"] for r in first["readings"]] == [10]
    assert hub.delta_text() is None  # nothing new, nothing sent

    hub.ingest_state(state(0.2, readings=[20]))
    second = json.loads(hub.delta_text())
    assert heat(second) == "absent" and "camera" not in second
    assert heat(second, 1) == "absent" and second["state"]["drones"][0]["telemetry"]["accepted"] == 1
    assert [r["bearing"] for r in second["readings"]] == [20]

    hub.ingest_state(state(0.3, p=b"\x03\x04", camera=None))
    third = json.loads(hub.delta_text())
    assert heat(third)["p"] == "AwQ=" and third["camera"] is None


def test_snapshot_has_everything_and_no_double_readings():
    hub = camera_hub()
    hub.ingest_state(state(0.1, camera={"rgba": b"\xff", "w": 1, "h": 1}, readings=[10]))
    hub.delta_text()
    hub.ingest_state(state(0.2, readings=[20]))
    snap = json.loads(hub.snapshot_text())
    assert heat(snap)["p"] == "AQI=" and snap["camera"]["rgba"] == "/w=="
    assert [r["bearing"] for r in snap["readings"]] == [10]  # 20 follows in the next delta
    assert [r["bearing"] for r in json.loads(hub.delta_text())["readings"]] == [20]


def test_first_camera_counts_even_when_none():
    hub = camera_hub()
    hub.ingest_state(state(0.1, camera=None))
    msg = json.loads(hub.delta_text())
    assert "camera" in msg and msg["camera"] is None


def test_new_run_drops_old_readings_and_resends_the_kept_parts():
    hub = camera_hub()
    hub.ingest_state(state(50.0, camera=None, readings=[10]))
    hub.delta_text()
    hub.ingest_state(state(0.1, camera=None, readings=[20]))
    delta = json.loads(hub.delta_text())  # the pages cleared their copies on the new run
    assert heat(delta)["p"] == "AQI=" and "camera" in delta
    assert [r["bearing"] for r in json.loads(hub.snapshot_text())["readings"]] == [20]


def test_lagging_screen_gets_a_fresh_snapshot():
    hub = Hub()
    hub.ingest_state(state(0.1))
    q = asyncio.Queue(maxsize=QUEUE)
    for i in range(QUEUE + 1):
        offer(q, f"delta {i}", hub)
    assert q.qsize() == 1 and "state" in json.loads(q.get_nowait())


def test_camera_only_while_a_classic_page_wants_it():
    hub = Hub()
    hub.ingest_state(state(0.1, camera={"rgba": b"\xff", "w": 1, "h": 1}))
    assert "camera" not in json.loads(hub.delta_text())
    assert "camera" not in json.loads(hub.snapshot_text())
    hub.want_camera = True  # a classic page joins: its snapshot carries the camera
    assert json.loads(hub.snapshot_text())["camera"]["rgba"] == "/w=="


def test_nan_and_infinity_become_null():
    hub = Hub()
    msg = state(0.1, readings=[math.nan])
    msg["drones"][1]["telemetry"]["score"] = math.inf
    hub.ingest_state(msg)
    out = json.loads(hub.delta_text())  # strict JSON: the browser's JSON.parse rejects NaN
    assert out["readings"][0]["bearing"] is None
    assert out["state"]["drones"][1]["telemetry"]["score"] is None
    assert "NaN" not in hub.snapshot_text()
