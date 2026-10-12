"""Dashboard fan-out: each state is serialized once, big parts only when they changed."""

import asyncio
import json

from earshot.dashboard.server import QUEUE, Hub, offer


def state(t, p=b"\x01\x02", camera=..., readings=()):
    msg = {"t": t, "drones": [
        {"id": 0, "telemetry": {"accepted": 1, "heatmap": {"w": 2, "h": 1, "p": p, "cls": b"\x00\x00"}}},
        {"id": 1, "telemetry": {"accepted": 2}},
    ], "readings": [{"t": t, "drone": 0, "sensor": "SOUND", "bearing": b} for b in readings]}
    if camera is not ...:
        msg["camera"] = camera
    return msg


def heat(msg, drone=0):
    return msg["state"]["drones"][drone]["telemetry"].get("heatmap", "absent")


def test_heatmap_and_camera_only_when_changed():
    hub = Hub()
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
    hub = Hub()
    hub.ingest_state(state(0.1, camera={"rgba": b"\xff", "w": 1, "h": 1}, readings=[10]))
    hub.delta_text()
    hub.ingest_state(state(0.2, readings=[20]))
    snap = json.loads(hub.snapshot_text())
    assert heat(snap)["p"] == "AQI=" and snap["camera"]["rgba"] == "/w=="
    assert [r["bearing"] for r in snap["readings"]] == [10]  # 20 follows in the next delta
    assert [r["bearing"] for r in json.loads(hub.delta_text())["readings"]] == [20]


def test_new_run_drops_old_readings():
    hub = Hub()
    hub.ingest_state(state(50.0, readings=[10]))
    hub.delta_text()
    hub.ingest_state(state(0.1, readings=[20]))
    hub.delta_text()
    assert [r["bearing"] for r in json.loads(hub.snapshot_text())["readings"]] == [20]


def test_lagging_screen_gets_a_fresh_snapshot():
    hub = Hub()
    hub.ingest_state(state(0.1))
    q = asyncio.Queue(maxsize=QUEUE)
    for i in range(QUEUE + 1):
        offer(q, f"delta {i}", hub)
    assert q.qsize() == 1 and "state" in json.loads(q.get_nowait())
