"""Live-run counters and Webots spare nodes: what the operator screens show must not overstate."""

from pathlib import Path
from types import SimpleNamespace

from earshot.backend.webots import WEBOTS_LIVE_SLOTS, def_names
from earshot.messages import TargetClass
from earshot.runners.world_proc import VerifyCounts

ROOT = Path(__file__).resolve().parents[1]


def ev(kind, tid, x=0.0, y=0.0, drone=0, task="verify"):
    return {"t": 1.0, "kind": kind, "drone": drone, "task": task, "tid": tid, "x": x, "y": y}


def test_a_task_counts_once_however_many_drones_report_it():
    c = VerifyCounts()
    c.add(ev("dispatch", 7, drone=0))
    c.add({"t": 1.1, "kind": "outbid", "drone": 0})  # no tid: never counted
    c.add(ev("dispatch", 7, drone=1))  # the nearer drone takes the same task
    c.add(ev("dispatch", 9, task="listen"))  # not a verification
    assert c.as_dict() == {"dispatch": 1, "verified": 0, "confirmed": 0, "dismissed": 0}
    c.add(ev("dismissed", 7, drone=1))
    c.add(ev("dismissed", 7, drone=1))
    assert c.as_dict() == {"dispatch": 1, "verified": 1, "confirmed": 0, "dismissed": 1}


def test_one_intruder_confirmed_by_several_drones_counts_once():
    c = VerifyCounts()
    c.add(ev("confirmed", 1, 100.0, 100.0, drone=0))
    c.add(ev("confirmed", 1, 102.0, 100.0, drone=2))  # same task, another drone
    c.add(ev("confirmed", 4, 180.0, 160.0, drone=3))  # another task, within 150 m
    c.add(ev("confirmed", 5, 900.0, 100.0, drone=4))  # a second intruder
    d = c.as_dict()
    assert d["confirmed"] == 2 and d["verified"] == 3


def test_webots_live_sources_beyond_the_spare_nodes_get_none():
    n = WEBOTS_LIVE_SLOTS
    src = [SimpleNamespace(id="a", cls=TargetClass.DRONE)] + [
        SimpleNamespace(id=f"live_drone_{k}", cls=TargetClass.DRONE) for k in range(n + 1)]
    names = def_names(src, n, wrap=False)
    assert names["a"] == "SRC_a"
    slots = [names.get(f"live_drone_{k}") for k in range(n + 1)]
    assert len(set(slots[:n])) == n and slots[-1] is None  # nobody's node taken


class _GoneSupervisor:
    def __init__(self):
        self.steps = 0

    def stepBegin(self, _ms):
        self.steps += 1
        return -1  # Webots quit

    def stepEnd(self):
        raise AssertionError("no step is open after Webots quit")


def test_webots_quitting_leaves_the_twin_running(monkeypatch):
    from earshot.backend import webots
    from earshot.backend.twin2d import Twin2D
    from earshot.city import City
    from earshot.scenario import Scenario

    sc = Scenario.load(ROOT / "scenarios/demo.yaml")
    b = object.__new__(webots.WebotsBackend)
    b.sup, b.basic_ms, b.scenario, b._quit, b._due = _GoneSupervisor(), 50, sc, False, None
    b._open, b._grabbed = False, {}
    b._verifier, b._views_on = 3, True
    b.twin = Twin2D(City.load(ROOT / sc.city), sc.drones)
    monkeypatch.setattr(b, "_push", lambda _states: None)
    monkeypatch.setattr(webots.time, "sleep", lambda _s: None)
    for _ in range(3):
        states = b.step(0.1, {})
    assert b._quit and b.sup.steps == 1 and len(states) == len(sc.drones)  # never asked again
    assert b.twin.t > 0.25 and b.camera_frame(3) is None and b.view_frames([0]) == {}


def test_webots_refuses_a_live_source_without_a_free_node():
    from earshot.backend.webots import WebotsBackend
    from earshot.scenario import Scenario

    b = object.__new__(WebotsBackend)
    b.scenario = Scenario.load(ROOT / "scenarios/demo.yaml")
    op = {"op": "add_source", "cls": "drone", "x": 0, "y": 0}
    for _ in range(WEBOTS_LIVE_SLOTS):
        assert b.refuses(op) is None
        b.scenario.apply(op, t=1.0)
    assert "spare Webots nodes" in b.refuses(op)
    assert b.refuses({**op, "cls": "nuisance"}) is None and b.refuses({"op": "jam"}) is None
