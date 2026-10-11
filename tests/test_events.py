from pathlib import Path

from earshot.messages import TargetClass
from earshot.runners.lockstep import build, run
from earshot.scenario import Scenario

ROOT = Path(__file__).resolve().parents[1]


def _scenario(seed=42, duration=60.0):
    sc = Scenario.load(ROOT / "scenarios/demo.yaml", seed=seed)
    sc.duration = duration
    return sc


def test_apply_ops():
    sc = _scenario()
    sc.apply({"op": "add_source", "cls": "drone", "x": 10, "y": 20}, t=5.0)
    src = sc.sources[-1]
    assert src.cls == TargetClass.DRONE and src.start == 5.0 and src.rf_bands == ("2g4",)
    assert src.id.startswith("live_drone_")

    sc.apply({"op": "jam", "x": 0, "y": 0, "r": 100}, t=5.0)
    assert len(sc.jam_zones(6.0)) == 1
    sc.apply({"op": "clear_jam"}, t=7.0)
    assert sc.jam_zones(8.0) == []

    sc.apply({"op": "fault", "drone": 2, "kind": "deaf"}, t=5.0)
    assert [f.kind for f in sc.faults_for(2, 6.0)] == ["deaf"]
    sc.apply({"op": "clear_faults", "drone": 2}, t=7.0)
    assert sc.faults_for(2, 8.0) == []

    sc.apply({"op": "set_env", "background_db": 55, "ego_filter": False}, t=5.0)
    assert sc.environment.background_db == 55 and not sc.environment.ego_filter


def test_scripted_event_fires_at_its_time():
    sc = _scenario()
    sc.events = [{"t": 3.0, "op": "add_source", "cls": "train",
                  "path": [[-1000, 820], [1000, 840]], "duration": 30}]
    _, world, _, _ = build(sc, ROOT)
    for _ in range(25):
        world.step({})
    assert world.applied == []
    for _ in range(10):
        world.step({})
    assert len(world.applied) == 1 and not world.applied[0].get("live")


def test_live_control_applies_next_step_and_is_tagged():
    _, world, _, _ = build(_scenario(), ROOT)
    world.step({})
    world.control({"op": "jam", "x": 0, "y": 0})
    world.step({})
    assert world.applied[-1]["live"] is True
    assert world.scenario.jam_zones(world.t)


def test_bad_op_is_rejected_not_fatal():
    _, world, _, _ = build(_scenario(), ROOT)
    world.control({"op": "teleport"})
    world.step({})
    assert world.applied[-1]["note"].startswith("rejected")


def test_replayed_events_are_deterministic():
    events = [{"t": 2.0, "op": "add_source", "cls": "drone", "x": -600, "y": -560},
              {"t": 10.0, "op": "jam", "x": 0, "y": -600}]

    def once():
        sc = _scenario()
        sc.events = [dict(e) for e in events]
        return run(sc, root=ROOT)

    a, b = once(), once()
    assert a.radio == b.radio and a.agents == b.agents
