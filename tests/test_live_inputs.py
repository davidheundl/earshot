"""Live-mode inputs: bad Events and view requests are rejected whole, live Sources get their
own Webots node slot per class, and an agent that falls behind loses no readings."""

import copy
import math
from pathlib import Path

import pytest

from earshot.backend.webots import WebotsBackend, def_names
from earshot.messages import Mode, Sensor
from earshot.runners.agent_proc import merge_ticks
from earshot.runners.lockstep import build
from earshot.scenario import Scenario
from earshot.state import DroneState, Sensed, SensorReading

ROOT = Path(__file__).resolve().parents[1]


def _scenario():
    sc = Scenario.load(ROOT / "scenarios/demo.yaml", seed=42)
    sc.duration = 30.0
    return sc


BAD_OPS = [
    {"op": "add_source", "cls": 1, "x": 0, "y": 0},
    {"op": "add_source", "cls": "ufo", "x": 0, "y": 0},
    {"op": "add_source", "cls": "drone", "path": []},
    {"op": "add_source", "cls": "drone", "path": [[0, 0, 0]]},
    {"op": "add_source", "cls": "drone", "path": [[0, "a"]]},
    {"op": "add_source", "cls": "drone", "x": "a", "y": 0},
    {"op": "add_source", "cls": "drone", "x": math.nan, "y": 0},
    {"op": "add_source", "cls": "drone", "x": 0, "y": 0, "to": [1]},
    {"op": "add_source", "cls": "drone", "x": 0, "y": 0, "duration": -1},
    {"op": "add_source", "cls": "drone", "x": 0, "y": 0, "level_db": "loud"},
    {"op": "add_source", "cls": "drone", "x": 0, "y": 0, "rf": "2g4"},
    {"op": "add_source", "cls": "drone", "x": 0, "y": 0, "rf": ["5g9"]},
    {"op": "add_source", "cls": "drone", "x": 0, "y": 0, "id": 7},
    {"op": "add_source", "cls": "drone"},
    {"op": "jam", "x": "100", "y": 0},
    {"op": "jam", "x": 0, "y": 0, "r": -5},
    {"op": "jam", "x": 0, "y": 0, "r": math.inf},
    {"op": "jam", "x": 0, "y": 0, "duration": 0},
    {"op": "fault", "drone": "two", "kind": "deaf"},
    {"op": "fault", "drone": 2, "kind": "sleepy"},
    {"op": "clear_faults", "drone": [2]},
    {"op": "set_env", "background_db": "loud"},
    {"op": "set_env", "background_db": math.nan},
    {"op": "set_env", "ego_filter": "no"},
    {"op": "teleport"},
]


@pytest.mark.parametrize("op", BAD_OPS, ids=lambda op: str(op))
def test_bad_op_raises_and_changes_nothing(op):
    sc = _scenario()
    sc.apply({"op": "fault", "drone": 1, "kind": "deaf"}, t=1.0)
    before = (copy.deepcopy(sc.sources), copy.deepcopy(sc.jamming), copy.deepcopy(sc.faults),
              copy.deepcopy(sc.environment), sc._live_count)
    with pytest.raises((ValueError, KeyError)):
        sc.apply(op, t=5.0)
    after = (sc.sources, sc.jamming, sc.faults, sc.environment, sc._live_count)
    assert after == before


def test_valid_ops_keep_their_values():
    sc = _scenario()
    sc.apply({"op": "add_source", "cls": "Truck", "path": [[0, 1], [2.5, 3]], "to": [4, 5],
              "rf": ["2g4"]}, t=1.0)
    src = sc.sources[-1]
    assert src.path == [(0, 1), (2.5, 3), (4, 5)] and src.rf_bands == ("2g4",)
    sc.apply({"op": "jam", "x": 0, "y": -600}, t=1.0)
    assert sc.jamming[-1].center == (0, -600) and isinstance(sc.jamming[-1].center[0], int)
    sc.apply({"op": "set_env", "background_db": 55, "ego_filter": False}, t=1.0)
    assert sc.environment.background_db == 55 and sc.environment.ego_filter is False


def test_bad_live_events_never_stop_the_world():
    _, world, _, _ = build(_scenario(), ROOT)
    world.step({})
    n_sources, n_jams = len(world.scenario.sources), len(world.scenario.jamming)
    for op in BAD_OPS:
        world.control(dict(op))
    for _ in range(20):
        world.step({})
    notes = [e["note"] for e in world.applied]
    assert len(notes) == len(BAD_OPS) and all(n.startswith("rejected") for n in notes)
    assert len(world.scenario.sources) == n_sources and len(world.scenario.jamming) == n_jams


def test_live_sources_get_a_slot_per_class():
    sc = _scenario()
    for cls in ("drone", "nuisance", "drone", "drone"):
        sc.apply({"op": "add_source", "cls": cls, "x": 0, "y": 0}, t=1.0)
    live = [s for s in sc.sources if s.id.startswith("live_")]
    assert len({s.id for s in live}) == len(live)  # ids stay unique
    names = def_names(sc.sources)
    assert [names[s.id] for s in live] == [
        "SRC_live_drone_slot0", "SRC_live_nothing_slot0", "SRC_live_drone_slot1",
        "SRC_live_drone_slot0",  # only the third drone wraps around
    ]
    scripted = [s for s in sc.sources if not s.id.startswith("live_")]
    assert all(names[s.id] == f"SRC_{s.id}" for s in scripted)


def test_malformed_view_requests_are_ignored():
    b = object.__new__(WebotsBackend)  # no Webots needed for set_views
    b._drones = {0: None, 1: None}
    b._view_cams = [None, None]
    b.views = [[0, "chase"], [0, "pilot"]]
    for bad in (None, 3, "x", {"a": 1}, [[0]], [[[1], "pilot"]], [[1, "sideways"]],
                [[9, "pilot"]], [None, [1]]):
        b.set_views(bad)
    assert b.views == [[0, "chase"], [0, "pilot"]]
    b.set_views([[1, "pilot"], [0]])
    assert b.views == [[1, "pilot"], [0, "pilot"]]


def _tick(t, readings):
    st = DroneState(1, t, 0.0, 0.0, 40.0, 0.0, Mode.LISTEN, None, 1.0, 0.0, False)
    return Sensed(st, [SensorReading(Sensor.SOUND, b, 5.0, (1.0, 0.0, 0.0, 0.0), 50.0)
                       for b in readings])


def test_merge_ticks_keeps_skipped_readings():
    ticks = [_tick(1.0, [10.0]), _tick(1.1, []), _tick(1.2, [20.0, 30.0])]
    merged = merge_ticks(ticks)
    assert merged.state == ticks[-1].state
    assert [r.bearing for r in merged.readings] == [10.0, 20.0, 30.0]
    assert merge_ticks(ticks[-1:]) is ticks[-1]
