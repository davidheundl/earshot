import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from earshot.eval import RunMetrics, as_json, censored_mean, evaluate, metrics, summarize
from earshot.runners.lockstep import RunResult
from earshot.tune import W_MISS, run_cost

ROOT = Path(__file__).resolve().parents[1]
DURATION = 300.0


def _drone(id_, start, x, y):
    return {"id": id_, "cls": "DRONE", "start": start, "end": DURATION, "path": [[x, y]]}


def _run(events=(), truth=None, jams=None, spread=None, listeners=None) -> RunResult:
    truth = truth if truth is not None else [_drone("a", 10.0, 0.0, 0.0),
                                             _drone("b", 50.0, 1000.0, 0.0)]
    return RunResult("synthetic", 7, 0, {"channel_load": 0.1, "delivered": 9, "collided": 1},
                     {}, list(events), truth, {0: 0.9}, 1, spread or [], jams or [],
                     listeners=listeners or [])


def _ev(kind, t, x, y):
    return {"kind": kind, "task": "verify", "t": t, "x": x, "y": y}


def test_a_missed_second_drone_counts_as_the_duration():
    m = metrics(_run([_ev("confirmed", 40.0, 5.0, 0.0)]), "earshot", DURATION)
    assert m.t_confirm_each == [30.0, None]
    assert m.t_confirm == pytest.approx((30.0 + DURATION) / 2)  # not min = 30: b is not hidden
    assert (m.drones, m.confirmed, m.missed) == (2, 1, 1)
    assert m.t_dispatch == DURATION  # nobody dispatched to either


def test_both_drones_confirmed_gives_the_mean_latency():
    m = metrics(_run([_ev("confirmed", 40.0, 0.0, 0.0), _ev("confirmed", 110.0, 1000.0, 0.0),
                      _ev("dispatch", 20.0, 0.0, 0.0), _ev("dispatch", 60.0, 1000.0, 0.0)]),
                "earshot", DURATION)
    assert m.t_confirm == pytest.approx((30.0 + 60.0) / 2)
    assert m.t_dispatch == pytest.approx((10.0 + 10.0) / 2)
    assert m.missed == 0


def test_no_drone_source_leaves_latencies_undefined():
    truth = [{"id": "t", "cls": "TRAIN", "start": 0.0, "end": DURATION, "path": [[0, 0]]}]
    m = metrics(_run(truth=truth), "earshot", DURATION)
    assert m.t_confirm is None and m.drones == 0 and m.missed == 0
    assert run_cost(m)["cost"] == pytest.approx(10.0)  # only the Gap (busy_max 1)


def test_tune_cost_sees_the_second_intruder():
    one = metrics(_run([_ev("confirmed", 40.0, 0.0, 0.0)]), "earshot", DURATION)
    both = metrics(_run([_ev("confirmed", 40.0, 0.0, 0.0), _ev("confirmed", 110.0, 1000.0, 0.0)]),
                   "earshot", DURATION)
    c1, c2 = run_cost(one), run_cost(both)
    assert c1["t_confirm"] > c2["t_confirm"]
    assert c1["cost"] - c2["cost"] == pytest.approx(W_MISS + c1["t_confirm"] - c2["t_confirm"])


def test_loc_err_counts_every_dispatch():
    # two good dispatches (10 m, 20 m) and three far ones: the old within-150 m median was 15
    far = [_ev("dispatch", 60.0, 400.0, 0.0), _ev("dispatch", 70.0, 500.0, 0.0),
           _ev("dispatch", 80.0, 0.0, 600.0)]
    m = metrics(_run([_ev("dispatch", 20.0, 10.0, 0.0), _ev("dispatch", 30.0, 0.0, 20.0), *far]),
                "earshot", DURATION)
    assert m.loc_err == pytest.approx(400.0)
    assert m.false_verify == 3 and m.verify_flights == 5


def test_a_dispatch_before_any_drone_is_infinitely_wrong_and_json_survives():
    m = metrics(_run([_ev("dispatch", 5.0, 0.0, 0.0)]), "earshot", DURATION)
    assert math.isinf(m.loc_err) and m.false_verify == 1
    back = json.loads(json.dumps(as_json([m])))
    assert math.isinf(back[0]["loc_err"])
    assert summarize([m])["loc_err_m"].startswith("inf")


def test_time_to_agree_tells_no_jam_from_never_agreed():
    no_jam = metrics(_run(), "earshot", DURATION)
    assert (no_jam.t_agree, no_jam.agreed) == (None, None)
    never = metrics(_run(jams=[(100.0, 200.0)], spread=[(210.0, 500.0), (220.0, None)]),
                    "earshot", DURATION)
    assert (never.t_agree, never.agreed) == (DURATION - 200.0, False)
    agreed = metrics(_run(jams=[(100.0, 200.0)], spread=[(150.0, 10.0), (230.0, 50.0)]),
                     "earshot", DURATION)
    assert (agreed.t_agree, agreed.agreed) == (30.0, True)
    outlasted = metrics(_run(jams=[(100.0, 400.0)]), "earshot", DURATION)
    assert (outlasted.t_agree, outlasted.agreed) == (None, None)


def _row(seed, t_confirm, confirmed, drones=1, agreed=None, t_agree=None):
    return RunMetrics(seed, "earshot", t_confirm, t_confirm, None, 0, 0, drones - confirmed, 0,
                      0.1, 0.9, t_agree, 0.1, duration=DURATION, drones=drones,
                      confirmed=confirmed, agreed=agreed)


def test_summary_keeps_misses_in_the_median():
    rows = [_row(0, 10.0, 1), _row(1, 20.0, 1), _row(2, DURATION, 0), _row(3, DURATION, 0),
            _row(4, DURATION, 0, agreed=False, t_agree=100.0),
            _row(5, 30.0, 1, agreed=True, t_agree=5.0)]
    s = summarize(rows)
    assert s["confirmed"] == "3/6"
    assert s["t_confirm_s"].startswith(f"{(30.0 + DURATION) / 2:.1f}")  # survivors alone: 20
    assert s["agreed"] == "1/2"
    assert s["t_agree_s"].startswith("52.5")  # only the jammed runs
    assert summarize(rows[:2])["agreed"] == "-"


def test_censored_mean():
    assert censored_mean([], 9.0) is None
    assert censored_mean([1.0, None], 9.0) == 5.0


def test_parallel_eval_matches_serial():
    scenario = str(ROOT / "scenarios/demo.yaml")
    serial = evaluate(scenario, 2, 3, "earshot", duration=90.0)
    parallel = evaluate(scenario, 2, 3, "earshot", jobs=2, duration=90.0)
    assert [asdict(r) for r in serial] == [asdict(r) for r in parallel]


def test_loc_err_splits_into_range_and_bearing_seen_from_the_nearest_listener():
    # listener due south of drone a: an error of (30, 40) is 40 m in range, 30 m in bearing;
    # the far listener and the one that only perches after the dispatch do not count
    listeners = [(0.0, [(0.0, -500.0), (5000.0, 5000.0)]), (25.0, [(-500.0, 0.0)])]
    m = metrics(_run([_ev("dispatch", 20.0, 30.0, 40.0)], listeners=listeners),
                "earshot", DURATION)
    assert m.loc_err == pytest.approx(50.0)
    assert m.loc_radial == pytest.approx(40.0) and m.loc_tangential == pytest.approx(30.0)
    row = summarize([m])
    assert row["loc_radial_m"].startswith("40.0") and row["loc_tangential_m"].startswith("30.0")


def test_loc_err_split_needs_a_listener_and_a_drone_in_the_air():
    nobody = metrics(_run([_ev("dispatch", 20.0, 30.0, 40.0)]), "earshot", DURATION)
    early = metrics(_run([_ev("dispatch", 5.0, 30.0, 40.0)], listeners=[(0.0, [(0.0, -500.0)])]),
                    "earshot", DURATION)
    for m in (nobody, early):
        assert m.loc_radial is None and m.loc_tangential is None
    assert summarize([nobody])["loc_radial_m"] == "-"
