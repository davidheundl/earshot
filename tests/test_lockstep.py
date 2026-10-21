from pathlib import Path

from earshot.runners.lockstep import run
from earshot.scenario import Scenario

ROOT = Path(__file__).resolve().parents[1]


def _short(seed=42):
    sc = Scenario.load(ROOT / "scenarios/demo.yaml", seed=seed)
    sc.duration = 120.0
    return sc


def test_six_agents_exchange_messages():
    r = run(_short(), root=ROOT)
    assert r.radio["delivered"] > 0
    assert len(r.agents) == 6
    assert sum(a["accepted"] for a in r.agents.values()) > 0


def test_same_seed_same_run():
    a, b = run(_short(), root=ROOT), run(_short(), root=ROOT)
    assert a.radio == b.radio
    assert a.agents == b.agents


def test_different_seed_differs():
    assert run(_short(1), root=ROOT).radio != run(_short(2), root=ROOT).radio


def test_demo_intruder_is_confirmed_and_the_train_is_left_alone():
    from earshot.eval import metrics

    sc = Scenario.load(ROOT / "scenarios/demo.yaml", seed=100)
    r = run(sc, root=ROOT)
    m = metrics(r, "earshot", sc.duration)
    assert m.t_confirm is not None and m.t_confirm < 200
    assert m.false_verify <= 30  # one seed (24 now); `earshot eval` tracks the distribution:
    #                               seeds 100-111 median 16, range 1-29 (9-23 before 2026-10-20)
    near_rail = [e for e in r.events if e.get("task") == "verify" and e["kind"] == "dispatch"
                 and e["y"] > 700]
    assert near_rail == [], "verified something on the rail: the train should be Explained"


def test_array_drones_never_leave_their_roof():
    r = run(Scenario.load(ROOT / "scenarios/room.yaml", seed=100), root=ROOT)
    assert r.battery[0] > 0.97 and r.battery[1] > 0.97  # perched all run: hardly any drain
    assert not [e for e in r.events if e.get("drone") in (0, 1) and e["kind"] == "claim"]
