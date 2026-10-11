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
