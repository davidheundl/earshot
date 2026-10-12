from pathlib import Path

import pytest

from earshot.agent.belief import Belief, TrackEst, cell_id
from earshot.agent.policy import VERIFY, Policy, decode_task, task_id, task_kind
from earshot.city import City
from earshot.messages import Claim, Header, Mode, TargetClass, TrackStatus
from earshot.state import DroneSpec, DroneState, SwarmConfig

CITY = City.load(Path(__file__).resolve().parents[1] / "data/cities/synthetic.json")

VALLEY = (-2000.0, -2000.0, 2000.0, 2000.0)  # 40 x 40 cells of 100 m


def test_task_ids_decode_on_a_4_km_city():
    north = cell_id(0.0, 1900.0, VALLEY)  # beyond the first 1024 cells
    assert north >= 1024
    tid = task_id(VERIFY, north)
    assert task_kind(tid) == VERIFY
    assert decode_task(tid, VALLEY) == ("verify", 50.0, 1950.0)


SUSPICION = TrackEst(cell_id(-150.0, -150.0, CITY.bounds), -150.0, -150.0, 120.0, 60.0, 0.0,
                     TargetClass.DRONE, 0.9, TrackStatus.SUSPECTED, origins=2, strength=9.0)


def _perched(policy: Policy, t: float) -> DroneState:
    roof = policy._home_site
    return DroneState(policy.me, t, roof.x, roof.y, roof.z, 0.0, Mode.LISTEN, roof.id, 1.0,
                      0.0, False)


@pytest.mark.parametrize("far_first", [False, True])
def test_only_the_nearest_drone_verifies_and_the_others_hold(far_first):
    swarm = SwarmConfig.from_specs([
        DroneSpec(0, -300.0, -300.0, home=(-300.0, -300.0), default=Mode.LISTEN),  # near
        DroneSpec(1, 600.0, 600.0, home=(600.0, 600.0), default=Mode.LISTEN),  # far
        DroneSpec(2, -600.0, 600.0, home=(-600.0, 600.0), default=Mode.LISTEN),  # far
    ])
    policies = [Policy(i, CITY, swarm) for i in (0, 1, 2)]
    if far_first:  # the far drones bid first; the near one still takes it over
        policies = policies[1:] + policies[:1]
    claims, last = {}, {}
    for k in range(60):
        t = k * 0.1
        for p in policies:
            d = p.decide(t, _perched(p, t), [SUSPICION], Belief(CITY), claims)
            for seq, (tid, bid, lease, x, y) in enumerate(d.claims):
                claims[(p.me, tid)] = Claim(Header(p.me, 100 * k + seq, 3), t, tid, bid, lease,
                                            x, y)
            last[p.me] = d.command
    assert last[0].mode == Mode.VERIFY
    for far in (1, 2):
        assert last[far].mode == Mode.LISTEN and last[far].perch_site is not None
