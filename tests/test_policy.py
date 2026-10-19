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


def test_policy_params_override_the_hand_tuned_thresholds():
    from earshot.state import PolicyParams

    assert PolicyParams.from_dict({}) == PolicyParams()
    with pytest.raises(ValueError):
        PolicyParams.from_dict({"no_such_knob": 1})
    gate = PolicyParams.from_dict({"verify_min_strength": 100.0})  # above any Suspicion
    swarm = SwarmConfig.from_specs([DroneSpec(0, -300.0, -300.0, home=(-300.0, -300.0))],
                                   params=gate)
    p = Policy(0, CITY, swarm)
    claims = []
    for k in range(60):
        t = k * 0.1
        claims += p.decide(t, _perched(p, t), [SUSPICION], Belief(CITY), {}).claims
    assert claims == []  # the default gate (4) lets this strength-9 Suspicion through


def _suspicion(x, y, strength=9.0):
    return TrackEst(cell_id(x, y, CITY.bounds), x, y, 120.0, 60.0, 0.0, TargetClass.DRONE, 0.9,
                    TrackStatus.SUSPECTED, origins=2, strength=strength)


def _one(x=-300.0, y=-300.0, me=0, **params):
    from earshot.state import PolicyParams

    specs = [DroneSpec(0, -300.0, -300.0, home=(-300.0, -300.0)),
             DroneSpec(1, 600.0, 600.0, home=(600.0, 600.0))]
    swarm = SwarmConfig.from_specs(specs, params=PolicyParams.from_dict(params))
    return Policy(me, CITY, swarm)


def _claim(origin, t, tid, bid, lease, x, y, seq=1):
    return Claim(Header(origin, seq, 3), t, tid, bid, lease, x, y)


def test_the_first_claim_is_sent_again_halfway_through_settle():
    p = _one()
    sent = []
    for k in range(80):
        t = round(k * 0.1, 1)
        sent += [(t, c[2]) for c in p.decide(t, _perched(p, t), [SUSPICION], Belief(CITY),
                                             {}).claims]
    # claim, again at settle / 2 (a lost first frame cannot let a rival settle too), renewal
    assert sent == [(0.0, 10.0), (1.5, 10.0), (5.5, 10.0)]


def test_an_outbid_drone_releases_its_claim_and_waits_before_bidding_nearby():
    p = _one(me=1)  # the far drone
    d = p.decide(0.0, _perched(p, 0.0), [SUSPICION], Belief(CITY), {})
    assert [c[2] for c in d.claims] == [10.0]
    tid = d.claims[0][0]
    ghost = _suspicion(SUSPICION.x + 150.0, SUSPICION.y)  # next door, outside the rival's Claim
    ghost_tid = task_id(VERIFY, ghost.track_id)
    # a nearer drone claims the same danger, from a point 200 m west of this drone's Suspicion
    rival = {(0, tid): _claim(0, 0.5, tid, 640.0, 10.0, SUSPICION.x - 200.0, SUSPICION.y)}
    d = p.decide(1.0, _perched(p, 1.0), [SUSPICION, ghost], Belief(CITY), rival)
    assert any(e["kind"] == "outbid" for e in d.events)
    assert [(c[0], c[2]) for c in d.claims] == [(tid, 0.0)]  # released at once, not in 10 s
    d = p.decide(3.0, _perched(p, 3.0), [ghost], Belief(CITY), rival)
    assert d.claims == []  # cooldown: the Ghost next door is the same danger
    d = p.decide(6.5, _perched(p, 6.5), [ghost], Belief(CITY), rival)
    assert [(c[0], c[2]) for c in d.claims] == [(ghost_tid, 10.0)]


def test_a_task_drifts_at_most_same_task_radius_from_its_claim():
    p = _one()
    events = []
    x = SUSPICION.x
    for k in range(60):
        t = k * 0.5
        d = p.decide(t, _perched(p, t), [_suspicion(x, SUSPICION.y)], Belief(CITY), {})
        events += [(e["kind"], round(x)) for e in d.events]
        if any(kind == "drifted" for kind, _ in events):
            break
        if k >= 10:
            x += 30.0  # after dispatch it walks east, from one Ghost to the next
    kinds = [kind for kind, _ in events]
    assert kinds[:2] == ["claim", "dispatch"]
    assert "drifted" in kinds
    drift_x = next(x for kind, x in events if kind == "drifted")
    assert SUSPICION.x + 200.0 < drift_x <= SUSPICION.x + 230.0
    assert d.claims[0][2] == 0.0  # released, so it can be auctioned afresh
