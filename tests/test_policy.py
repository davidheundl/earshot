from pathlib import Path

from earshot.agent.belief import Belief, TrackEst, cell_id
from earshot.agent.policy import (
    LISTEN_A,
    LISTEN_B,
    VERIFY,
    Policy,
    decode_task,
    task_id,
    task_kind,
)
from earshot.city import City
from earshot.messages import Claim, Header, Mode, TargetClass, TrackStatus
from earshot.state import DroneSpec, DroneState, SwarmConfig

CITY = City.load(Path(__file__).resolve().parents[1] / "data/cities/synthetic.json")

VALLEY = (-2000.0, -2000.0, 2000.0, 2000.0)  # 40 x 40 cells of 100 m


def test_task_ids_stay_distinct_on_a_4_km_city():
    north = cell_id(0.0, 1900.0, VALLEY)  # beyond the first 1024 cells
    assert north >= 1024
    ids = {task_id(kind, north) for kind in (VERIFY, LISTEN_A, LISTEN_B)}
    assert len(ids) == 3
    assert task_id(VERIFY, north) not in {task_id(LISTEN_A, c) for c in range(1600)}
    for kind in (VERIFY, LISTEN_A, LISTEN_B):
        tid = task_id(kind, north)
        assert task_kind(tid) == kind
        name, x, y = decode_task(tid, VALLEY)
        assert (name == "verify") == (kind == VERIFY)
        assert (x, y) == (50.0, 1950.0)


def test_listener_that_switches_to_verify_does_not_take_its_roof_back():
    swarm = SwarmConfig.from_specs([
        DroneSpec(0, -600.0, -600.0, home=(-600.0, -600.0), default=Mode.LISTEN),
        DroneSpec(1, 600.0, 600.0, home=(600.0, 600.0), default=Mode.PATROL),
    ])
    policy = Policy(0, CITY, swarm)
    roof = policy._home_site
    suspicion = TrackEst(cell_id(-150.0, -150.0, CITY.bounds), -150.0, -150.0, 120.0, 60.0, 0.0,
                         TargetClass.DRONE, 0.75, TrackStatus.SUSPECTED, origins=2, strength=9.0)
    claims, kinds = {}, []
    for k in range(30):
        t = k * 0.1
        s = DroneState(0, t, roof.x, roof.y, roof.z, 0.0, Mode.LISTEN, roof.id, 1.0, 0.0, False)
        d = policy.decide(t, s, [suspicion], Belief(CITY), claims)
        for seq, (tid, bid, lease, x, y) in enumerate(d.claims):
            claims[(0, tid)] = Claim(Header(0, 100 * k + seq, 3), t, tid, bid, lease, x, y)
        kinds += [e["kind"] for e in d.events]
    assert kinds.count("switch") <= 1
    assert policy._task is not None and policy._task.kind == VERIFY
