"""The Threat Class is per Scenario: drones at the airport, vehicles (poachers) in a Reserve.

The default (drone) must leave every existing run unchanged; these tests pin what changes
when a Scenario says `threat: vehicle`: priors, RF from handheld radios, camera confirmation,
tasks, Follow after confirming, and the ground-truth flags the metrics count.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from earshot.agent.belief import (
    PRIOR_THREAT,
    PRIOR_VEHICLE_ELSEWHERE,
    PRIOR_VEHICLE_ROAD,
    Belief,
    TrackEst,
    cell_id,
)
from earshot.agent.policy import TASK_STRIDE, VERIFY, Policy, decode_task, task_id, task_kind
from earshot.city import City
from earshot.messages import (
    Header,
    Mode,
    Observation,
    Sensor,
    SensorReading,
    TargetClass,
    TrackStatus,
    decode,
    encode,
)
from earshot.scenario import Scenario, Source
from earshot.sensors import SensorSuite, rf
from earshot.state import DroneSpec, DroneState, PolicyParams, SwarmConfig

ROOT = Path(__file__).resolve().parents[1]
CITY = City.load(ROOT / "data/cities/synthetic.json")  # one road at y = -40, rail in the north
V = TargetClass.VEHICLE
VEHICLE_P = (0.03, 0.85, 0.07, 0.05)
OFF_ROAD = (-340.0, -560.0)
WEST, SOUTH, EAST = (-560.0, -560.0), (-390.0, -730.0), (-120.0, -600.0)


def _bearing(frm, to):
    return math.degrees(math.atan2(to[0] - frm[0], to[1] - frm[1])) % 360.0


def _obs(origin, seq, t, at, target, probs=VEHICLE_P, sigma=15.0, sensor=Sensor.SOUND,
         level=50.0, mode=Mode.LISTEN):
    b = _bearing(at, target) if target is not None else float("nan")
    return Observation(Header(origin, seq, 1), t, at[0], at[1], mode,
                       (SensorReading(sensor, b, sigma, probs, level),))


def _cell(b: Belief, x, y):
    return b._cell(x, y)


def test_the_threat_class_gets_the_threat_prior_off_the_road():
    drone, vehicle = Belief(CITY), Belief(CITY, threat=V)  # the cache must tell them apart
    off, on = _cell(drone, *OFF_ROAD), _cell(drone, 0.0, -40.0)
    assert drone.prior[1][off] == PRIOR_VEHICLE_ELSEWHERE
    assert vehicle.prior[1][off] == PRIOR_THREAT
    assert vehicle.prior[1][on] == drone.prior[1][on] == PRIOR_VEHICLE_ROAD
    assert Belief(CITY).prior[1][off] == PRIOR_VEHICLE_ELSEWHERE  # and back
    with pytest.raises(ValueError):
        Belief(CITY, threat=TargetClass.NOTHING)


def test_crossing_engine_bearings_off_the_road_are_a_vehicle_suspicion():
    def listen(b):
        seq = 0
        for k in range(25):
            for o, at in enumerate((WEST, SOUTH, EAST)):
                seq += 1
                b.add_evidence(_obs(o, seq, k, at, OFF_ROAD), k)
        return [tr for tr in b.tracks(25) if tr.cls == V]

    hunted = listen(Belief(CITY, threat=V))
    assert hunted and hunted[0].status == TrackStatus.SUSPECTED
    assert math.hypot(hunted[0].x - OFF_ROAD[0], hunted[0].y - OFF_ROAD[1]) < 80
    ignored = listen(Belief(CITY))  # at the airport a truck off the road is rare: weaker
    assert not ignored or ignored[0].strength < hunted[0].strength - 2.0


def test_a_handheld_radio_raises_the_vehicle_ring_not_the_drone_one():
    band = rf.BANDS["uhf"]
    assert band.emitter == V and band.sensor == Sensor.RF_UHF
    b = Belief(CITY, threat=V)
    at = (-560.0, -560.0)
    level = rf.mean_rx_power(band, 300.0, True)
    b.add_evidence(_obs(5, 1, 0.0, at, None, probs=(0.0333, 0.9, 0.0333, 0.0334),
                        sensor=Sensor.RF_UHF, level=level, mode=Mode.PATROL), 0.0)
    g = b._G[(5, "rf")]
    ring = _cell(b, at[0] + 300.0, at[1])
    assert g[1][ring] > 1.5 and g[0][ring] == 0.0


def test_the_radio_frame_carries_the_uhf_band():
    o = _obs(3, 9, 12.0, (10.0, 20.0), None, sensor=Sensor.RF_UHF, level=-80.0, mode=Mode.PATROL)
    frame = encode(o, b"k" * 16)
    assert len(frame) <= 40
    assert decode(frame, b"k" * 16).readings[0].sensor == Sensor.RF_UHF


def test_a_camera_sighting_of_the_threat_class_confirms_it():
    for threat, confirmed in ((V, True), (TargetClass.DRONE, False)):
        b = Belief(CITY, threat=threat)
        b.add_evidence(_obs(2, 1, 1.0, (OFF_ROAD[0], OFF_ROAD[1] - 40.0), OFF_ROAD,
                            probs=(0.0167, 0.95, 0.0167, 0.0167), sigma=2.0,
                            sensor=Sensor.CAMERA, mode=Mode.VERIFY), 1.0)
        assert b.confirmed_near(*OFF_ROAD, 1.0) == confirmed


def test_task_ids_cover_a_10_km_reserve():
    reserve = (-5000.0, -5000.0, 5000.0, 5000.0)  # 100 x 100 cells of 100 m
    cid = cell_id(4950.0, 4950.0, reserve)
    assert cid == 9999 < TASK_STRIDE
    tid = task_id(VERIFY, cid)
    assert task_kind(tid) == VERIFY and tid < 2**16
    assert decode_task(tid, reserve) == ("verify", 4950.0, 4950.0)


# -- policy -----------------------------------------------------------------------------------

def _track(cls, x=-150.0, y=-150.0, status=TrackStatus.SUSPECTED, strength=9.0):
    return TrackEst(cell_id(x, y, CITY.bounds), x, y, 120.0, 60.0, 0.0, cls, 0.9, status,
                    origins=2, strength=strength)


def _policy(threat=V, **params):
    swarm = SwarmConfig.from_specs([DroneSpec(0, -300.0, -300.0, home=(-300.0, -300.0))],
                                   params=PolicyParams.from_dict(params), threat=threat)
    return Policy(0, CITY, swarm)


def _perched(p: Policy, t: float) -> DroneState:
    roof = p._home_site
    return DroneState(0, t, roof.x, roof.y, roof.z, 0.0, Mode.LISTEN, roof.id, 1.0, 0.0, False)


def _near(x, y, t) -> DroneState:
    return DroneState(0, t, x, y - 30.0, 40.0, 0.0, Mode.VERIFY, None, 1.0, 5700.0, True)


def test_only_suspicions_of_the_threat_class_are_verified():
    for threat, cls, flies in ((V, V, True), (V, TargetClass.DRONE, False),
                               (TargetClass.DRONE, V, False)):
        p = _policy(threat)
        kinds = [e["kind"] for k in range(50) for e in p.decide(
            k * 0.1, _perched(p, k * 0.1), [_track(cls)], Belief(CITY), {}).events]
        assert ("dispatch" in kinds) == flies, (threat, cls)


def _follow_run(p, track_at, steps):
    """Suspected until t = 5, then Confirmed and driving east at 10 m/s while `track_at(t)`
    says where it is (None: gone)."""
    out = []
    for k in range(steps):
        t = round(k * 0.1, 1)
        pos = track_at(t)
        tracks = [] if pos is None else [_track(V, *pos, TrackStatus.CONFIRMED if t >= 5.0
                                                else TrackStatus.SUSPECTED)]
        s = _perched(p, t) if t < 3.5 else _near(*(pos or (0.0, 0.0)), t)
        d = p.decide(t, s, tracks, Belief(CITY), {})
        out += [(t, e["kind"], e) for e in d.events]
        out += [(t, "command", d.command)]
    return out


def test_a_verifier_follows_what_it_confirmed_and_hands_off_after_follow_for():
    p = _policy(follow_for=20.0, same_task_radius=400.0)
    drive = lambda t: (-150.0 + max(0.0, t - 5.0) * 10.0, -150.0)
    out = _follow_run(p, drive, 300)
    kinds = [k for _, k, _ in out if k != "command"]
    assert kinds[:3] == ["claim", "dispatch", "confirmed"]
    following = [(t, e) for t, k, e in out if k == "following"]
    assert len(following) >= 15 and all(e["cls"] == "VEHICLE" for _, e in following)
    xs = [e["x"] for _, e in following]
    assert xs == sorted(xs) and xs[-1] > xs[0] + 100.0  # it moves with the vehicle
    t_end = next(t for t, k, _ in out if k == "handed_off")
    assert 24.5 <= t_end <= 25.5
    cmds = [c for t, k, c in out if k == "command" and 6.0 <= t < t_end]
    assert all(c.mode == Mode.VERIFY and c.camera_on for c in cmds)
    assert cmds[-1].look_at[0] > cmds[0].look_at[0] + 100.0


def test_follow_ends_when_the_confirmed_track_is_gone():
    p = _policy(follow_for=600.0, dismiss_after=8.0)
    out = _follow_run(p, lambda t: None if t > 12.0 else (-150.0, -150.0), 300)
    lost = [t for t, k, _ in out if k == "lost"]
    assert lost and 20.0 <= lost[0] <= 21.0


def test_without_follow_the_verifier_releases_on_confirmation():
    p = _policy()
    out = _follow_run(p, lambda t: (-150.0, -150.0), 100)
    kinds = [k for _, k, _ in out if k != "command"]
    assert "confirmed" in kinds and "following" not in kinds
    confirmed = next(e for _, k, e in out if k == "confirmed")
    assert "cls" not in confirmed  # the event the airport logs, unchanged


# -- scenario and sensors -----------------------------------------------------------------------

def _scenario(tmp_path, body: str) -> Scenario:
    f = tmp_path / "s.yaml"
    f.write_text("name: t\nseed: 1\ncity: data/cities/synthetic.json\nduration: 10\n"
                 "drones: [{id: 0, x: 0, y: 0}]\n" + body)
    sc = Scenario.load(f)
    return sc


def test_scenario_threat_intruder_kind_and_confusion(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    src = "start: 0, end: 9, level_db: 100, path: [[0, 0]]"
    sc = _scenario(tmp_path, "threat: vehicle\nsources:\n"
                   f"  - {{id: p, cls: vehicle, {src}, kind: poacher pickup}}\n"
                   f"  - {{id: r, cls: vehicle, {src}, intruder: false}}\n"
                   f"  - {{id: e, cls: nuisance, {src}, kind: elephant}}\n"
                   "confusion: {nuisance: {drone: 0.05, vehicle: 0.3, train: 0, nothing: 0.65}}\n")
    assert sc.threat == V
    assert [(s.id, s.intruder, s.kind) for s in sc.sources] == [
        ("p", True, "poacher pickup"), ("r", False, ""), ("e", False, "elephant")]
    assert sc.confusion[TargetClass.NOTHING][V] == 0.3
    sc.apply({"op": "add_source", "cls": "vehicle", "x": 5, "y": 5}, 1.0)
    assert sc.sources[-1].intruder  # live vehicles are intruders where vehicles are hunted
    airport = _scenario(tmp_path, "sources: []\n")
    assert airport.threat == TargetClass.DRONE and airport.confusion == {}
    with pytest.raises(ValueError):
        _scenario(tmp_path, "confusion: {nuisance: {vehicle: 1.0}}\n")  # a whole row
    with pytest.raises(ValueError):
        _scenario(tmp_path, "threat: nothing\n")


def _suite(sources, **kw):
    sc = Scenario("t", 1, "data/cities/synthetic.json", 100.0, 0.1, [], sources, **kw)
    return SensorSuite(CITY, sc, np.random.default_rng(0))


def test_handhelds_transmit_in_bursts_and_read_as_vehicles():
    src = Source("p", V, 0, 1000, [(0.0, -500.0)], 100.0, ("uhf",), True, "poacher pickup")
    suite = _suite([src])
    me = DroneState(0, 0.0, 0.0, -1500.0, 40.0, 0.0, Mode.LISTEN, "p", 1.0, 0.0, False)
    rf_hits = [d for k in range(400) for d in suite.sense(me, float(k))
               if d.reading.sensor == Sensor.RF_UHF]
    assert 0.3 * 400 < len(rf_hits) < 0.5 * 400  # duty 0.4: push-to-talk
    assert all(int(np.argmax(d.reading.class_probs)) == V for d in rf_hits)


def test_the_camera_names_what_it_saw_and_animals_are_no_threat():
    poacher = Source("p", V, 0, 100, [(0.0, 50.0)], 100.0, (), True, "poacher pickup")
    rhino = Source("rh", TargetClass.NOTHING, 0, 100, [(30.0, 0.0)], 60.0, (), False, "rhino")
    me = DroneState(0, 0.0, 0.0, 0.0, 40.0, 0.0, Mode.VERIFY, None, 1.0, 5700.0, True)

    def camera(sources):
        suite = _suite(sources)
        return [d for k in range(40) for d in suite.sense(me, float(k))
                if d.reading.sensor == Sensor.CAMERA]

    both = camera([poacher, rhino])
    hits = [d for d in both if not math.isnan(d.reading.bearing)]
    assert len(hits) > 0.8 * len(both)  # the stand-in sees 90 %; a missed window: "rhino only"
    assert all(d.label == "poacher pickup" and d.truth == "p" for d in hits)
    alone = camera([rhino])
    assert alone and all(math.isnan(d.reading.bearing) for d in alone)  # "nothing here"
    assert sum(d.label == "rhino" for d in alone) > 0.8 * len(alone)


def test_sources_built_in_code_count_drones_as_intruders():
    # scenario_gen and eval build Sources directly: the airport's intruders are its drones
    assert Source("d", TargetClass.DRONE, 0, 1, [(0.0, 0.0)], 82.0).intruder
    assert not Source("t", V, 0, 1, [(0.0, 0.0)], 105.0).intruder
    from earshot.scenario_gen import generate

    sc = generate(Scenario.load(ROOT / "scenarios/demo.yaml"), 1000)
    assert any(s.intruder for s in sc.sources)
    assert all(s.intruder == (s.cls == TargetClass.DRONE) for s in sc.sources)


def test_the_reserves_own_vehicles_are_no_threat_to_the_camera():
    ranger = Source("r", V, 0, 100, [(0.0, 50.0)], 100.0, ("uhf",), False, "ranger Land Cruiser")
    me = DroneState(0, 0.0, 0.0, 0.0, 40.0, 0.0, Mode.VERIFY, None, 1.0, 5700.0, True)
    suite = _suite([ranger], threat=V)
    seen = [d for k in range(20) for d in suite.sense(me, float(k)) if d.reading.sensor == Sensor.CAMERA]
    assert seen and all(math.isnan(d.reading.bearing) for d in seen)
    assert sum(d.label == "ranger Land Cruiser" for d in seen) > 0.8 * len(seen)


def test_a_vehicle_heard_beside_its_road_is_explained_within_its_ellipse():
    b = Belief(CITY, threat=V)  # the road runs along y = -40
    assert b._status(V, 0.0, -190.0) == TrackStatus.SUSPECTED  # 150 m off, a tight fix
    assert b._status(V, 0.0, -190.0, reach=200.0) == TrackStatus.EXPLAINED  # road in its ellipse
    assert b._status(V, 0.0, -100.0) == TrackStatus.EXPLAINED  # within 60 m anyway


def test_a_ranged_sighting_confirms_the_vehicle_where_it_is():
    # a thermal camera geolocates the pickup 220 m ahead (level = range): the confirmed point
    # and the Confirmed Track sit on the vehicle, so a Follower flies to it, not to itself
    b = Belief(CITY, threat=V)
    at, car = (-600.0, -700.0), (-380.0, -700.0)
    seq = 0
    for k in range(1, 20):
        for o, frm in enumerate((WEST, SOUTH, EAST)):
            seq += 1
            b.add_evidence(_obs(o, seq, k, frm, car), k)
        b.add_evidence(_obs(9, 100 + k, k, at, car, probs=(0.0167, 0.95, 0.0167, 0.0167),
                            sigma=2.0, sensor=Sensor.CAMERA, mode=Mode.VERIFY, level=220.0), k)
    assert b.confirmed_near(*car, 19.0, radius=30.0)
    confirmed = [tr for tr in b.tracks(20) if tr.status == TrackStatus.CONFIRMED]
    assert confirmed and math.hypot(confirmed[0].x - car[0], confirmed[0].y - car[1]) < 60


def test_of_two_followers_the_farther_one_yields():
    from earshot.messages import Claim

    p = _policy(follow_for=600.0)
    out = _follow_run(p, lambda t: (-150.0, -150.0), 80)  # confirmed at 5 s, following
    assert any(k == "following" for _, k, _ in out)
    tid = p._task.tid
    near = {(4, tid): Claim(Header(4, 9, 3), 8.0, tid, 654.0, 10.0, -150.0, -150.0)}
    d = p.decide(8.1, _near(-150.0, -40.0, 8.1), [_track(V, status=TrackStatus.CONFIRMED)],
                 Belief(CITY), near)
    assert [e["kind"] for e in d.events] == ["outbid"] and p._task is None


def test_a_drone_does_not_bid_on_a_task_beyond_max_verify_distance():
    far = _track(V, x=900.0, y=900.0)  # ~1700 m from drone 0's roof
    for limit, flies in ((1000.0, False), (3000.0, True)):
        p = _policy(max_verify_distance=limit)
        kinds = [e["kind"] for k in range(50) for e in p.decide(
            k * 0.1, _perched(p, k * 0.1), [far], Belief(CITY), {}).events]
        assert ("claim" in kinds) == flies


def test_tracks_and_claims_are_relayed_only_near_their_place():
    from earshot.agent.store import MessageStore
    from earshot.messages import Claim, Track

    key = b"k" * 16
    track = encode(Track(Header(2, 1, 3), 1.0, 9, 3000.0, 0.0, 50.0, 20.0, 0.0, V, 0.9,
                         TrackStatus.SUSPECTED, 5.0), key)
    claim = encode(Claim(Header(2, 2, 3), 1.0, 9, 600.0, 10.0, 3000.0, 0.0), key)
    for radius, here, relays in ((4000.0, (0.0, 0.0), 2), (2000.0, (0.0, 0.0), 0),
                                 (2000.0, (2500.0, 0.0), 2), (math.inf, (-9000.0, 0.0), 2)):
        store = MessageStore(0, key, relay_radius=radius)
        store.receive(track, 0.0, 0.0, here)
        store.receive(claim, 0.0, 0.0, here)
        assert len(store.due(5.0)) == relays, (radius, here)

