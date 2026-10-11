import math
import random
from pathlib import Path

import numpy as np

from earshot.agent.belief import Belief
from earshot.city import City
from earshot.messages import Header, Mode, Observation, Sensor, TargetClass, TrackStatus

ROOT = Path(__file__).resolve().parents[1]
CITY = City.load(ROOT / "data/cities/synthetic.json")
DRONE_P = (0.85, 0.05, 0.05, 0.05)
RUMBLE_P = (0.1, 0.4, 0.45, 0.05)


def bearing(frm, to):
    return math.degrees(math.atan2(to[0] - frm[0], to[1] - frm[1])) % 360.0


def obs(origin, seq, t, at, target, probs=DRONE_P, sigma=5.0, sensor=Sensor.SOUND, level=40.0,
        mode=Mode.LISTEN):
    b = bearing(at, target) if target is not None else float("nan")
    return Observation(Header(origin, seq, 1), t, at[0], at[1], 2.0, b, sigma, probs, sensor,
                       mode, level)


def listen(belief, origins_at, target, seconds=25, probs=DRONE_P, t0=0.0):
    seq = 0
    for k in range(seconds):
        for o, at in origins_at.items():
            seq += 1
            belief.add_evidence(obs(o, seq, t0 + k, at, target, probs), t0 + k)
    return t0 + seconds


TARGET = (-340.0, -560.0)
WEST, SOUTH = (-560.0, -560.0), (-390.0, -730.0)


def test_two_crossing_bearings_make_a_localised_drone_track():
    b = Belief(CITY)
    t = listen(b, {0: WEST, 1: SOUTH}, TARGET)
    tracks = b.tracks(t)
    assert tracks, "no track from two crossing bearings"
    tr = tracks[0]
    assert tr.cls == TargetClass.DRONE and tr.status == TrackStatus.SUSPECTED
    assert math.hypot(tr.x - TARGET[0], tr.y - TARGET[1]) < 40
    assert tr.semi_major < 60 and tr.origins == 2


def test_one_bearing_line_alone_is_not_a_track():
    b = Belief(CITY)
    t = listen(b, {0: WEST}, TARGET)
    assert b.tracks(t) == []


def test_rumble_at_the_rail_is_explained_as_train():
    b = Belief(CITY)
    rail_point = (-200.0, 805.0)  # on the rail line
    t = listen(b, {0: (-390.0, 630.0), 1: (-50.0, 630.0)}, rail_point, probs=RUMBLE_P)
    tracks = b.tracks(t)
    assert tracks and tracks[0].cls == TargetClass.TRAIN
    assert tracks[0].status == TrackStatus.EXPLAINED


def test_camera_confirms_and_camera_nothing_dismisses():
    b = Belief(CITY)
    t = listen(b, {0: WEST, 1: SOUTH}, TARGET)
    near = (TARGET[0] - 20, TARGET[1] - 20)
    b.add_evidence(obs(2, 1, t, near, TARGET, sensor=Sensor.CAMERA, mode=Mode.VERIFY), t)
    assert b.tracks(t)[0].status == TrackStatus.CONFIRMED

    b2 = Belief(CITY)
    t = listen(b2, {0: WEST, 1: SOUTH}, TARGET)
    none = obs(2, 1, t, near, None, probs=(0.0, 0.0, 0.0, 1.0), sensor=Sensor.CAMERA,
               mode=Mode.VERIFY)
    none = obs(2, 1, t, TARGET, None, probs=(0.0, 0.0, 0.0, 1.0), sensor=Sensor.CAMERA,
               mode=Mode.VERIFY)  # the Verifier looks from the Suspicion itself
    b2.add_evidence(none, t)
    assert all(math.hypot(tr.x - TARGET[0], tr.y - TARGET[1]) > 100 for tr in b2.tracks(t))


def test_rf_ring_and_one_bearing_localise():
    b = Belief(CITY)
    from earshot.sensors import rf

    patrol = (-340.0, -300.0)  # 260 m north of the target
    level = rf.mean_rx_power(rf.BANDS["2g4"], 260.0, False)
    seq = 0
    for k in range(25):
        seq += 1
        b.add_evidence(obs(0, seq, k, WEST, TARGET), k)
        seq += 1
        b.add_evidence(obs(3, seq, k, patrol, None, sensor=Sensor.RF_2G4, level=level,
                           mode=Mode.PATROL), k)
    tracks = [tr for tr in b.tracks(25) if tr.cls == TargetClass.DRONE]
    assert any(math.hypot(tr.x - TARGET[0], tr.y - TARGET[1]) < 120 for tr in tracks)


def test_arrival_order_does_not_matter():
    evidence = [obs(o, k, float(k), at, TARGET)
                for k in range(20) for o, at in ((0, WEST), (1, SOUTH))]
    a = Belief(CITY)
    for e in evidence:  # in time order, each folded in when it happens
        a.add_evidence(e, e.t)
    shuffled = evidence[:]
    random.Random(1).shuffle(shuffled)
    b = Belief(CITY)
    for e in shuffled:  # all late, in random order
        b.add_evidence(e, 19.0)
    a.advance(30.0)
    b.advance(30.0)
    assert np.allclose(a.posterior(), b.posterior(), atol=1e-9)


def test_stuck_bearing_is_flagged_as_faulty_sensor():
    b = Belief(CITY)
    honest = {0: WEST, 1: SOUTH, 3: (-340.0, -680.0), 4: (-340.0, -380.0), 5: (-220.0, -720.0)}
    seq = 0
    for k in range(40):
        for o, at in honest.items():
            seq += 1
            b.add_evidence(obs(o, seq, k, at, TARGET), k)
        seq += 1
        stuck = obs(2, seq, k, (-220.0, -560.0), TARGET)  # 120 m east, clear line of sight
        b.add_evidence(Observation(stuck.header, stuck.t, stuck.x, stuck.y, 2.0, 0.0, 5.0,
                                   DRONE_P, Sensor.SOUND, Mode.LISTEN, 40.0), k)
        b.tracks(k)
        b.check_faults(k)
    assert b.faulty == {2}
