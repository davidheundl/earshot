import math
import random
from pathlib import Path

import numpy as np
import pytest

from earshot.agent.belief import (
    CONFIRM_SPEED,
    TAU,
    Belief,
    TrackEst,
    cell_id,
    merge_overlapping,
)
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


def test_evidence_of_a_silent_sensor_fades_with_tau():
    b = Belief(CITY)
    t = listen(b, {0: WEST, 1: SOUTH}, TARGET, seconds=60)  # saturated: W >> N_EFF
    iy, ix = b._cell(*TARGET)
    b.advance(t)
    z_now = b._logits()[0, iy, ix] - b.prior[0, iy, ix]
    b.advance(t + 30.0)  # both drones silent for one TAU
    z_later = b._logits()[0, iy, ix] - b.prior[0, iy, ix]
    assert z_later == pytest.approx(z_now * math.exp(-30.0 / TAU), rel=0.05)


def _zone(x, y, a=120.0, b=60.0, angle=0.0, strength=5.0, status=TrackStatus.SUSPECTED):
    return TrackEst(cell_id(x, y, CITY.bounds), x, y, a, b, angle, TargetClass.DRONE, 0.9, status,
                    origins=2, strength=strength)


def test_overlapping_zones_merge_at_their_likelihood_weighted_mean():
    a, b = _zone(0, 0, strength=9.0), _zone(160, 40, strength=9.0)  # equally likely
    ghost = _zone(-150, 0, strength=3.0)  # e^6 times less likely
    far = _zone(0, 600)  # a second danger: no overlap
    out = merge_overlapping([a, far, b, ghost], CITY.bounds)
    assert len(out) == 2
    merged = out[0]
    assert merged.x == pytest.approx(80.0, abs=1.0) and merged.y == pytest.approx(20.0, abs=0.5)
    assert merged.semi_major == a.semi_major and merged.strength == 9.0
    assert out[1] == far


def test_zones_that_only_touch_end_to_end_overlap_and_side_by_side_ones_do_not():
    # long axes along x: 120 m reach each way along x, 60 m along y
    assert len(merge_overlapping([_zone(0, 0), _zone(230, 0)], CITY.bounds)) == 1
    assert len(merge_overlapping([_zone(0, 0), _zone(0, 130)], CITY.bounds)) == 2


def test_a_suspicion_beside_a_confirmed_zone_stays_its_own():
    # it may be a second drone: the policy's confirmed_shadow decides, the merge does not hide it
    confirmed = _zone(100, 0, status=TrackStatus.CONFIRMED)
    out = merge_overlapping([_zone(0, 0), confirmed, _zone(150, 0, status=TrackStatus.CONFIRMED)],
                            CITY.bounds)
    assert [tr.status for tr in out] == [TrackStatus.SUSPECTED, TrackStatus.CONFIRMED]
    assert out[1].x == pytest.approx(125.0)


def test_a_confirmed_point_follows_one_peak_and_never_hops_on():
    b = Belief(CITY)
    b._confirmed = [(0.0, 0.0, 10.0)]
    near, lobe, next_one = _zone(100, 0), _zone(-20, 130), _zone(220, 0)
    out = b._follow_confirmed([near, lobe, next_one], 11.0)
    # the nearest peak is the drone; a peak beside it is its side lobe; one beyond is not its
    assert [tr.status for tr in out] == [TrackStatus.CONFIRMED, TrackStatus.EXPLAINED,
                                         TrackStatus.SUSPECTED]
    (x, y, t), = b._confirmed
    assert (x, y, t) == (pytest.approx(CONFIRM_SPEED), pytest.approx(0.0), 11.0)
