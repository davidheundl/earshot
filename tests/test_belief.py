import math
import random
from pathlib import Path

import numpy as np
import pytest

from earshot.agent.belief import (
    A_CAMERA_NONE,
    CONFIRM_HOLD,
    CONFIRM_SPEED,
    N_EFF,
    TAU,
    TRACK_SUPPRESS,
    Belief,
    TrackEst,
    cell_id,
    merge_overlapping,
)
from earshot.city import City
from earshot.messages import (
    Header,
    Mode,
    Observation,
    Sensor,
    SensorReading,
    TargetClass,
    Track,
    TrackStatus,
)

ROOT = Path(__file__).resolve().parents[1]
CITY = City.load(ROOT / "data/cities/synthetic.json")
DRONE_P = (0.85, 0.05, 0.05, 0.05)
RUMBLE_P = (0.1, 0.4, 0.45, 0.05)


def bearing(frm, to):
    return math.degrees(math.atan2(to[0] - frm[0], to[1] - frm[1])) % 360.0


def obs(origin, seq, t, at, target, probs=DRONE_P, sigma=5.0, sensor=Sensor.SOUND, level=40.0,
        mode=Mode.LISTEN):
    b = bearing(at, target) if target is not None else float("nan")
    return Observation(Header(origin, seq, 1), t, at[0], at[1], mode,
                       (SensorReading(sensor, b, sigma, probs, level),))


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
               mode=Mode.VERIFY, level=0.0)  # from the Suspicion itself; 0: full camera range
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
        b.add_evidence(Observation(stuck.header, stuck.t, stuck.x, stuck.y, Mode.LISTEN,
                                   (SensorReading(Sensor.SOUND, 0.0, 5.0, DRONE_P, 40.0),)), k)
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
    b._confirmed = [(0.0, 0.0, 10.0, 10.0)]
    near, lobe, next_one = _zone(100, 0), _zone(-20, 130), _zone(220, 0)
    out = b._follow_confirmed([near, lobe, next_one], 11.0)
    # the nearest peak is the drone; a peak beside it is its side lobe; one beyond is not its
    assert [tr.status for tr in out] == [TrackStatus.CONFIRMED, TrackStatus.EXPLAINED,
                                         TrackStatus.SUSPECTED]
    (x, y, seen, moved), = b._confirmed
    # it moved, but following a peak is no sighting: the hold still counts from t = 10
    assert (x, y, seen, moved) == (pytest.approx(CONFIRM_SPEED), pytest.approx(0.0), 10.0, 11.0)


# -- a second drone beside a strong one (review 2026-10-18, 2.2 #4) ------------------------

D1, D2 = (0.0, 0.0), (260.0, 0.0)


def _two_drones(listeners, sigma):
    """D1 heard by sound and two RF patrollers, D2 (weaker) by sound only; every listener
    within 330 m bears on each drone it hears."""
    from earshot.sensors import rf

    b = Belief(CITY)
    patrol = {9: (0.0, 300.0), 10: (-300.0, 0.0)}
    seq = 0
    for k in range(30):
        for o, at in listeners.items():
            for d in (D1, D2):
                if math.dist(at, d) <= 330.0:
                    seq += 1
                    b.add_evidence(obs(o, seq, k, at, d, sigma=sigma), k)
        for o, at in patrol.items():
            level = rf.mean_rx_power(rf.BANDS["2g4"], math.dist(at, D1), False)
            seq += 1
            b.add_evidence(obs(o, seq, k, at, None, sensor=Sensor.RF_2G4, level=level,
                               mode=Mode.PATROL), k)
    return [tr for tr in b.tracks(30) if tr.cls == TargetClass.DRONE and tr.strength > 2.0]


# Fix A of docs/review-2026-10-18.md (mask suppressed cells, merge only on direct overlap)
# makes both pass, but it turns RF Ghost side lobes 200-500 m from a confirmed drone into
# Suspicions of their own: paired eval, 20 random worlds, false verifies 17.5 -> 49.5 and
# flights 29.5 -> 58.5 (docs/functionality.md, Results). Kept as the spec for when a
# corroboration gate keeps Ghosts from being dispatched.
SECOND_DRONE = pytest.mark.xfail(strict=True, reason="fix A held back until Ghosts are gated")


@SECOND_DRONE
@pytest.mark.parametrize("sigma, listeners", [
    # the stronger peak's cells used to pull the second Track 61 m toward the first
    (6.0, {0: (-277, 285), 1: (-269, -271), 2: (-48, -230), 3: (-97, -95), 4: (333, -63),
           5: (-68, -5)}),
    # Ghost ellipses used to chain the second drone into the first and average it away
    (6.0, {0: (373, 12), 1: (93, 271), 2: (351, -34), 3: (-169, 87), 4: (-148, 8)}),
])
def test_a_weaker_second_drone_260_m_away_keeps_its_own_track(sigma, listeners):
    tracks = _two_drones(listeners, sigma)
    assert len(tracks) == 2
    assert min(math.dist((tr.x, tr.y), D2) for tr in tracks) < 50.0
    assert min(math.dist((tr.x, tr.y), D1) for tr in tracks) < 100.0


@SECOND_DRONE
def test_merging_is_not_transitive():
    # b overlaps the strong seed a and the far c, but c does not overlap a: c stays its own
    a, b, c = _zone(0, 0, strength=9.0), _zone(200, 0, strength=3.0), _zone(400, 0, strength=6.0)
    out = merge_overlapping([a, b, c], CITY.bounds)
    assert len(out) == 2
    assert out[1] == c


# -- Confirmed points and camera evidence ---------------------------------------------------


def _track(origin, t, x, y, status=TrackStatus.CONFIRMED, seq=1):
    return Track(Header(origin, seq, 3), t, cell_id(x, y, CITY.bounds), x, y, 60.0, 40.0, 0.0,
                 TargetClass.DRONE, 0.95, status, 9.0)


def _camera(origin, seq, t, at, target=None, level=0.0):
    probs = DRONE_P if target is not None else (0.0, 0.0, 0.0, 1.0)
    return obs(origin, seq, t, at, target, probs=probs, sensor=Sensor.CAMERA, mode=Mode.VERIFY,
               level=level)


def test_a_received_confirmed_track_never_adds_a_confirmed_point():
    b = Belief(CITY)
    b.set_track(_track(3, 4.0, 0.0, 0.0))
    assert b._confirmed == []  # only a camera adds one
    b._confirmed = [(100.0, 0.0, 5.0, 5.0)]
    b.set_track(_track(3, 6.0, 0.0, 0.0, seq=2))  # moves it, at most CONFIRM_SPEED, hold kept
    b.set_track(_track(4, 7.0, 500.0, 0.0))  # far from it: no second point
    (x, y, seen, moved), = b._confirmed
    assert (x, y, seen, moved) == (pytest.approx(100.0 - CONFIRM_SPEED), 0.0, 5.0, 6.0)


def test_camera_evidence_is_capped_per_camera():
    b = Belief(CITY)
    for k in range(20):  # a Verifier hovering 20 s, seeing nothing
        b.add_evidence(_camera(2, k, float(k), TARGET), float(k))
    iy, ix = b._cell(*TARGET)
    assert b._camera_evidence()[0, iy, ix] >= N_EFF * A_CAMERA_NONE - 1e-6  # not 20 x -8

    b = Belief(CITY)
    t = listen(b, {0: WEST, 1: SOUTH}, TARGET)
    near = (TARGET[0] - 20, TARGET[1] - 20)
    for k in range(20):  # ... or seeing the drone: probability must not saturate to a plateau
        b.add_evidence(_camera(2, 100 + k, t + k, near, TARGET), t + k)
    assert b.posterior()[:3].sum(axis=0).max() < 1.0
    tr = b.tracks(t + 20)[0]
    assert tr.status == TrackStatus.CONFIRMED
    assert math.dist((tr.x, tr.y), TARGET) < 40


def test_camera_nothing_drops_a_confirmed_point_it_no_longer_sees():
    b = Belief(CITY)
    near = (TARGET[0] - 20, TARGET[1] - 20)
    b.add_evidence(_camera(2, 1, 10.0, near, TARGET), 10.0)
    assert len(b._confirmed) == 1
    b.add_evidence(_camera(2, 2, 11.0, near), 11.0)  # one missed frame: still there
    assert len(b._confirmed) == 1
    b.add_evidence(_camera(2, 3, 12.0, (TARGET[0] + 300, TARGET[1])), 12.0)  # looked elsewhere
    assert len(b._confirmed) == 1
    b.add_evidence(_camera(2, 4, 12.0, near), 12.0)  # two seconds without it: gone
    assert b._confirmed == []


def test_a_confirmed_point_expires_without_a_new_sighting():
    b = Belief(CITY)
    b.add_evidence(_camera(2, 1, 0.0, (0.0, 0.0), (30.0, 0.0)), 0.0)
    for k in range(1, 130):  # a drone peak beside it every second, but no camera
        b._follow_confirmed([_zone(0, 0)], float(k))
        assert bool(b._confirmed) == (k < CONFIRM_HOLD)


def test_camera_nothing_clears_only_the_radius_its_detector_checked():
    b = Belief(CITY)
    at = (0.0, 0.0)
    b.add_evidence(_camera(2, 1, 0.0, (60.0, 0.0), (80.0, 0.0)), 0.0)  # confirmed 60 m away
    b.add_evidence(_camera(2, 2, 5.0, at, level=40.0), 5.0)  # "nothing within 40 m"
    assert len(b._confirmed) == 1
    cam = b._camera_evidence()[0]
    assert cam[b._cell(20.0, 0.0)] < 0  # cleared
    assert cam[b._cell(0.0, 80.0)] == 0  # beyond 40 m: not looked at
    b.add_evidence(_camera(2, 3, 6.0, at), 6.0)  # level 0: the full CAMERA_RANGE
    assert b._confirmed == []


# -- received Tracks -----------------------------------------------------------------------


def test_a_newer_track_from_the_same_origin_replaces_the_older_one():
    b = Belief(CITY)
    old = _track(3, 10.0, 85.0, 0.0, status=TrackStatus.SUSPECTED)
    new = _track(3, 11.0, 115.0, 0.0, status=TrackStatus.SUSPECTED, seq=2)  # next 100 m cell
    assert old.track_id != new.track_id
    other = _track(4, 10.0, 85.0, 0.0, status=TrackStatus.SUSPECTED)
    for tr in (old, new, other):
        b.set_track(tr)
    assert b.received_tracks(11.0) == [new, other]
    b.set_track(old)  # relayed late: the newer one stands
    assert b.received_tracks(11.0) == [new, other]
    assert b.received_tracks(25.5) == [new]  # the older origin's Track aged out
    assert list(b._tracks_rx) == [3]


@pytest.mark.parametrize("cell", [20.0, 25.0, 33.3, 50.0])
def test_a_peaks_window_holds_every_cell_it_can_suppress(cell):
    b = Belief(CITY, cell=cell)
    for iy, ix in [(0, 0), (b.shape[0] // 2, b.shape[1] // 3), (b.shape[0] - 1, b.shape[1] - 1)]:
        inside = np.zeros(b.shape, dtype=bool)
        inside[b._window(iy, ix)] = True
        near = np.hypot(b.X - b.X[iy, ix], b.Y - b.Y[iy, ix]) <= TRACK_SUPPRESS + cell
        assert inside[near].all()
