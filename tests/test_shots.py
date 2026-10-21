"""A Shot (a rifle fired, the `shot` Event) is heard for kilometres with a sharp bearing and
is Evidence for the Threat Class where it was fired, in a family of its own: a tower's
engine and animal bearings must not average it away, and it must not dent the map
elsewhere. It is no Source, and it leaves every run without Shots as it was."""

import math
from pathlib import Path

import numpy as np
import pytest

from earshot.agent.belief import HEARING_RANGE, SHOT_HEARING_RANGE, Belief
from earshot.city import City
from earshot.messages import Header, Mode, Observation, Sensor, TargetClass, decode, encode
from earshot.scenario import Scenario, Source
from earshot.sensors import SensorSuite
from earshot.sensors import acoustics as ac
from earshot.state import DroneState

ROOT = Path(__file__).resolve().parents[1]
V = TargetClass.VEHICLE
SHOT_P = (0.0333, 0.9, 0.0333, 0.0334)


@pytest.fixture(scope="module")
def reserve():
    return City.load(ROOT / "data/cities/savanna.json")


def _bearing(frm, to):
    return math.degrees(math.atan2(to[0] - frm[0], to[1] - frm[1])) % 360.0


def _obs(origin, seq, t, at, target, sensor, probs=SHOT_P, sigma=2.0):
    return Observation(Header(origin, seq, 1), t, at[0], at[1], 2.0, _bearing(at, target),
                       sigma, probs, sensor, Mode.LISTEN, 60.0)


def _tower(city, x, y, t=0.0, i=0):
    """A drone perched on a 12 m lookout tower."""
    z = float(city.ground(x, y)) + 12.0
    return DroneState(i, t, x, y, z, 0.0, Mode.LISTEN, "p_tower", 1.0, 0.0, False)


def test_a_shot_is_heard_for_kilometres():
    def p(d_m, kopjes=0):
        return ac.detection_probability(ac.received_level(ac.SHOT_LEVEL_DB, ac.SHOT, d_m, kopjes)
                                        - 35.0, ac.SHOT)

    assert p(3000) > 0.95 and 0.4 < p(6000) < 0.7 and p(7000) < 0.25  # bush at night
    assert p(2000, kopjes=1) > 0.5 > p(3000, kopjes=1)
    engine = ac.received_level(105.0, V, 6000, 0) - 35.0
    assert ac.detection_probability(engine, V) < 0.01  # an engine is not heard at 6 km
    assert ac.bearing_sigma(ac.SHOT, True, 12.0) == 2.0 < ac.bearing_sigma(V, True, 12.0)


def test_the_shot_event_fires_from_its_source():
    poacher = Source("p", V, 100, 200, [(0.0, 0.0), (1000.0, 0.0)], 105.0)
    sc = Scenario("t", 1, "data/cities/savanna.json", 300.0, 0.1, [], [poacher], threat=V)
    assert sc.apply({"op": "shot", "source": "p", "t": 150}, 150.0) == "shot by p at 500, 0"
    assert sc.apply({"op": "shot", "x": 10, "y": -20}, 160.0) == "shot at 10, -20"
    first, second = sc.shots
    assert (first.t, first.x, first.y, first.by) == (150.0, 500.0, 0.0, "p")
    assert first.level_db == ac.SHOT_LEVEL_DB and second.by == ""
    assert sc.sources == [poacher]  # nothing to find, see or follow
    for bad in ({"source": "nobody"}, {"source": "p", "_t": 50.0}, {"source": 3},
                {"x": "a", "y": 0}, {"x": 0, "y": 0, "level_db": float("nan")}):
        t = bad.pop("_t", 150.0)
        with pytest.raises((ValueError, KeyError)):
            sc.apply({"op": "shot", **bad}, t)
    assert len(sc.shots) == 2  # a bad op changes nothing


def test_a_source_stops_where_its_path_says():
    """A poacher stops at its kill: [x, y, s] on a path stands s seconds there and drives the
    rest at the same speed as without the stop."""
    drive = Source("d", V, 0, 30, [(0.0, 0.0), (100.0, 0.0), (300.0, 0.0)], 105.0)
    stop = Source("s", V, 0, 40, [(0.0, 0.0), (100.0, 0.0, 10), (300.0, 0.0)], 105.0)
    assert stop.waits == (0.0, 10.0, 0.0) and stop.path[1] == (100.0, 0.0)
    assert drive.waits == () and stop.position(5.0) == drive.position(5.0) == (50.0, 0.0)
    assert stop.position(10.0) == stop.position(19.9) == stop.pose(15.0)[:2] == (100.0, 0.0)
    assert stop.position(25.0) == pytest.approx(drive.position(15.0))
    assert stop.position(40.0) == (300.0, 0.0) and stop.position(40.1) is None


def test_each_tower_hears_a_shot_once_when_its_sound_arrives(reserve):
    sc = Scenario("t", 1, "data/cities/savanna.json", 300.0, 0.1, [], [], threat=V)
    suite = SensorSuite(reserve, sc, np.random.default_rng(0))
    towers = {13: (1807.0, 1668.0), 14: (3848.0, 1385.0)}  # 0.8 and 1.2 km from the shot
    sc.apply({"op": "shot", "x": 2624, "y": 1562}, 0.0)
    heard = {i: [] for i in towers}
    for k in range(1, 30):
        for i, at in towers.items():
            heard[i] += [(k, d) for d in suite.sense(_tower(reserve, *at, float(k), i), float(k))]
    for i, at in towers.items():
        (k, d), = heard[i]  # exactly once
        dist = math.hypot(2624 - at[0], 1562 - at[1])
        assert dist / ac.SPEED_OF_SOUND <= k < dist / ac.SPEED_OF_SOUND + 1.5
        r = d.reading
        assert r.sensor == Sensor.SHOT and int(np.argmax(r.class_probs)) == V
        assert abs((r.bearing - _bearing(at, (2624, 1562)) + 180) % 360 - 180) < 8.0


def test_shots_leave_every_other_reading_as_it_was(reserve):
    engine = Source("e", V, 0, 100, [(2624.0, 1562.0)], 105.0)

    def readings(shot: bool):
        sc = Scenario("t", 3, "data/cities/savanna.json", 100.0, 0.1, [], [engine], threat=V)
        if shot:
            sc.apply({"op": "shot", "x": 2624, "y": 1562}, 0.0)
        suite = SensorSuite(reserve, sc, np.random.default_rng(3))
        return [d.reading for k in range(40)
                for d in suite.sense(_tower(reserve, 1807.0, 1668.0, float(k)), float(k))]

    with_shot, without = readings(True), readings(False)
    assert [r for r in with_shot if r.sensor != Sensor.SHOT] == without
    assert sum(r.sensor == Sensor.SHOT for r in with_shot) == 1


def test_a_shot_is_not_averaged_away_by_the_towers_engine_bearings(reserve):
    """Two towers 0.8 and 1.2 km from where the shot fell have heard the elephants for 30 s.
    As SOUND, their one shot bearing each counts as one reading in thirty; as SHOT, in full."""
    target, towers = (2624.0, 1562.0), [(1807.0, 1668.0), (3848.0, 1385.0)]
    herd = (2767.0, 1526.0)

    def lift(sensor):
        b = Belief(reserve, cell=50.0, threat=V)
        seq = 0
        for k in range(30):
            for o, at in enumerate(towers):
                seq += 1
                b.add_evidence(_obs(o, seq, k, at, herd, Sensor.SOUND,
                                    probs=(0.05, 0.45, 0.0, 0.5), sigma=20.0), k)
        before = b._logits()[1][b._cell(*target)]
        for o, at in enumerate(towers):
            seq += 1
            b.add_evidence(_obs(o, seq, 30.0, at, target, sensor), 30.0)
        return b._logits()[1][b._cell(*target)] - before, b

    as_sound, _ = lift(Sensor.SOUND)
    as_shot, b = lift(Sensor.SHOT)
    assert as_sound < 1.0 and as_shot > 4.0
    assert {fam for _, fam in b._G} == {"sound", "shot"}


def test_a_shot_reaches_far_and_lowers_nothing(reserve):
    b = Belief(reserve, cell=50.0, threat=V)
    tower, target = (-307.0, 1822.0), (2624.0, 1562.0)  # tower 12, 2.9 km
    assert SHOT_HEARING_RANGE[True] > 6000.0 > HEARING_RANGE[(V, True)]
    b.add_evidence(_obs(12, 1, 0.0, tower, target, Sensor.SHOT), 0.0)
    g = b._G[(12, "shot")]
    assert g.min() >= 0.0 and g[1][b._cell(*target)] > 2.5
    far = (tower[0] + 5500.0 * (target[0] - tower[0]) / 2942.0,
           tower[1] + 5500.0 * (target[1] - tower[1]) / 2942.0)
    assert g[1][b._cell(*far)] > 2.5  # the vehicle wedge would stop at 3 km


def test_the_radio_frame_carries_a_shot():
    o = _obs(13, 7, 561.5, (1807.0, 1668.0), (2624.0, 1562.0), Sensor.SHOT)
    frame = encode(o, b"k" * 16)
    assert len(frame) <= 40
    back = decode(frame, b"k" * 16)
    assert back.sensor == Sensor.SHOT and back.bearing_sigma == 2.0
