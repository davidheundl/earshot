"""The sensor physics must reproduce the concept's range tables (sections 5.1 and 5.2).

These ranges are the pitch's honesty claim ("we only hear small drones from close by"), so a
parameter change that drifts away from them should fail loudly.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from earshot.city import City
from earshot.messages import Mode, Sensor, TargetClass
from earshot.scenario import Environment, Scenario, Source
from earshot.sensors import SensorSuite, rf
from earshot.sensors import acoustics as ac
from earshot.state import DroneState

ROOT = Path(__file__).resolve().parents[1]
NIGHT, DAY = 40.0, 55.0
EGO_FILTERED = ac.ego_noise(ac.EGO_RPM_REF, True, 8.0)


def range_50(f, lo=0.5, hi=50_000.0):
    """Distance where the monotonically decreasing margin f(d) crosses zero (log bisection)."""
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return lo


def sound_range(lw, cls, noise_db):
    a = ac.ACOUSTICS[cls]
    return range_50(lambda d: ac.received_level(lw, cls, d, 0) - noise_db - a.snr50_db)


P = ac.SOUND_POWER
D, V, T = TargetClass.DRONE, TargetClass.VEHICLE, TargetClass.TRAIN
FLY = ac.db_sum(NIGHT, EGO_FILTERED)



def point(v):  # the concept gives "~v": allow +-35 %
    return (0.65 * v, 1.35 * v)


def span(lo, hi):  # the concept gives "lo-hi": allow 20 % beyond either end
    return (0.8 * lo, 1.2 * hi)


# (label, Lw, class, noise, expected (lo, hi) metres)
TABLE_5_1 = [
    ("small drone, perched night", P["small_drone"], D, NIGHT, point(90)),
    ("small drone, perched day", P["small_drone"], D, DAY, point(16)),
    ("small drone, flying", P["small_drone"], D, FLY, span(1, 5)),
    ("large drone, perched night", P["large_drone"], D, NIGHT, point(180)),
    ("large drone, perched day", P["large_drone"], D, DAY, point(35)),
    ("large drone, flying", P["large_drone"], D, FLY, span(3, 11)),
    ("heavy vehicle, perched night", P["heavy_vehicle"], V, NIGHT, point(1300)),
    ("heavy vehicle, perched day", P["heavy_vehicle"], V, DAY, point(250)),
    ("heavy vehicle, flying", P["heavy_vehicle"], V, FLY, span(20, 80)),
    ("freight train, perched night", P["freight_train"], T, NIGHT, span(2000, 9000)),
    ("freight train, perched day", P["freight_train"], T, DAY, point(1300)),
    ("freight train, flying", P["freight_train"], T, FLY, span(100, 500)),
]


@pytest.mark.parametrize("label,lw,cls,noise,expected", TABLE_5_1, ids=[r[0] for r in TABLE_5_1])
def test_sound_ranges_match_section_5_1(label, lw, cls, noise, expected):
    r = sound_range(lw, cls, noise)
    assert expected[0] <= r <= expected[1], f"{label}: {r:.0f} m not in {expected}"


def test_flying_is_deaf_and_filter_helps():
    lw = P["small_drone"]
    unfiltered = ac.db_sum(NIGHT, ac.ego_noise(ac.EGO_RPM_REF, False, 8.0))
    assert sound_range(lw, D, NIGHT) > 15 * sound_range(lw, D, FLY)
    assert sound_range(lw, D, FLY) > sound_range(lw, D, unfiltered)


def test_hover_is_not_quieter_than_cruise():
    from earshot.backend.twin2d import Kinematics

    k = Kinematics()
    assert ac.ego_noise(k.hover_rpm, True, 8) >= ac.ego_noise(k.cruise_rpm, True, 8)


def test_buildings_hurt_drone_tones_more_than_rumble():
    d = 300.0
    drone_loss = ac.received_level(90, D, d, 0) - ac.received_level(90, D, d, 1)
    rumble_loss = ac.received_level(90, V, d, 0) - ac.received_level(90, V, d, 1)
    assert drone_loss > rumble_loss


# section 5.2, through buildings (city with shadowing)
TABLE_5_2 = [("2g4", point(400)), ("5g8", point(120)),
             ("915", span(2000, 10000))]


@pytest.mark.parametrize("band,expected", TABLE_5_2, ids=[b for b, _ in TABLE_5_2])
def test_rf_ranges_match_section_5_2(band, expected):
    b = rf.BANDS[band]
    r = range_50(lambda d: rf.mean_rx_power(b, d, False) - b.noise_floor_dbm - rf.SNR50_DB)
    assert expected[0] <= r <= expected[1], f"{band}: {r:.0f} m not in {expected}"


def test_rf_classifier_operating_points():
    b = rf.BANDS["2g4"]
    assert rf.detection_probability(b, b.noise_floor_dbm - 14) == pytest.approx(0.5)
    assert rf.detection_probability(b, b.noise_floor_dbm - 12) == pytest.approx(0.85, abs=0.01)


# -- SensorSuite sampling ----------------------------------------------------------------


def _suite(sources, env=None):
    sc = Scenario("t", 0, "data/cities/synthetic.json", 10.0, 0.1, [], sources,
                  environment=env or Environment())
    return SensorSuite(City.load(ROOT / sc.city), sc, np.random.default_rng(0))


def _drone(x, y, perched, rpm=0.0, mode=Mode.LISTEN):
    return DroneState(0, 0.0, x, y, 40.0, 0.0, mode, "p" if perched else None, 1.0, rpm, False)


def test_suite_one_window_per_second_and_truth_attached():
    src = Source("tr", TargetClass.TRAIN, 0, 100, [(0, 900)], P["freight_train"])
    suite = _suite([src])
    dets = [d for k in range(50) for d in suite.sense(_drone(0, 880, True), k * 0.1)]
    assert 4 <= len(dets) <= 5  # 5 s of 1 s windows, near-certain detection at 20 m
    assert all(d.truth == "tr" and d.reading.sensor == Sensor.SOUND for d in dets)


def test_suite_nuisance_is_sometimes_heard_as_drone():
    fan = Source("fan", TargetClass.NOTHING, 0, 1000, [(0, 0)], 80.0)
    suite = _suite([fan])
    preds = [int(np.argmax(d.reading.class_probs)) for k in range(400)
             for d in suite.sense(_drone(5, 5, True), float(k))]
    assert preds.count(int(TargetClass.DRONE)) > 0.2 * len(preds)
