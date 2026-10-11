import numpy as np
import pytest

from earshot.radio.medium import RadioConfig, RadioMedium, lora_airtime
from earshot.scenario import JamZone


def test_airtime_matches_concept_numbers():
    # concept 8.3: 40 B ~ 82 ms at SF7/125 kHz, ~21 ms at 500 kHz
    assert lora_airtime(40, 7, 125) == pytest.approx(0.082, abs=0.002)
    assert lora_airtime(40, 7, 500) == pytest.approx(0.0205, abs=0.001)
    assert lora_airtime(40, 12, 125) > 1.5


def _medium(**kw):
    cfg = RadioConfig(shadowing_db=0.0, jitter_s=0.0, **kw)
    return RadioMedium(cfg, np.random.default_rng(0))


def _run(m, pos, outgoing, jam=()):
    got = m.step(0.1, pos, outgoing, jam)
    for t in np.arange(0.2, 2.0, 0.1):
        for r, frames in m.step(float(t), pos, {}, jam).items():
            got.setdefault(r, []).extend(frames)
    return got


def test_in_range_delivered_out_of_range_not():
    m = _medium(nominal_range_m=500)
    pos = {0: (0, 0, 40), 1: (300, 0, 40), 2: (900, 0, 40)}
    got = _run(m, pos, {0: [b"x" * 30]})
    assert [f for f, _ in got.get(1, [])] == [b"x" * 30]
    assert got[1][0][1] > 0  # margin above sensitivity
    assert 2 not in got


def test_jammed_receiver_hears_nothing():
    m = _medium(nominal_range_m=500)
    pos = {0: (0, 0, 40), 1: (300, 0, 40)}
    jam = [JamZone(0, 100, (300, 0), 50)]
    assert 1 not in _run(m, pos, {0: [b"x" * 30]}, jam)


def test_claims_jump_the_queue():
    from collections import deque

    from earshot.radio.medium import _enqueue
    q = deque()
    for f in [bytes([1]) + b"obs", bytes([2]) + b"trk", bytes([3]) + b"clm", bytes([1]) + b"ob2"]:
        _enqueue(q, f)
    assert [f[0] for f in q] == [3, 2, 1, 1]


def test_simultaneous_equal_power_collide():
    m = _medium(nominal_range_m=500)
    pos = {0: (-200, 0, 40), 1: (200, 0, 40), 2: (0, 0, 40)}
    got = _run(m, pos, {0: [b"a" * 30], 1: [b"b" * 30]})
    assert 2 not in got
