import math
from dataclasses import replace

import pytest

from earshot.messages import (
    MAX_FRAME,
    TAG_LEN,
    BadFrame,
    Claim,
    Digest,
    Header,
    Mode,
    Observation,
    Request,
    Sensor,
    SensorReading,
    TargetClass,
    Track,
    TrackStatus,
    decode,
    encode,
    with_hops,
)
from earshot.messages import _tag as tag

KEY = b"test-key"
H = Header(origin=3, seq=513, hops_left=3)
SOUND = SensorReading(Sensor.SOUND, 123.45, 5.0, (0.7, 0.1, 0.1, 0.1), 61.5)
UHF = SensorReading(Sensor.RF_UHF, float("nan"), 0.0, (0.0333, 0.9, 0.0333, 0.0334), -101.25)

SAMPLES = [
    Observation(H, 12.345, -812.0, 640.0, Mode.LISTEN, (SOUND, UHF)),
    Track(H, 99.5, 7, 150.0, -300.0, 100.0, 40.0, 30.0, TargetClass.DRONE, 0.8,
          TrackStatus.SUSPECTED, 12.3),
    Claim(H, 5.0, 42, 3.21, 12.0, -340.0, -560.0),
    Digest(H, 1.0, tuple((o, 1000 + o) for o in range(6))),
    Request(H, 2, 10, 20),
]


@pytest.mark.parametrize("msg", SAMPLES, ids=lambda m: type(m).__name__)
def test_roundtrip_and_size(msg):
    frame = encode(msg, KEY)
    assert len(frame) <= MAX_FRAME
    back = decode(frame, KEY)
    assert type(back) is type(msg)
    assert back.header == msg.header


def test_observation_quantisation():
    back = decode(encode(SAMPLES[0], KEY), KEY)
    assert (back.t, back.x, back.y, back.mode) == (12.345, -812.0, 640.0, Mode.LISTEN)
    sound, uhf = back.readings
    assert sound.sensor == Sensor.SOUND and uhf.sensor == Sensor.RF_UHF
    assert sound.bearing == pytest.approx(123.45, abs=0.03)
    assert sound.bearing_sigma == 5.0
    assert sound.class_probs == pytest.approx((0.7, 0.1, 0.1, 0.1), abs=0.01)  # Nothing: the rest
    assert sound.level == pytest.approx(61.5, abs=0.01)
    assert math.isnan(uhf.bearing) and uhf.level == pytest.approx(-101.25, abs=0.01)


def test_two_readings_fill_one_frame_and_one_is_smaller():
    assert len(encode(SAMPLES[0], KEY)) == MAX_FRAME
    one = replace(SAMPLES[0], readings=(SOUND,))
    assert len(encode(one, KEY)) == MAX_FRAME - 8
    with pytest.raises(ValueError):
        encode(replace(SAMPLES[0], readings=(SOUND,) * (Observation.MAX_READINGS + 1)), KEY)
    with pytest.raises(ValueError):
        encode(replace(SAMPLES[0], readings=()), KEY)


def test_bearings_wrap_at_north():
    for b in (0.0, 359.99, 360.0, -0.01):
        r = replace(SOUND, bearing=b)
        back = decode(encode(replace(SAMPLES[0], readings=(r,)), KEY), KEY).readings[0].bearing
        assert 0.0 <= back < 360.0
        assert abs((back - b + 180.0) % 360.0 - 180.0) <= 0.03


def test_non_finite_fields_encode_instead_of_crashing():
    bad = replace(SAMPLES[0], readings=(replace(SOUND, level=math.nan, bearing_sigma=math.inf),))
    back = decode(encode(bad, KEY), KEY)
    assert len(encode(bad, KEY)) <= 40 and math.isfinite(back.readings[0].level)


def test_rf_observation_has_no_bearing():
    obs = Observation(H, 1.0, 0, 0, Mode.PATROL,
                      (SensorReading(Sensor.RF_2G4, float("nan"), 0.0, (0.6, 0.1, 0.1, 0.2), -80.0),))
    assert math.isnan(decode(encode(obs, KEY), KEY).readings[0].bearing)


def test_a_malformed_observation_is_a_bad_frame():
    head = encode(SAMPLES[0], KEY)[:-TAG_LEN]
    for body in (head[:-3],  # not a whole reading
                 head[:5 + 9],  # no reading at all
                 head[:5 + 9] + bytes([0xFF, 0xFF]) + head[5 + 11:5 + 17]):  # sensor 7
        with pytest.raises(BadFrame):
            decode(body + tag(KEY, body[:4] + body[5:]), KEY)


def test_tampered_or_wrong_key_rejected():
    frame = bytearray(encode(SAMPLES[1], KEY))
    frame[8] ^= 0xFF
    with pytest.raises(BadFrame):
        decode(bytes(frame), KEY)
    with pytest.raises(BadFrame):
        decode(encode(SAMPLES[1], KEY), b"other-key")


def test_forwarding_keeps_tag_valid():
    fwd = with_hops(encode(SAMPLES[0], KEY), 2)
    assert decode(fwd, KEY).header.hops_left == 2
