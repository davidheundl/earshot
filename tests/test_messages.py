import math

import pytest

from earshot.messages import (
    MAX_FRAME,
    BadFrame,
    Claim,
    Digest,
    Header,
    Mode,
    Observation,
    Request,
    Sensor,
    TargetClass,
    Track,
    TrackStatus,
    decode,
    encode,
    with_hops,
)

KEY = b"test-key"
H = Header(origin=3, seq=513, hops_left=3)

SAMPLES = [
    Observation(H, 12.345, -812.0, 640.0, 2.0, 123.45, 5.0, (0.7, 0.1, 0.1, 0.1),
                Sensor.SOUND, Mode.LISTEN, 61.5),
    Track(H, 99.5, 7, 150.0, -300.0, 100.0, 40.0, 30.0, TargetClass.DRONE, 0.8,
          TrackStatus.SUSPECTED),
    Claim(H, 5.0, 42, 3.21, 12.0),
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
    assert back.bearing == pytest.approx(123.45, abs=0.01)
    assert back.class_probs[0] == pytest.approx(0.7, abs=0.005)
    assert back.level == pytest.approx(61.5, abs=0.01)


def test_rf_observation_has_no_bearing():
    obs = Observation(H, 1.0, 0, 0, 2.0, float("nan"), 0.0, (0.6, 0.1, 0.1, 0.2),
                      Sensor.RF, Mode.PATROL, -80.0)
    assert math.isnan(decode(encode(obs, KEY), KEY).bearing)


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
