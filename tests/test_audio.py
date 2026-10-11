import numpy as np
import pytest

from earshot.audio.doa import estimate_bearing, respeaker_v2_geometry

FS = 16000


def plane_wave(compass_deg, mic_xy, seconds=1.0, seed=0):
    """Broadband noise arriving from a compass bearing, delayed per mic (far field)."""
    rng = np.random.default_rng(seed)
    n = int(FS * seconds)
    src = rng.normal(size=n)
    az = np.radians(90.0 - compass_deg)  # array frame: counter-clockwise from +x
    direction = np.array([np.cos(az), np.sin(az)])
    delays = -(mic_xy.T @ direction) / 343.0  # mics closer to the source hear it earlier
    spec = np.fft.rfft(src)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    out = np.stack([np.fft.irfft(spec * np.exp(-2j * np.pi * f * d), n) for d in delays])
    return out + 0.05 * rng.normal(size=out.shape)


@pytest.mark.parametrize("bearing", [0, 45, 100, 200, 300])
def test_srp_phat_finds_the_bearing(bearing):
    mics = respeaker_v2_geometry()
    est, power = estimate_bearing(plane_wave(bearing, mics), FS, mics)
    err = abs((est - bearing + 180) % 360 - 180)
    assert err <= 10, f"estimated {est:.0f} for {bearing}"
    assert power > 1.0


def test_rotation_maps_the_room_onto_city_north():
    mics = respeaker_v2_geometry()
    est, _ = estimate_bearing(plane_wave(90, mics), FS, mics, rotation_deg=30)
    assert abs((est - 120 + 180) % 360 - 180) <= 10


def test_array_sensor_turns_a_block_into_a_reading():
    from earshot.audio.array import ArraySensor
    from earshot.messages import Sensor

    sensor = ArraySensor(device=None, start=False, calibration_db=120.0)
    mics = respeaker_v2_geometry()
    r = sensor.process_block(plane_wave(250, mics))
    assert r is not None and r.sensor == Sensor.SOUND
    assert abs((r.bearing - 250 + 180) % 360 - 180) <= 10
    assert sensor.process_block(np.zeros((4, FS))) is None  # silence: nothing to report
