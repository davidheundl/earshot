"""Direction of arrival from one multichannel block. Pure functions, no audio device.

    bearing, power = estimate_bearing(block, fs, mic_xy)     # compass degrees
    level = level_db(block, calibration_db)
    probs = coarse_class_probs(block, fs)                     # stand-in for EfficientAT

Geometry: mic_xy in metres in the array's own frame (x right, y forward). The array's
"forward" is rotated onto city north by `rotation_deg` (measured once when setting up the
room: the direction the array's forward points, as a compass bearing in the miniature city).
The room is a 1:100 miniature of the city, and bearings do not change with scale, so a
bearing measured in the room is a bearing in the city.
"""

from __future__ import annotations

import math

import numpy as np
import pyroomacoustics as pra

from earshot.sensors.acoustics import SPEED_OF_SOUND

BAND = (500.0, 4000.0)  # Hz; below 500 Hz a 6.5 cm array cannot resolve direction
NFFT = 512


def respeaker_v2_geometry(radius: float = 0.032) -> np.ndarray:
    """(2, 4) mic positions of the ReSpeaker Mic Array v2.0 (raw channels 1-4).

    Mic angles are 45, 135, 225, 315 deg on a 32 mm circle. Which raw channel sits at which
    angle must be checked on arrival (tap each mic, watch the channel levels)."""
    ang = np.radians([45.0, 135.0, 225.0, 315.0])
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)])


def estimate_bearing(block: np.ndarray, fs: int, mic_xy: np.ndarray, rotation_deg: float = 0.0,
                     band: tuple[float, float] = BAND, scan: list | None = None) -> tuple[float, float]:
    """(compass bearing in degrees, relative peak power) of the loudest source in the block.

    block: (n_mics, n_samples) float. SRP-PHAT over the band, as in the concept. If `scan` is
    a list, it receives the SRP power over 72 compass directions (every 5 deg, from north)."""
    stft = np.stack([pra.transform.stft.analysis(ch, NFFT, NFFT // 2).T for ch in block])
    doa = pra.doa.algorithms["SRP"](mic_xy, fs, NFFT, c=SPEED_OF_SOUND, num_src=1)
    doa.locate_sources(stft, freq_range=list(band))
    az = float(np.degrees(doa.azimuth_recon[0]))  # counter-clockwise from the array +x axis
    grid = np.asarray(doa.grid.values)
    power = float(grid.max() / max(grid.mean(), 1e-12))
    if scan is not None:
        az_grid = np.degrees(np.asarray(doa.grid.azimuth))
        compass = (90.0 - az_grid + rotation_deg) % 360.0
        bins = np.zeros(72)
        np.maximum.at(bins, (compass // 5).astype(int) % 72, grid / max(grid.max(), 1e-12))
        scan[:] = [round(float(v), 3) for v in bins]
    # array frame: +y is "forward"; compass: clockwise from north
    return (90.0 - az + rotation_deg) % 360.0, power


def level_db(block: np.ndarray, calibration_db: float = 94.0) -> float:
    """Sound level from the RMS of all channels: dBFS + calibration (set with a known source)."""
    rms = float(np.sqrt(np.mean(np.square(block)))) or 1e-12
    return 20.0 * math.log10(rms) + calibration_db


def coarse_class_probs(block: np.ndarray, fs: int) -> tuple[float, float, float, float]:
    """Placeholder classifier until EfficientAT runs: rumble (energy below 300 Hz) reads as
    vehicle/train, a tonal mid band (propeller harmonics) as drone. Indexed by TargetClass."""
    spec = np.abs(np.fft.rfft(block.mean(axis=0))) ** 2
    f = np.fft.rfftfreq(block.shape[1], 1.0 / fs)
    low = spec[f < 300].sum()
    mid = spec[(f >= 300) & (f < 4000)]
    tonal = mid.max() / max(mid.mean(), 1e-12) if mid.size else 0.0
    total = spec.sum() or 1.0
    p = np.array([0.25, 0.25, 0.25, 0.25])
    if low / total > 0.5:
        p = np.array([0.1, 0.45, 0.35, 0.1])
    elif tonal > 20:
        p = np.array([0.7, 0.1, 0.05, 0.15])
    return tuple(float(x) for x in p)  # type: ignore[return-value]

