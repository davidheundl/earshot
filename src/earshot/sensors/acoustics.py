"""Sound propagation and detection, concept sections 5.1 and 9. Pure functions, no randomness.

Received level (dB(A)) at distance d from a source of sound power Lw:

    Lp = Lw - (20 log10 d + 11) - a_cls * d / 1000 - building_loss

Noise at the microphone is background (+) ego-noise (+) masking by other sources. A detection
in one 1 s window succeeds with probability logistic((SNR - SNR50_cls) / SLOPE).

Calibration: the per-class constants below were backed out of the section 5.1 range table
(tests/test_sensors.py checks the 50 % detection ranges against it):
  - one threshold of about -8 to -9 dB SNR explains every row (tonal / long-integration
    detectors work below the broadband noise floor);
  - vehicle and train rumble is low-frequency, so air absorbs far less than the 3.7 dB/km
    quoted at 1 kHz (0.7 and 0.5 dB/km make the night and day columns agree);
  - low frequencies bend around buildings better: -10 dB per building vs. -20 for drone tones.

A rifle Shot (a Scenario `shot` Event, not a Source) uses SHOT in place of a class's constants:
every function here takes a TargetClass or a ClassAcoustics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from earshot.messages import TargetClass


@dataclass(frozen=True)
class ClassAcoustics:
    absorption_db_per_km: float
    building_loss_db: float  # per blocking building
    snr50_db: float  # SNR at which half of the 1 s windows detect
    bearing_sigma_perched: float  # degrees; 5 deg above 1 kHz, 10-20 deg at 250-500 Hz


ACOUSTICS = {
    TargetClass.DRONE: ClassAcoustics(3.7, 20.0, -8.0, 5.0),
    TargetClass.VEHICLE: ClassAcoustics(0.7, 10.0, -9.0, 15.0),
    TargetClass.TRAIN: ClassAcoustics(0.5, 10.0, -9.0, 15.0),
    TargetClass.NOTHING: ClassAcoustics(3.7, 20.0, -8.0, 5.0),  # nuisance: fan, AC unit
}

# A rifle Shot: one impulse. Its level is what the 1 s window integrates (its sound exposure),
# not the muzzle blast's peak; its energy sits higher than an engine's rumble (0.5-1 kHz), so
# air absorbs more; an onset detector times the impulse across the array to a few degrees.
# Chosen, not measured: a tower hears it almost surely at 3 km in the open at night, half the
# time at 6 km, rarely beyond 7 km; a kopje in the way hides it beyond about 2.5 km.
SHOT = ClassAcoustics(2.0, 15.0, 0.0, 2.0)
SHOT_LEVEL_DB = 134.0
SPEED_OF_SOUND = 343.0  # m/s: a tower 5 km away hears a Shot 15 s after it was fired

# Typical sound power levels, dB(A) (section 9; larger drone and train backed out of 5.1).
SOUND_POWER = {"small_drone": 82.0, "large_drone": 89.0, "heavy_vehicle": 105.0,
               "freight_train": 120.0, "fan": 70.0}

# Default sound power per class when an event does not say (small drone, truck, train, fan).
DEFAULT_LEVEL = {TargetClass.DRONE: 82.0, TargetClass.VEHICLE: 105.0,
                 TargetClass.TRAIN: 120.0, TargetClass.NOTHING: 75.0}

SLOPE_DB = 2.0
MAX_BUILDING_LOSS_DB = 25.0  # barrier attenuation saturates (ISO 9613-2 caps at 20-25 dB)
EGO_HOVER_DB = 85.0  # at the drone's own microphones, motors at hover speed
EGO_RPM_REF = 6000.0
BEARING_SIGMA_FLYING = 25.0  # degrees (20-30)
OUTLIER_PERCHED = 0.05
OUTLIER_FLYING = 0.25  # (20-35 %)


def _acoustics(cls: TargetClass | ClassAcoustics) -> ClassAcoustics:
    return cls if isinstance(cls, ClassAcoustics) else ACOUSTICS[cls]


def received_level(lw_db: float, cls: TargetClass | ClassAcoustics, d: float,
                   n_buildings: int) -> float:
    a = _acoustics(cls)
    d = max(d, 1.0)
    building = min(n_buildings * a.building_loss_db, MAX_BUILDING_LOSS_DB)
    return lw_db - (20.0 * math.log10(d) + 11.0) - a.absorption_db_per_km * d / 1000.0 - building


def ego_noise(rpm: float, ego_filter: bool, filter_gain_db: float) -> float:
    """Own motor noise at the microphones. Motors off -> -inf.

    Mostly "motors on or off": hovering is near the most power-hungry state (section 4), so rpm
    only nudges the level. The filter subtracts the noise predicted from the known rpm.
    """
    if rpm <= 0:
        return -math.inf
    level = EGO_HOVER_DB + 20.0 * math.log10(rpm / EGO_RPM_REF)
    return level - (filter_gain_db if ego_filter else 0.0)


def db_sum(*levels: float) -> float:
    """Incoherent sum of levels in dB; -inf entries are silence."""
    p = sum(10.0 ** (lv / 10.0) for lv in levels if lv != -math.inf)
    return 10.0 * math.log10(p) if p > 0 else -math.inf


def detection_probability(snr_db: float, cls: TargetClass | ClassAcoustics) -> float:
    return 1.0 / (1.0 + math.exp(-(snr_db - _acoustics(cls).snr50_db) / SLOPE_DB))


def bearing_sigma(cls: TargetClass | ClassAcoustics, perched: bool, snr_db: float) -> float:
    """Bearing error grows as the signal sinks toward the detection threshold (up to x3)."""
    a = _acoustics(cls)
    base = a.bearing_sigma_perched if perched else BEARING_SIGMA_FLYING
    margin = snr_db - (a.snr50_db + 10.0)
    return base * min(3.0, max(1.0, 10.0 ** (-margin / 20.0)))


def outlier_rate(perched: bool) -> float:
    return OUTLIER_PERCHED if perched else OUTLIER_FLYING
