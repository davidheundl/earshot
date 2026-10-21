"""Passive RF energy per band, concept sections 5.2 and 9. Pure functions, no randomness.

We measure only how much energy arrives in which band; nothing is decoded (section 2.4).

    rx_dBm = EIRP - FSPL(1 m, f) - 10 n log10(d) + shadowing
    n = 2.5 with a clear line, 3.0 through buildings; shadowing sigma 3 / 6 dB

Detection in one 1 s window: logistic around SNR = -14 dB, reaching 85 % at -12 dB (the
noise-robust classifier quoted in section 9). Noise floors include urban interference
(2.4 GHz is Wi-Fi-crowded), chosen so the through-building ranges match section 5.2:
2.4 GHz video ~400 m, 5.8 GHz ~150 m, 1 W 915 MHz control link several km.

Each band names the Target Class its emitters are (`emitter`): the drone links are drones;
the UHF band (400-470 MHz handheld radios and their push-to-talk bursts) is people in a
vehicle, the poachers of a Reserve. A Probability Map adds a band's ring to that class only.
UHF: a 2 W handheld at 1.5 m above the bush to a drone on a 12 m tower. Ground reflection
and the bush make that path fall off with n of about 4 beyond a few hundred metres, which
the single exponent here does not model; the noise floor (VHF/UHF repeater traffic in a
25 MHz energy detector) is chosen so the 50 % range is about 5 km in the open and 1.2 km
behind a kopje, what ranger radio networks report for handhelds. `duty` is the share of 1 s
windows in which the emitter transmits at all (push-to-talk).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from earshot.messages import Sensor, TargetClass


@dataclass(frozen=True)
class Band:
    sensor: Sensor
    eirp_dbm: float
    freq_hz: float
    noise_floor_dbm: float
    emitter: TargetClass = TargetClass.DRONE  # what transmits in this band
    duty: float = 1.0  # share of 1 s windows with a transmission

    @property
    def fspl_1m(self) -> float:
        return 20.0 * math.log10(4.0 * math.pi * self.freq_hz / 3e8)


BANDS = {
    "2g4": Band(Sensor.RF_2G4, 20.0, 2.44e9, -85.0),  # 100 mW video downlink
    "5g8": Band(Sensor.RF_5G8, 14.0, 5.80e9, -85.0),  # 25 mW analog FPV video
    "915": Band(Sensor.RF_915, 30.0, 0.915e9, -105.0),  # 1 W long-range control link
    "uhf": Band(Sensor.RF_UHF, 33.0, 0.45e9, -71.0, TargetClass.VEHICLE, 0.4),  # handhelds
}
BAND_OF_SENSOR = {b.sensor: b for b in BANDS.values()}

EXP_LOS, EXP_NLOS = 2.5, 3.0
SHADOW_LOS_DB, SHADOW_NLOS_DB = 3.0, 6.0
SNR50_DB = -14.0
SLOPE_DB = 2.0 / math.log(0.85 / 0.15)  # 85 % at -12 dB


def mean_rx_power(band: Band, d: float, line_of_sight: bool) -> float:
    n = EXP_LOS if line_of_sight else EXP_NLOS
    return band.eirp_dbm - band.fspl_1m - 10.0 * n * math.log10(max(d, 1.0))


def shadowing_sigma(line_of_sight: bool) -> float:
    return SHADOW_LOS_DB if line_of_sight else SHADOW_NLOS_DB


def detection_probability(band: Band, rx_dbm: float) -> float:
    snr = rx_dbm - band.noise_floor_dbm
    return 1.0 / (1.0 + math.exp(-(snr - SNR50_DB) / SLOPE_DB))
