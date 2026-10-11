"""A microphone array in the room as a perched drone's sound sensor.

    sensor = ArraySensor(device="ReSpeaker 4 Mic Array", rotation_deg=0)
    readings = sensor.readings()   # SensorReadings from the blocks captured since last call

Captures 1 s blocks over sounddevice (open at the array's native 16 kHz so macOS does not
resample), estimates the bearing with SRP-PHAT on the raw mic channels, and reports
SensorReadings exactly like the simulated sensors do: the swarm cannot tell the difference.

ReSpeaker Mic Array v2.0 with the 6-channel firmware: channel 0 is the processed output,
channels 1-4 the raw mics, channel 5 the playback reference.
"""

from __future__ import annotations

import queue

import numpy as np

from earshot.audio.doa import (
    coarse_class_probs,
    estimate_bearing,
    level_db,
    respeaker_v2_geometry,
)
from earshot.messages import Sensor
from earshot.state import SensorReading

BEARING_SIGMA = 5.0  # degrees above 1 kHz (concept 5.1)
MIN_POWER = 1.5  # SRP peak over mean: below this the block has no clear direction
MIN_LEVEL_DB = 45.0  # quieter blocks are room noise


class ArraySensor:
    def __init__(self, device: str | int, rotation_deg: float = 0.0, fs: int = 16000,
                 n_channels: int = 6, mic_channels=(1, 2, 3, 4), calibration_db: float = 94.0,
                 start: bool = True):
        self.device = device
        self.rotation_deg = rotation_deg
        self.fs = fs
        self.n_channels = n_channels
        self.mic_channels = list(mic_channels)
        self.calibration_db = calibration_db
        self.mics = respeaker_v2_geometry()
        self._q: queue.Queue = queue.Queue()
        self._stream = None
        if start:
            self.start()

    def start(self) -> None:
        import sounddevice as sd  # only needed with real hardware

        self._stream = sd.InputStream(device=self.device, channels=self.n_channels,
                                      samplerate=self.fs, blocksize=self.fs, dtype="float32",
                                      callback=lambda data, *_: self._q.put(data.copy()))
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def readings(self) -> list[SensorReading]:
        out = []
        while True:
            try:
                block = self._q.get_nowait()
            except queue.Empty:
                return out
            r = self.process_block(block[:, self.mic_channels].T)
            if r is not None:
                out.append(r)

    def process_block(self, block: np.ndarray) -> SensorReading | None:
        """(n_mics, n_samples) -> one SensorReading, or None if nothing clear was heard."""
        level = level_db(block, self.calibration_db)
        if level < MIN_LEVEL_DB:
            return None
        bearing, power = estimate_bearing(block, self.fs, self.mics, self.rotation_deg)
        if power < MIN_POWER:
            return None
        return SensorReading(Sensor.SOUND, bearing, BEARING_SIGMA,
                             coarse_class_probs(block, self.fs), level)


def list_arrays(min_channels: int = 4) -> list[tuple[int, str, int]]:
    import sounddevice as sd

    return [(i, d["name"], d["max_input_channels"]) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] >= min_channels]
