"""Waveforms: what each drone's microphone hears, as audio, calibrated to the level model.

Interface:
    bank = ClipBank(Path("data/sounds"))            recordings per class, synthetic fallback
    mixer = Mixer(bank)
    win = mixer.window(drone, t, parts, ego_db, background_db)   -> AudioWindow (1 s mix)
    beam = mixer.beam(win, i)                       source i + the others at -MASKING_OFFSET_DB
    clf = load_classifier("pkg.module:attr")        the trained model's hook (Nikolas)

Calibration: a signal of RMS 1.0 is FULL_SCALE_DB. Every part of a window is scaled so its
RMS equals the level `acoustics` computes (received level per source, ego-noise from rpm,
background), so the waveform's SNR is the SNR in the detection logs and those are valid
labels. Clips are sliced by sim time minus the travel time (d / 343 m/s), so consecutive
windows continue each other. No randomness: the same run gives the same audio.

Clips: data/sounds/<class>/*.wav with <class> one of drone, vehicle, train, nothing, ego,
background (any PCM width, mono or stereo, any rate). A class without recordings uses a
synthetic loop (blade-pass harmonics, engine rumble, wheel clacks, fan hum, motor whine,
pink noise): good enough to test the plumbing, not to train a model that meets real sound.

Agents never see waveforms (ADR 0001): only the World renders them, and a classifier turns
a beam into the class probabilities of a SensorReading before it leaves the World.
"""

from __future__ import annotations

import importlib
import math
import wave
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from earshot.messages import TargetClass

SR = 16000  # Hz, the arrays' native rate (audio/array.py)
FULL_SCALE_DB = 110.0  # dB(A) of a signal with RMS 1.0
SPEED_OF_SOUND = 343.0
LOOP_S = 10.0  # length of each synthetic loop
SYNTH_VARIANTS = 3  # synthetic loops per class, so two drones do not sound identical
MASKING_OFFSET_DB = 6.0  # other sources raise the floor, partly separable in frequency (beam)

CLASS_DIRS = {TargetClass.DRONE: "drone", TargetClass.VEHICLE: "vehicle",
              TargetClass.TRAIN: "train", TargetClass.NOTHING: "nothing"}


class Classifier(Protocol):
    """A trained sound classifier: one beam (mono, float, RMS 1.0 = FULL_SCALE_DB) ->
    probabilities indexed by TargetClass (DRONE, VEHICLE, TRAIN, NOTHING), summing to 1."""

    def classify(self, audio: np.ndarray, sr: int) -> tuple[float, float, float, float]: ...


def load_classifier(spec: str) -> Classifier:
    """"pkg.module:attr": attr is a Classifier, a class or factory returning one, or a plain
    function (audio, sr) -> probabilities."""
    mod, _, attr = spec.partition(":")
    obj = getattr(importlib.import_module(mod), attr or "classifier")
    if hasattr(obj, "classify"):
        return obj
    if isinstance(obj, type):
        return obj()
    try:  # a factory taking no arguments
        made = obj()
        if hasattr(made, "classify"):
            return made
    except TypeError:
        pass

    class _Fn:
        def classify(self, audio, sr):
            return tuple(obj(audio, sr))

    return _Fn()


# -- clips ----------------------------------------------------------------------------------


def read_wav(path: Path, sr: int = SR) -> np.ndarray:
    """Mono float32 at `sr`, normalised to RMS 1.0."""
    with wave.open(str(path), "rb") as w:
        n, ch, width, rate = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if width == 3:  # 24-bit: widen to 32
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        raw = np.concatenate([np.zeros((len(b), 1), np.uint8), b], axis=1).tobytes()
        width = 4
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype).astype(np.float64)
    if width == 1:
        x -= 128.0
    x = x.reshape(-1, ch).mean(axis=1)
    if rate != sr:
        x = np.interp(np.arange(0, len(x) * sr / rate) * rate / sr, np.arange(len(x)), x)
    return _unit(x)


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    """32-bit PCM, full scale = FULL_SCALE_DB + 20 dB (peaks above that clip), so a 40 dB
    background still keeps 20 bits."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(4)
        w.setframerate(sr)
        w.writeframes(pcm32(audio))


def pcm32(audio: np.ndarray) -> bytes:
    """Little-endian 32-bit PCM, full scale = FULL_SCALE_DB + 20 dB."""
    x = np.clip(np.asarray(audio, np.float64) / 10.0, -1.0, 1.0)
    return (x * (2**31 - 1)).astype("<i4").tobytes()


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64) - np.mean(x)
    rms = math.sqrt(float(np.mean(x * x)))
    return (x / rms if rms > 0 else x).astype(np.float32)


def _pink(rng: np.random.Generator, n: int) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) / np.sqrt(np.maximum(f, 20.0))
    return np.fft.irfft(spec, n)


def _brown(rng: np.random.Generator, n: int, corner: float = 80.0) -> np.ndarray:
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = (rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) / np.maximum(f, corner)
    return np.fft.irfft(spec, n)


def _harmonics(t: np.ndarray, f0: np.ndarray | float, n: int, decay: float,
               rng: np.random.Generator) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(np.broadcast_to(f0, t.shape)) / SR
    return sum(decay**k * np.sin((k + 1) * phase + rng.uniform(0, 2 * np.pi)) for k in range(n))


def synth(kind: str, variant: int) -> np.ndarray:
    """A LOOP_S synthetic loop for one class (or "ego" / "background"), RMS 1.0."""
    rng = np.random.default_rng(zlib.crc32(f"{kind}/{variant}".encode()))
    n = int(LOOP_S * SR)
    t = np.arange(n) / SR
    if kind in ("drone", "ego"):  # blade-pass harmonics, rpm wandering, broadband rotor wash
        f0 = (190.0 if kind == "drone" else 200.0) * (1 + 0.15 * variant)
        wobble = 1 + 0.03 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t) \
            + 0.01 * np.sin(2 * np.pi * rng.uniform(2, 4) * t)
        x = _harmonics(t, f0 * wobble, 14, 0.8, rng) + 0.6 * _unit(_pink(rng, n))
    elif kind == "vehicle":  # engine firing harmonics over tyre and road rumble
        f0 = 35.0 * (1 + 0.2 * variant) * (1 + 0.05 * np.sin(2 * np.pi * 0.1 * t))
        x = 0.7 * _harmonics(t, f0, 10, 0.75, rng) + _unit(_brown(rng, n))
    elif kind == "train":  # deep rumble with wheel clacks on rail joints
        x = _unit(_brown(rng, n, 40.0)) + 0.3 * _harmonics(t, 28.0 + 6 * variant, 6, 0.7, rng)
        clack = np.zeros(n)
        period = 0.7 + 0.15 * variant
        for start in np.arange(rng.uniform(0, period), LOOP_S, period):
            for dt in (0.0, 0.09):
                i = int((start + dt) * SR)
                m = min(n - i, int(0.05 * SR))
                if m > 0:
                    clack[i:i + m] += np.exp(-np.arange(m) / (0.008 * SR)) * rng.normal(size=m)
        x = x + 4.0 * clack
    elif kind == "nothing":  # fan / AC unit: steady tonal hum, drone-like but no wobble
        x = _harmonics(t, 120.0 + 25 * variant, 8, 0.7, rng) + 0.3 * np.sin(2 * np.pi * 50 * t) \
            + 0.8 * _unit(_pink(rng, n))
    elif kind == "background":
        x = _pink(rng, n)
    else:
        raise ValueError(kind)
    return _unit(x)


class ClipBank:
    """Recordings per class from `root`/<class>/*.wav; a class without any is synthesised."""

    def __init__(self, root: Path | None = Path("data/sounds")):
        self.clips: dict[str, list[np.ndarray]] = {}
        self.recorded: dict[str, int] = {}
        for kind in [*CLASS_DIRS.values(), "ego", "background"]:
            files = sorted((root / kind).glob("*.wav")) if root and (root / kind).is_dir() else []
            clips = [c for c in (read_wav(f) for f in files) if len(c) >= SR]
            self.recorded[kind] = len(clips)
            self.clips[kind] = clips or [synth(kind, v) for v in range(SYNTH_VARIANTS)]

    def clip(self, kind: str, key: str) -> np.ndarray:
        """The clip a given source (by id) uses, stable across runs."""
        c = self.clips[kind]
        return c[zlib.crc32(key.encode()) % len(c)]


# -- mixing ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Part:
    """One source as one drone hears it in one window."""

    source: str
    cls: TargetClass
    level_db: float  # received level (acoustics.received_level)
    distance: float
    n_buildings: int
    bearing: float  # true bearing, degrees from north
    snr_db: float


@dataclass
class AudioWindow:
    drone: int
    t: float  # end of the window, sim seconds
    mix: np.ndarray  # everything the microphone gets
    parts: list[Part]
    signals: list[np.ndarray] = field(repr=False)  # per part, calibrated
    noise: np.ndarray = field(repr=False)  # background + ego-noise
    ego_db: float
    background_db: float

    def label(self) -> dict:
        return {"drone": self.drone, "t": round(self.t, 3), "ego_db": _r(self.ego_db),
                "background_db": _r(self.background_db), "level_db": _r(_db(self.mix)),
                "sources": [{"id": p.source, "cls": p.cls.name, "level_db": _r(p.level_db),
                             "snr_db": _r(p.snr_db), "distance": round(p.distance, 1),
                             "buildings": p.n_buildings, "bearing": round(p.bearing, 1)}
                            for p in self.parts]}


def _r(x: float) -> float | None:
    return None if x == -math.inf else round(x, 2)


def _db(x: np.ndarray) -> float:
    ms = float(np.mean(np.square(x, dtype=np.float64)))
    return FULL_SCALE_DB + 10 * math.log10(ms) if ms > 0 else -math.inf


def _amp(level_db: float) -> float:
    return 0.0 if level_db == -math.inf else 10 ** ((level_db - FULL_SCALE_DB) / 20)


def _lowpass(x: np.ndarray, fc: float) -> np.ndarray:
    if fc >= SR / 2:
        return x
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(spec / np.sqrt(1 + (f / fc) ** 4), len(x)).astype(np.float32)


def _slice(clip: np.ndarray, t_end: float, n: int) -> np.ndarray:
    start = round(t_end * SR) - n
    idx = (start + np.arange(n)) % len(clip)
    return clip[idx]


class Mixer:
    def __init__(self, bank: ClipBank, window_s: float = 1.0):
        self.bank = bank
        self.n = int(window_s * SR)

    def window(self, drone: int, t: float, parts: list[Part], ego_db: float,
               background_db: float) -> AudioWindow:
        signals = []
        for p in parts:
            x = _slice(self.bank.clip(CLASS_DIRS[p.cls], p.source),
                       t - p.distance / SPEED_OF_SOUND, self.n)
            # air absorbs high frequencies first; buildings in the way muffle further
            fc = 12000.0 / (1 + p.distance / 400.0) / (1 + p.n_buildings)
            signals.append(_unit(_lowpass(x, max(fc, 300.0))) * _amp(p.level_db))
        bg = _slice(self.bank.clip("background", f"drone{drone}"), t, self.n) * _amp(background_db)
        ego = _slice(self.bank.clip("ego", f"drone{drone}"), t, self.n) * _amp(ego_db)
        noise = (bg + ego).astype(np.float32)
        mix = noise + sum(signals, np.zeros(self.n, np.float32))
        return AudioWindow(drone, t, mix.astype(np.float32), parts, signals, noise, ego_db,
                           background_db)

    def beam(self, win: AudioWindow, i: int) -> np.ndarray:
        """What a beam steered at part i hears: it, the others MASKING_OFFSET_DB down, noise.
        Matches the masking in the SNR model, so the classifier sees the reading's SNR."""
        g = 10 ** (-MASKING_OFFSET_DB / 20)
        others = sum((s for j, s in enumerate(win.signals) if j != i), np.zeros(self.n, np.float32))
        return (win.signals[i] + g * others + win.noise).astype(np.float32)
