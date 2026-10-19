"""Shared by tools/train_sound_head.py and tools/eval_sound_classifier.py: the windows from
tools/fetch_sound_data.py, mixed into what a beam sounds like, and their features.

A beam (sensors.waveform.Mixer.beam) is the Source, other Sources 6 dB down, background and
the drone's own motors. `noisy` imitates that on a real window: background and ego-noise
(the simulator's synthetic loops) at a given SNR, sometimes a second real window as a masker,
and a distance lowpass like Mixer.window's.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from earshot.audio import classifier as clf
from earshot.sensors import waveform as wf

WINDOWS = Path("data/raw/sounds/windows.npz")
CACHE = Path("data/raw/sounds/features")
SNR_RANGE = (-10.0, 20.0)  # dB; sim detections start around -8 (acoustics.ACOUSTICS.snr50_db)


def load() -> dict[str, np.ndarray]:
    d = np.load(WINDOWS)
    return {k: d[k] for k in d.files}


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    r = np.sqrt(np.mean(x * x))
    return x / r if r > 0 else x


_NOISE = {k: [wf.synth(k, v) for v in range(wf.SYNTH_VARIANTS)] for k in ("background", "ego")}


def noisy(x: np.ndarray, snr_db: float, rng: np.random.Generator,
          masker: np.ndarray | None = None) -> np.ndarray:
    """Window x (any scale) at `snr_db` over background + ego-noise, lowpassed for distance,
    with an optional masker 6 dB under x."""
    n = len(x)
    s = unit(wf._lowpass(unit(x).astype(np.float32), float(rng.uniform(1500, 12000))))
    bg = _NOISE["background"][rng.integers(wf.SYNTH_VARIANTS)]
    ego = _NOISE["ego"][rng.integers(wf.SYNTH_VARIANTS)]
    i, j = rng.integers(len(bg) - n), rng.integers(len(ego) - n)
    ego_share = rng.uniform(0.0, 1.0)  # perched (motors off) .. flying
    noise = unit(np.sqrt(1 - ego_share) * unit(bg[i:i + n]) + np.sqrt(ego_share) * unit(ego[j:j + n]))
    out = s + noise * 10 ** (-snr_db / 20)
    if masker is not None:
        out = out + unit(masker) * 10 ** (-wf.MASKING_OFFSET_DB / 20)
    return out


def features(audio: np.ndarray, tag: str, device: str | None = None,
             batch: int = 32) -> np.ndarray:
    """Head features for each row of `audio` (n, 16000), cached under `tag` + content hash."""
    h = hashlib.sha1(np.ascontiguousarray(audio).view(np.uint8)).hexdigest()[:12]
    path = CACHE / f"{tag}-{h}.npy"
    if path.exists():
        return np.load(path)
    if device is None:
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    bb = clf.Backbone(device=device)
    out = []
    for k in range(0, len(audio), batch):
        logits, emb = bb(audio[k:k + batch], wf.SR)
        out.append(clf.features(logits, emb))
        if (k // batch) % 20 == 0:
            print(f"  {tag}: {k + len(audio[k:k + batch])}/{len(audio)}", flush=True)
    f = np.concatenate(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, f)
    return f


def variants(d: dict[str, np.ndarray], split: str, seed: int, per_window: int,
             clean: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(audio, fine label index, snr) for one split: each window clean (snr inf) and
    `per_window` noisy copies at random SNR, a third of them with a masker."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(d["split"] == split)
    audio = d["audio"][idx].astype(np.float64) / 32767
    y = np.array([clf.FINE.index(f) for f in d["fine"][idx]])
    xs, ys, snrs = [], [], []
    for a, label in zip(audio, y, strict=True):
        if clean:
            xs.append(unit(a))
            ys.append(label)
            snrs.append(np.inf)
        for _ in range(per_window):
            snr = float(rng.uniform(*SNR_RANGE))
            m = audio[rng.integers(len(audio))] if rng.random() < 1 / 3 else None
            xs.append(noisy(a, snr, rng, m))
            ys.append(label)
            snrs.append(snr)
    return np.stack(xs).astype(np.float32), np.array(ys), np.array(snrs)
