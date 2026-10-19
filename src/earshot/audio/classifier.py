"""Sound classifiers on EfficientAT (AudioSet-pretrained MobileNet), for `--classifier`.

    --classifier earshot.audio.classifier:zeroshot      AudioSet labels folded into 4 classes
    --classifier earshot.audio.classifier:efficientat   trained head (tools/train_sound_head.py)

Both take one beam (mono float, any level, any rate) and return probabilities indexed by
TargetClass, like waveform.Classifier asks. Fine classes are what the head learns; FOLD maps
them onto the 4 Target Classes the radio carries: a plane or helicopter is no threat to a
site and counts as NOTHING, like a fan.

Needs the `ml` group (`uv sync --group ml`) and the weights (tools/fetch_efficientat.py).
Inference only: eval mode, no augmentation, no random numbers, so lockstep runs stay
reproducible.
"""

from __future__ import annotations

import csv
import math
import os
from functools import cache
from pathlib import Path

import numpy as np

from earshot.messages import TargetClass

MODELS = Path("data/models")
BACKBONE = MODELS / "mn10_as_mAP_471.pt"
HEAD = MODELS / "sound_head.pt"
INPUT_RMS = 0.05  # beams arrive calibrated (RMS 1 = 110 dB); the net wants ordinary audio
# The AudioSet models were trained on 10 s clips: on 1 s their logits blow up (edge effects
# dominate the pooled features). Tiling the window to CONTEXT_S brings them back in range.
CONTEXT_S = 5.0
N_LABELS = 527  # AudioSet

# machine: fans, vacuum cleaners, saws (the Nuisances); insect: buzzing that sounds drone-like
FINE = ("drone", "plane", "helicopter", "vehicle", "train", "machine", "insect", "ambient",
        "other")
_C = TargetClass
FOLD = {"drone": _C.DRONE, "plane": _C.NOTHING, "helicopter": _C.NOTHING,
        "vehicle": _C.VEHICLE, "train": _C.TRAIN, "machine": _C.NOTHING, "insect": _C.NOTHING,
        "ambient": _C.NOTHING, "other": _C.NOTHING}

# AudioSet labels per fine class for the zero-shot baseline. AudioSet has no UAV label: a
# small drone is closest to propeller / light engine, and these also fire on planes.
ZEROSHOT = {
    "drone": ["Propeller, airscrew", "Light engine (high frequency)"],
    "plane": ["Aircraft", "Aircraft engine", "Jet engine", "Fixed-wing aircraft, airplane"],
    "helicopter": ["Helicopter"],
    "vehicle": ["Car", "Car passing by", "Motor vehicle (road)", "Traffic noise, roadway noise",
                "Motorcycle", "Truck", "Bus", "Medium engine (mid frequency)",
                "Heavy engine (low frequency)", "Vehicle horn, car horn, honking"],
    "train": ["Train", "Rail transport", "Railroad car, train wagon", "Train wheels squealing"],
    "machine": ["Mechanical fan", "Air conditioning", "Vacuum cleaner", "Chainsaw", "Sawing",
                "Hum", "Mains hum"],
    "insect": ["Insect", "Bee, wasp, etc.", "Mosquito", "Buzz"],
    "ambient": ["Wind", "Wind noise (microphone)", "Rain", "Silence",
                "Outside, urban or manmade"],
    "other": ["Speech", "Music", "Animal", "Bird"],
}


def fold(fine_probs: np.ndarray) -> tuple[float, float, float, float]:
    """Probabilities over FINE -> over TargetClass, summing to 1."""
    out = np.zeros(len(TargetClass))
    for name, p in zip(FINE, fine_probs, strict=True):
        out[FOLD[name]] += p
    s = out.sum()
    return tuple(out / s) if s > 0 else (0.25,) * 4  # type: ignore[return-value]


@cache
def audioset_labels() -> list[str]:
    with open(Path(__file__).parent / "efficientat" / "audioset_labels.csv") as f:
        return [row["display_name"] for row in csv.DictReader(f)]


class Backbone:
    """EfficientAT mn10_as: audio -> (AudioSet logits (527,), embedding (960,)).

    `device`: "cpu" (default, ~170 ms per beam on Apple silicon, whose CPU torch has no fast
    depthwise convolution) or "mps" (~18 ms per beam batched, for feature extraction).
    """

    def __init__(self, weights: Path = BACKBONE, device: str = "cpu"):
        import torch

        from earshot.audio import efficientat

        if not weights.exists():
            raise FileNotFoundError(f"{weights}: run tools/fetch_efficientat.py")
        self.torch = torch
        self.device = device
        self.net = efficientat.load(weights).to(device)
        self.mel = efficientat.MelFrontend().eval().to(device)
        self.sr = efficientat.SR

    def prepare(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Mono at 32 kHz, RMS INPUT_RMS, tiled to CONTEXT_S."""
        from scipy.signal import resample_poly

        x = np.asarray(audio, np.float64).ravel()
        if sr != self.sr:
            g = math.gcd(sr, self.sr)
            x = resample_poly(x, self.sr // g, sr // g)
        x = x - x.mean()
        rms = math.sqrt(float(np.mean(x * x)))
        x = x * (INPUT_RMS / rms) if rms > 0 else x
        n = int(CONTEXT_S * self.sr)
        return np.tile(x, -(-n // len(x)))[:n].astype(np.float32)

    def __call__(self, batch: list[np.ndarray] | np.ndarray, sr: int):
        """Clips of equal length -> (logits (n, 527), embeddings (n, 960)) as numpy."""
        torch = self.torch
        xs = np.stack([self.prepare(a, sr) for a in np.atleast_2d(batch)])
        with torch.inference_mode():
            logits, emb = self.net(self.mel(torch.from_numpy(xs).to(self.device)))
        return logits.cpu().numpy().astype(np.float64), emb.cpu().numpy().astype(np.float64)


@cache
def _backbone() -> Backbone:
    """The shared backbone; EARSHOT_SOUND_DEVICE=mps runs it on the Apple GPU (~2.5x faster
    per beam, same numbers to ~1e-5)."""
    return Backbone(device=os.environ.get("EARSHOT_SOUND_DEVICE", "cpu"))


class ZeroShot:
    """No training: each fine class scores the strongest of its AudioSet labels."""

    def __init__(self):
        self.backbone = _backbone()
        names = audioset_labels()
        self.idx = [[names.index(n) for n in ZEROSHOT[c]] for c in FINE]

    def fine(self, audio: np.ndarray, sr: int) -> np.ndarray:
        logits, emb = self.backbone(audio, sr)
        return self.fine_from_features(features(logits, emb))[0]

    def fine_from_features(self, f: np.ndarray) -> np.ndarray:
        p = f[:, -N_LABELS:]  # AudioSet label probabilities (see features)
        score = np.stack([p[:, i].max(axis=1) for i in self.idx], axis=1) + 1e-3
        return score / score.sum(axis=1, keepdims=True)

    def classify(self, audio: np.ndarray, sr: int) -> tuple[float, float, float, float]:
        return fold(self.fine(audio, sr))


def features(logits: np.ndarray, emb: np.ndarray) -> np.ndarray:
    """What the head sees: the embedding plus AudioSet label probabilities."""
    return np.concatenate([emb, 1 / (1 + np.exp(-logits))], axis=1)


def make_head(n_in: int, n_out: int, hidden: int = 128):
    from torch import nn

    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(hidden, n_out))


class Head:
    """Trained head over frozen EfficientAT features (tools/train_sound_head.py)."""

    def __init__(self, path: Path = HEAD):
        import torch

        if not path.exists():
            raise FileNotFoundError(f"{path}: run tools/train_sound_head.py")
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        if tuple(ck["fine"]) != FINE:
            raise ValueError(f"{path} was trained on {ck['fine']}, this code expects {FINE}")
        self.torch = torch
        self.backbone = _backbone()
        self.mean, self.std = np.asarray(ck["mean"]), np.asarray(ck["std"])
        self.net = make_head(len(self.mean), len(FINE), ck["hidden"])
        self.net.load_state_dict(ck["state"])
        self.net.eval()

    def fine(self, audio: np.ndarray, sr: int) -> np.ndarray:
        logits, emb = self.backbone(audio, sr)
        return self.fine_from_features(features(logits, emb))[0]

    def fine_from_features(self, f: np.ndarray) -> np.ndarray:
        torch = self.torch
        z = torch.from_numpy(((f - self.mean) / self.std).astype(np.float32))
        with torch.inference_mode():
            return torch.softmax(self.net(z), dim=1).numpy().astype(np.float64)

    def classify(self, audio: np.ndarray, sr: int) -> tuple[float, float, float, float]:
        return fold(self.fine(audio, sr))


@cache
def zeroshot() -> ZeroShot:
    return ZeroShot()


@cache
def efficientat() -> Head:
    return Head()
