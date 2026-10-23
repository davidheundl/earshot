"""EfficientAT MobileNet (mn10_as), vendored from github.com/fschmid56/EfficientAT (MIT, see
LICENSE here). Only the model code is vendored, with imports made relative and torchvision's
deprecated ConvNormActivation replaced by Conv2dNormActivation; the weights come from the
repo's v0.0.1 release (tools/fetch_efficientat.py -> data/models/).

    net = load("data/models/mn10_as_mAP_471.pt")      # eval mode, CPU
    mel = MelFrontend()(wave_32k)                     # (batch, 1, 128, frames)
    logits, embedding = net(mel)                      # (batch, 527), (batch, 960)

The frontend is EfficientAT's AugmentMelSTFT in eval mode (no masking, fixed fmin/fmax),
rewritten without torchaudio: the mel bank is torchaudio.compliance.kaldi.get_mel_banks'
formula, computed once in numpy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import get_model

SR = 32000  # the AudioSet models' rate
N_MELS, N_FFT, WIN, HOP = 128, 1024, 800, 320
FMIN, FMAX = 0.0, SR // 2 - 1000  # AugmentMelSTFT's eval-mode defaults


def _mel(f: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log(1.0 + f / 700.0)


def kaldi_mel_banks(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SR,
                    fmin: float = FMIN, fmax: float = FMAX) -> np.ndarray:
    """(n_mels, n_fft // 2 + 1) triangular filters, as torchaudio's kaldi get_mel_banks plus
    the zero column EfficientAT pads for the Nyquist bin."""
    n_bins = n_fft // 2
    fft_mel = _mel(sr / n_fft * np.arange(n_bins))
    lo, hi = _mel(np.float64(fmin)), _mel(np.float64(fmax))
    delta = (hi - lo) / (n_mels + 1)
    left = lo + np.arange(n_mels)[:, None] * delta
    center, right = left + delta, left + 2 * delta
    up = (fft_mel[None] - left) / (center - left)
    down = (right - fft_mel[None]) / (right - center)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1))).astype(np.float32)


class MelFrontend(nn.Module):
    """Waveform at 32 kHz (batch, samples) -> normalised log-mel (batch, 1, 128, frames)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN, periodic=False), persistent=False)
        self.register_buffer("banks", torch.from_numpy(kaldi_mel_banks()), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 1:] - 0.97 * x[:, :-1]  # preemphasis
        spec = torch.stft(x, N_FFT, hop_length=HOP, win_length=WIN, window=self.window,
                          center=True, return_complex=True).abs() ** 2
        mel = torch.log(self.banks @ spec + 1e-5)
        return ((mel + 4.5) / 5.0).unsqueeze(1)


def load(weights: str | Path) -> nn.Module:
    """mn10_as with its AudioSet head, in eval mode on the CPU."""
    net = get_model(num_classes=527, pretrained_name=None, width_mult=1.0, head_type="mlp")
    net.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    return net.eval()

