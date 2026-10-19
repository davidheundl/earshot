"""Train the sound head: frozen EfficientAT features -> fine classes (classifier.FINE).

    uv run --group ml python tools/fetch_efficientat.py
    uv run --group ml python tools/fetch_sound_data.py
    uv run --group ml python tools/train_sound_head.py [--noisy 2] [--epochs 80]

Train windows are used clean and as `--noisy` beam-like copies (tools/sound_data.noisy);
features are cached, so re-training the head takes seconds. Classes are weighted by inverse
frequency (there are far more drone windows than plane windows). Seeded: the same data gives
the same head. Writes data/models/sound_head.pt and prints the test confusion (fold 5 of
ESC-50 and held-out drone recordings).
"""

from __future__ import annotations

import argparse

import numpy as np
import sound_data as sd
import torch

from earshot.audio import classifier as clf
from earshot.messages import TargetClass


def confusion(y: np.ndarray, p: np.ndarray, names) -> str:
    k = len(names)
    m = np.zeros((k, k), int)
    for t, q in zip(y, p, strict=True):
        m[t, q] += 1
    w = max(len(n) for n in names)
    lines = [" " * (w + 2) + " ".join(f"{n[:6]:>6}" for n in names) + "  recall"]
    for i, n in enumerate(names):
        r = m[i, i] / m[i].sum() if m[i].sum() else float("nan")
        lines.append(f"{n:>{w}}  " + " ".join(f"{v:6d}" for v in m[i]) + f"  {r:6.2f}")
    return "\n".join(lines)


def coarse(fine_idx: np.ndarray) -> np.ndarray:
    return np.array([int(clf.FOLD[clf.FINE[i]]) for i in fine_idx])


def report(label: str, y: np.ndarray, probs: np.ndarray) -> float:
    """Fine and coarse confusion; returns coarse accuracy."""
    pf = probs.argmax(1)
    pc = np.array([int(np.argmax(clf.fold(p))) for p in probs])
    yc = coarse(y)
    acc = float(np.mean(pc == yc))
    print(f"\n== {label}: fine acc {np.mean(pf == y):.3f}, coarse acc {acc:.3f}")
    print(confusion(y, pf, clf.FINE))
    print(confusion(yc, pc, [c.name.lower() for c in TargetClass]))
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--noisy", type=int, default=2, help="noisy copies per train window")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = sd.load()
    xtr, ytr, _ = sd.variants(d, "train", a.seed, a.noisy)
    xte, yte, snr = sd.variants(d, "test", a.seed + 1, 1)
    print(f"train {len(xtr)} windows, test {len(xte)}")
    ftr = sd.features(xtr, "train")
    fte = sd.features(xte, "test")

    torch.manual_seed(a.seed)
    mean, std = ftr.mean(0), ftr.std(0) + 1e-6
    z = torch.from_numpy(((ftr - mean) / std).astype(np.float32))
    y = torch.from_numpy(ytr)
    counts = np.bincount(ytr, minlength=len(clf.FINE)).astype(np.float64)
    weight = torch.tensor(np.where(counts > 0, counts.sum() / np.maximum(counts, 1) /
                                   len(clf.FINE), 0.0), dtype=torch.float32)
    net = clf.make_head(z.shape[1], len(clf.FINE), a.hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    gen = torch.Generator().manual_seed(a.seed)
    for epoch in range(a.epochs):
        net.train()
        perm = torch.randperm(len(z), generator=gen)
        total = 0.0
        for k in range(0, len(z), 256):
            b = perm[k:k + 256]
            opt.zero_grad()
            loss = loss_fn(net(z[b]), y[b])
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        if epoch % 20 == 0 or epoch == a.epochs - 1:
            print(f"epoch {epoch:3d} loss {total / len(z):.3f}")
    net.eval()

    ck = {"fine": list(clf.FINE), "mean": mean, "std": std, "hidden": a.hidden,
          "state": net.state_dict(), "args": vars(a)}
    clf.HEAD.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, clf.HEAD)
    print(f"wrote {clf.HEAD}")

    head = clf.Head()
    probs = head.fine_from_features(fte)
    report("test, clean", yte[np.isinf(snr)], probs[np.isinf(snr)])
    report("test, noisy (-10..20 dB SNR)", yte[~np.isinf(snr)], probs[~np.isinf(snr)])


if __name__ == "__main__":
    main()
