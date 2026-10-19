"""How well the sound classifiers tell a drone from planes, trains, cars and noise.

    uv run --group ml python tools/eval_sound_classifier.py [--beams runs/audio/<run>]
        [--out docs/sound-classifier-results.json]

(a) Held-out real windows (ESC-50 fold 5, drone1, which no head saw), clean and as beam-like
    mixtures at fixed SNRs (tools/sound_data.noisy).
(b) With --beams: beams the simulator exported (`earshot fast --audio DIR --audio-beams`,
    data/sounds stocked by tools/fetch_sound_data.py), binned by their true SNR.

For each of zero-shot and the trained head: coarse accuracy, drone recall (a drone called a
drone) and false-drone rate per fine class (a plane, fan or truck called a drone), the two
errors that matter to the swarm.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import sound_data as sd

from earshot.audio import classifier as clf
from earshot.messages import TargetClass
from earshot.sensors import waveform as wf

SNRS = [-10.0, -5.0, 0.0, 5.0, 10.0, 20.0]
D = int(TargetClass.DRONE)


def coarse_probs(fine: np.ndarray) -> np.ndarray:
    return np.array([clf.fold(p) for p in fine])


def scores(y_coarse: np.ndarray, p: np.ndarray, y_fine: np.ndarray | None = None) -> dict:
    pred = p.argmax(1)
    out = {"n": len(pred), "acc": round(float(np.mean(pred == y_coarse)), 3)}
    drone = y_coarse == D
    if drone.any():
        out["drone_recall"] = round(float(np.mean(pred[drone] == D)), 3)
        out["p_drone_on_drone"] = round(float(np.mean(p[drone, D])), 3)
    if (~drone).any():
        out["false_drone"] = round(float(np.mean(pred[~drone] == D)), 3)
    if y_fine is not None:
        out["false_drone_by"] = {clf.FINE[k]: round(float(np.mean(pred[y_fine == k] == D)), 3)
                                 for k in np.unique(y_fine) if clf.FINE[k] != "drone"}
    return out


def real_windows(models: dict) -> dict:
    d = sd.load()
    idx = np.flatnonzero(d["split"] == "test")
    audio = d["audio"][idx].astype(np.float64) / 32767
    y = np.array([clf.FINE.index(f) for f in d["fine"][idx]])
    yc = np.array([int(clf.FOLD[clf.FINE[k]]) for k in y])
    out: dict = {}
    for snr in [np.inf, *SNRS]:
        rng = np.random.default_rng(int(1000 + (snr if np.isfinite(snr) else 99)))
        x = np.stack([sd.unit(a) if np.isinf(snr) else sd.noisy(a, snr, rng) for a in audio])
        f = sd.features(x.astype(np.float32), f"eval-snr{snr}")
        key = "clean" if np.isinf(snr) else f"{snr:+.0f}dB"
        out[key] = {name: scores(yc, coarse_probs(m.fine_from_features(f)), y)
                    for name, m in models.items()}
    return out


def sim_beams(root: Path, models: dict) -> dict:
    beams, ys, snrs = [], [], []
    for line in (root / "labels.jsonl").read_text().splitlines():
        rec = json.loads(line)
        for src in rec["sources"]:
            name = rec.get("beams", {}).get(src["id"])
            if name is None or src["snr_db"] is None:
                continue
            beams.append(wf.read_wav(root / name))
            ys.append(int(TargetClass[src["cls"]]))
            snrs.append(src["snr_db"])
    x, y, snr = np.stack(beams), np.array(ys), np.array(snrs)
    print(f"{root}: {len(x)} beams, classes {np.bincount(y, minlength=4).tolist()}")
    f = sd.features(x.astype(np.float32), f"beams-{root.name}")
    edges = [-20, -10, -5, 0, 5, 10, 20, 100]
    out: dict = {}
    for lo, hi in itertools.pairwise(edges):
        sel = (snr >= lo) & (snr < hi)
        if sel.sum() < 5:
            continue
        out[f"{lo}..{hi}dB"] = {name: scores(y[sel], coarse_probs(m.fine_from_features(f[sel])))
                                for name, m in models.items()}
    out["all"] = {name: scores(y, coarse_probs(m.fine_from_features(f)))
                  for name, m in models.items()}
    return out


def table(title: str, res: dict) -> None:
    print(f"\n== {title}")
    print(f"{'':12s} {'model':12s} {'n':>5s} {'acc':>6s} {'drone_rec':>9s} {'false_drn':>9s}"
          "  false-drone by class")
    for key, by_model in res.items():
        for name, s in by_model.items():
            by = " ".join(f"{k}={v:.2f}" for k, v in s.get("false_drone_by", {}).items())
            print(f"{key:12s} {name:12s} {s['n']:5d} {s['acc']:6.3f} "
                  f"{s.get('drone_recall', float('nan')):9.3f} "
                  f"{s.get('false_drone', float('nan')):9.3f}  {by}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--beams", type=Path, action="append", default=[],
                    help="an `earshot fast --audio DIR --audio-beams` run directory")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    models = {"zeroshot": clf.ZeroShot()}
    if clf.HEAD.exists():
        models["head"] = clf.Head()
    res = {"real": real_windows(models)}
    table("held-out real windows (ESC-50 fold 5, unseen drone)", res["real"])
    for root in a.beams:
        res[f"sim:{root.name}"] = sim_beams(root, models)
        table(f"simulated beams {root.name} (by true SNR)", res[f"sim:{root.name}"])
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
