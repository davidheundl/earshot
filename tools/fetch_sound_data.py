"""Real sound for the classifier: download, cut into 1 s windows, split, and stock data/sounds.

    uv run --group ml python tools/fetch_sound_data.py [--drone-shards 1,5,8,12,15,19]

Sources (both downloaded into data/raw/sounds/, gitignored):
    ESC-50          github.com/karolpiczak/ESC-50, CC BY-NC 3.0: airplane, helicopter, engine,
                    train, vacuum cleaner, insects, wind ... (2000 clips of 5 s, 5 folds)
    DroneAudioSet   huggingface.co/datasets/ahlab-drone-project/DroneAudioSet, MIT: drone-only
                    recordings, a few of its 28 shards (~100 MB each). Only two drones: shards
                    1-14 are drone2, 15-28 drone1 (each at 25/50 cm, low/high throttle)

Writes:
    data/raw/sounds/windows.npz     1 s windows at 16 kHz (int16, each at its own peak scale),
                                    fine label, group (source recording), split (train/test)
    data/sounds/<class>/*.wav       held-out test recordings for the simulator's ClipBank, so
                                    simulated beams are real sound the head never trained on

Split by source recording, never by window: ESC-50 fold 5 and all of drone1 are test, so the
drone test is a drone the head never heard. Silent stretches (ESC-50 pads short sounds) are dropped.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
RAW = Path("data/raw/sounds")
BANK = Path("data/sounds")
ESC_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
DRONE_URL = ("https://huggingface.co/datasets/ahlab-drone-project/DroneAudioSet/resolve/main/"
             "drone-only/train_{:03d}-00000-of-00001.parquet")

ESC_FINE = {
    "airplane": "plane", "helicopter": "helicopter",
    "engine": "vehicle", "car_horn": "vehicle", "siren": "vehicle",
    "train": "train",
    "vacuum_cleaner": "machine", "washing_machine": "machine", "chainsaw": "machine",
    "hand_saw": "machine",
    "insects": "insect",
    "wind": "ambient", "rain": "ambient", "sea_waves": "ambient", "thunderstorm": "ambient",
    "crackling_fire": "ambient", "water_drops": "ambient", "pouring_water": "ambient",
}  # everything else: "other"
BANK_CLASS = {"drone": "drone", "vehicle": "vehicle", "train": "train", "plane": "nothing",
              "helicopter": "nothing", "machine": "nothing", "insect": "nothing"}
DRONE_WINDOWS = 240  # per recording, evenly spread
OTHER_WINDOWS = 1  # per ESC-50 clip outside the classes we care about
QUIET_DB = -20.0  # windows this far below the clip's loudest are dropped


def fetch(url: str, out: Path) -> Path:
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        tmp = out.with_suffix(out.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(out)
    return out


def to16k(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, np.float64)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x - x.mean()


def windows(x: np.ndarray, keep: int | None = None) -> list[np.ndarray]:
    """Non-overlapping 1 s windows, quiet ones dropped, at most `keep` (evenly spread)."""
    w = [x[i:i + SR] for i in range(0, len(x) - SR + 1, SR)]
    if not w:
        return []
    rms = np.array([math.sqrt(float(np.mean(v * v))) for v in w])
    if rms.max() <= 1e-6:
        return []
    loud = [v for v, r in zip(w, rms, strict=True)
            if r > 1e-6 and 20 * math.log10(r / rms.max()) > QUIET_DB]
    if keep is not None and len(loud) > keep:
        if keep == 1:
            return [w[int(np.argmax(rms))]]
        loud = [loud[int(i)] for i in np.linspace(0, len(loud) - 1, keep)]
    return loud


def esc50(rows: list) -> list[tuple[np.ndarray, str, str, str]]:
    """(audio 16 kHz, fine, group, split) per clip."""
    z = zipfile.ZipFile(fetch(ESC_URL, RAW / "ESC-50-master.zip"))
    meta = list(csv.DictReader(io.TextIOWrapper(z.open("ESC-50-master/meta/esc50.csv"))))
    out = []
    for m in meta:
        x, sr = sf.read(io.BytesIO(z.read(f"ESC-50-master/audio/{m['filename']}")))
        fine = ESC_FINE.get(m["category"], "other")
        split = "test" if m["fold"] == "5" else "train"
        out.append((to16k(x, sr), fine, f"esc50/{m['src_file']}", split))
    rows += out
    return out


def droneaudioset(rows: list, shards: list[int]) -> list[tuple[np.ndarray, str, str, str]]:
    import pyarrow.parquet as pq

    out = []
    for i in shards:
        path = fetch(DRONE_URL.format(i), RAW / "droneaudioset" / f"drone-only_{i:03d}.parquet")
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=1, columns=["file_path", "audio"]):
            rec = batch.to_pylist()[0]
            a = rec["audio"]
            arr = np.asarray(a["array"], np.float64)
            ch0 = arr[0] if arr.ndim == 2 and arr.shape[0] < arr.shape[1] else arr[:, 0]
            split = "test" if "/drone1-only/" in rec["file_path"] else "train"
            out.append((to16k(ch0, a["sampling_rate"]), "drone", f"drone/{rec['file_path']}",
                        split))
    rows += out
    return out


def stock_bank(recs: list[tuple[np.ndarray, str, str, str]]) -> dict[str, int]:
    """Held-out test recordings -> data/sounds/<class>/ (quiet stretches cut out)."""
    count: dict[str, int] = {}
    for k, (x, fine, group, split) in enumerate(recs):
        cls = BANK_CLASS.get(fine)
        if split != "test" or cls is None:
            continue
        w = windows(x, 30)
        if len(w) < 2:
            continue
        d = BANK / cls
        d.mkdir(parents=True, exist_ok=True)
        y = np.concatenate(w)
        name = group.replace("/", "_").replace(".", "_")[-60:]
        sf.write(d / f"{fine}-{k:04d}-{name}.wav", (y / np.abs(y).max() * 0.9).astype(np.float32), SR,
                 subtype="PCM_16")
        count[cls] = count.get(cls, 0) + 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--drone-shards", default="1,5,8,12,15,19",
                    help="DroneAudioSet drone-only shards (1-14 drone2, 15-28 drone1)")
    ap.add_argument("--no-bank", action="store_true", help="leave data/sounds alone")
    a = ap.parse_args()

    recs: list = []
    esc50(recs)
    droneaudioset(recs, [int(i) for i in a.drone_shards.split(",")])

    xs, fines, groups, splits = [], [], [], []
    for x, fine, group, split in recs:
        keep = DRONE_WINDOWS if fine == "drone" else OTHER_WINDOWS if fine == "other" else None
        for w in windows(x, keep):
            xs.append((w / max(np.abs(w).max(), 1e-9) * 32767).astype(np.int16))
            fines.append(fine)
            groups.append(group)
            splits.append(split)
    out = RAW / "windows.npz"
    np.savez_compressed(out, audio=np.stack(xs), fine=np.array(fines), group=np.array(groups),
                        split=np.array(splits))
    print(f"{out}: {len(xs)} windows")
    for f in sorted(set(fines)):
        n = [sum(1 for g, s in zip(fines, splits, strict=True) if g == f and s == sp)
             for sp in ("train", "test")]
        print(f"  {f:11s} train {n[0]:5d}  test {n[1]:5d}")
    if not a.no_bank:
        print("data/sounds:", stock_bank(recs))


if __name__ == "__main__":
    main()
