"""The live detector (earshot.sensors.camera.YoloDetector) on the rendered sequences: per
1 s window its verified hits. Usage: eval_seq.py <repo> <seqdir> conf verify_conf every [names]"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

repo, seq = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "src"))
from earshot.sensors.camera import YoloDetector  # noqa: E402

conf, vconf, every = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
meta = json.loads((seq / "seq.json").read_text())
names = sys.argv[6].split(",") if len(sys.argv) > 6 else list(meta)
CLASSES = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle", "motorcycle": "vehicle",
           "elephant": "nothing", "zebra": "nothing", "giraffe": "nothing", "cow": "nothing",
           "horse": "nothing", "sheep": "nothing", "dog": "nothing", "cat": "nothing",
           "bear": "nothing", "bird": "nothing"}
det = YoloDetector(weights=os.environ.get("WEIGHTS") or repo / "yolo11n.pt", classes=CLASSES, conf=conf, verify_conf=vconf,
                   every=every, confirm_frames=3)
for name in names:
    m = meta[name]
    det.reset()
    wins = []
    for k, f in enumerate(m["frames"]):
        rgb = np.asarray(Image.open(seq / name / f"{k:03d}.jpg").convert("RGB"))
        bgra = np.dstack([rgb[..., ::-1], np.full(rgb.shape[:2], 255, np.uint8)]).tobytes()
        det.observe({"bgra": bgra, "w": m["w"], "h": m["h"], "fov": m["fov"], "yaw": f["yaw"],
                     "pitch": math.radians(f["pitch"]), "height": 39.7})
        if k % 10 == 9:
            v = det.verdict(hovering=False)
            wins.append((f["t"], f["range"], [(round(b), round(c, 2), lab, round(r)) for b, _, c, lab, r in v.hits],
                         det._streaks.copy(), v.others))
    hit_t = [t for t, _, hits, _, _ in wins if hits]
    print(f"{name}: windows with a verified vehicle {len(hit_t)}/{len(wins)}, first at "
          f"{hit_t[0] if hit_t else None} s; ranges {wins[0][1]}..{wins[-1][1]} m; "
          f"median inference {sorted(det.infer_ms)[len(det.infer_ms) // 2]:.0f} ms")
    for t, r, hits, streaks, others in wins:
        print(f"   t {t:5.1f} range {r:6.1f}  hits {hits}  others {others}")
