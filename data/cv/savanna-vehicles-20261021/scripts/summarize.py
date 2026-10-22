"""Tables from yolo_<imgsz>.json: per model x distance x pitch mode, how many frames (of the
yaws and spots) have a vehicle label on the target at >= thr, and the top labels."""
import collections
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
imgsz = sys.argv[2] if len(sys.argv) > 2 else "1280"
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
rows = json.loads((out / f"yolo_{imgsz}.json").read_text())  # imgsz may carry a suffix: 1280_yolo11s
VEH = {"car", "truck", "bus", "motorcycle"}
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["model"], r["pitch_mode"], r["d"])].append(r)
print(f"imgsz {imgsz}, vehicle label on target >= {thr}: hits/frames in frame (top labels)")
for (m, pm, d), g in sorted(groups.items()):
    inf = [r for r in g if r["gt"]]
    hits = [r for r in inf if any(lab in VEH and c >= thr for lab, c in r["on"])]
    top = collections.Counter(r["on"][0][0] for r in inf if r["on"])
    best = max((c for r in inf for lab, c in r["on"] if lab in VEH), default=0)
    fp = sum(1 for r in g if r["off_vehicle"])
    print(f"{m:17s} {pm:5s} {d:4d}m  {len(hits):2d}/{len(inf):2d}/{len(g):2d}  best veh {best:.2f}  "
          f"top {dict(top.most_common(3))}  offFP {fp}")
