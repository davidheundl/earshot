"""YOLO (COCO yolo11n.pt) on the captured stills: per frame the boxes on the target (centre
inside the recognition box grown by 30 %) and any vehicle-class boxes elsewhere.
Usage: evaluate.py <repo> <outdir> [imgsz]"""
import json
import sys
from pathlib import Path

from ultralytics import YOLO

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 1280
rows = json.loads((out / "labels.json").read_text())
model = YOLO(str(repo / "yolo11n.pt"))
names = {int(k): v for k, v in model.names.items()}
VEH = {"car", "truck", "bus", "motorcycle"}


def on_target(b, gts):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    for g in gts:
        mx, my = 0.3 * (g[2] - g[0]) + 4, 0.3 * (g[3] - g[1]) + 4
        if g[0] - mx <= cx <= g[2] + mx and g[1] - my <= cy <= g[3] + my:
            return True
    return False


res_rows = []
for r in rows:
    res = model.predict(str(out / r["image"]), imgsz=imgsz, conf=0.05, verbose=False, device="mps")[0]
    on, off = [], []
    for box in res.boxes:
        b = [float(v) for v in box.xyxy[0].tolist()]
        item = (names[int(box.cls[0])], round(float(box.conf[0]), 3))
        (on if on_target(b, r["gt"]) else off).append(item)
    on.sort(key=lambda x: -x[1])
    res_rows.append({**r, "on": on, "off_vehicle": [o for o in off if o[0] in VEH and o[1] >= 0.25],
                     "off": sorted(off, key=lambda x: -x[1])[:5]})
(out / f"yolo_{imgsz}.json").write_text(json.dumps(res_rows, indent=1))
for r in res_rows:
    print(r["image"].split("/")[-1], "inframe" if r["gt"] else "OUT", r["on"][:3], r["off_vehicle"][:2])
