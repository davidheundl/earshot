"""Extern supervisor (the RIG): Verifier camera stills of each candidate model on the ground,
per horizontal distance, camera pitch and vehicle yaw. Usage: capture.py <repo> <outdir> [quick]"""
import json
import math
import sys
from pathlib import Path

from controller import Supervisor
from PIL import Image

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "src"))
from earshot.backend.webots import _yaw, look_rotation  # noqa: E402
from earshot.city import City  # noqa: E402

quick = len(sys.argv) > 3 and sys.argv[3] == "quick"
city = City.load(repo / "data/cities/savanna.json")
(out / "images").mkdir(parents=True, exist_ok=True)
sup = Supervisor()
cam = sup.getDevice("camera")
cam.enable(20)
cam.recognitionEnable(20)
rig = sup.getFromDef("CAMV")
w, h = cam.getWidth(), cam.getHeight()

MODELS = {  # name: (DEF, lift above ground as the backend places it)
    "pickup": ("SRC_poacher1", 0.5), "ranger_pickup": ("SRC_ranger1", 0.5),
    "truck": ("TEST_TRUCK", 0.5), "truck_cab": ("TEST_CAB", 0.5),
    "rangerover": ("TEST_RR", 0.4), "rangerover_white": ("TEST_RRW", 0.4),
    "tesla": ("TEST_TESLA", 0.4),
    "rangerover_shadow": ("TEST_RRS", 0.4), "rangerover_white_shadow": ("TEST_RRWS", 0.4),
    "tesla_shadow": ("TEST_TESLAS", 0.4), "pickup_shadow": ("TEST_PKS", 0.5),
    "elephants": ("SRC_elephants", 0.7), "buffalo": ("SRC_buffalo", 0.7),
    "rhino": ("SRC_rhino", 0.7), "lions": ("SRC_lions", 0.7), "empty": (None, 0.0),
}
SPOTS = {"A": (-1950.0, -2600.0, 0.0), "B": (3000.0, 2300.0, 225.0)}  # x, y, camera heading
DISTS = [30, 50, 80, 120, 160, 220, 300]
RELYAW = [0, 45, 90, 135, 180]
FIXED_PITCH = math.degrees(math.atan2(10.0, 30.0))  # today: TARGET_DROP_M over standoff 30
if len(sys.argv) > 3 and sys.argv[3] not in ("quick", ""):
    MODELS = {k: MODELS[k] for k in sys.argv[3].split(",")}
    DISTS = [30, 50, 80, 120, 160]
if quick:
    MODELS = {k: MODELS[k] for k in ("pickup", "truck", "rangerover", "elephants")}
    SPOTS, DISTS, RELYAW = {"A": SPOTS["A"]}, [50, 120], [90]

nodes = {k: sup.getFromDef(d) for k, (d, _) in MODELS.items() if d}
missing = [k for k, n in nodes.items() if n is None]
if missing:
    print("missing nodes:", missing, flush=True)
nodes = {k: n for k, n in nodes.items() if n is not None}
PARK = {k: n.getField("translation").getSFVec3f() for k, n in nodes.items()}


def step(n=3):
    for _ in range(n):
        if sup.step(20) == -1:
            raise SystemExit("Webots quit")


def boxes(defname):
    res = []
    for o in cam.getRecognitionObjects():
        node = sup.getFromId(o.getId())
        while node is not None and not (node.getDef() or "").startswith(("SRC_", "TEST_")):
            node = node.getParentNode()
        if node is None or node.getDef() != defname:
            continue
        x, y = o.getPositionOnImage()
        bw, bh = o.getSizeOnImage()
        res.append([max(0, x - bw / 2), max(0, y - bh / 2), min(w, x + bw / 2), min(h, y + bh / 2)])
    return res


rows = []
for spot, (tx, ty, ch) in SPOTS.items():
    gz = float(city.ground(tx, ty))
    for model, (defname, lift) in MODELS.items():
        if defname and model not in nodes:
            continue
        node = nodes.get(model)
        for d in DISTS:
            cx, cy = tx - d * math.sin(math.radians(ch)), ty - d * math.cos(math.radians(ch))
            cz = float(city.ground(cx, cy)) + 40.0 - 0.3
            aimed = math.degrees(math.atan2(cz - (gz + 1.0), d))
            for pm, pitch in (("fixed", FIXED_PITCH), ("aimed", aimed)):
                for ry in RELYAW if defname else [0]:
                    if node is not None:
                        node.getField("translation").setSFVec3f([tx, ty, gz + lift])
                        node.getField("rotation").setSFRotation([0, 0, 1, _yaw(ch + ry)])
                    rig.getField("translation").setSFVec3f([cx, cy, cz])
                    rig.getField("rotation").setSFRotation(look_rotation(ch, pitch))
                    step()
                    im = Image.frombytes("RGBA", (w, h), bytes(cam.getImage()), "raw", "BGRA").convert("RGB")
                    name = f"images/{spot}_{model}_{d:03d}_{pm}_{ry:03d}.jpg"
                    im.save(out / name, quality=92)
                    rows.append({"image": name, "spot": spot, "model": model, "d": d, "pitch_mode": pm,
                                 "pitch": round(pitch, 1), "relyaw": ry,
                                 "slant": round(math.hypot(d, cz - gz - 1.0), 1),
                                 "gt": boxes(defname) if defname else []})
            print(spot, model, d, flush=True)
        if node is not None:
            node.getField("translation").setSFVec3f(PARK[model])
(out / "labels.json").write_text(json.dumps(rows, indent=1))
print("captured", len(rows), flush=True)
