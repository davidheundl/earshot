"""Extern supervisor: 10 Hz sequences of a Verifier following a target as the twin flies it
(15 m/s towards a look_at that lags the target by `lag` s, holding `standoff`, camera pitch
fixed at atan2(drop, standoff)). Usage: capture_seq.py <repo> <outdir> <spec,...>
spec = name:DEF:lift:speed:standoff:lag  (speed m/s of the target)."""
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

city = City.load(repo / "data/cities/savanna.json")
sup = Supervisor()
cam = sup.getDevice("camera")
cam.enable(20)
rig = sup.getFromDef("CAMV")
w, h = cam.getWidth(), cam.getHeight()
DT, N = 0.1, 300
meta = {}
for spec in sys.argv[3].split(","):
    name, defname, lift, speed, standoff, lag = spec.split(":")
    lift, speed, standoff, lag = float(lift), float(speed), float(standoff), float(lag)
    node = sup.getFromDef(defname)
    park = node.getField("translation").getSFVec3f()
    (out / name).mkdir(parents=True, exist_ok=True)
    x0, y0, hd = -2300.0, -2700.0, 70.0  # open thornveld, heading ENE
    pitch = math.degrees(math.atan2(40.0 - lift, standoff))
    cx, cy = x0 - 140 * math.sin(math.radians(hd + 30)), y0 - 140 * math.cos(math.radians(hd + 30))
    frames = []
    for k in range(N):
        t = k * DT
        tx, ty = x0 + speed * t * math.sin(math.radians(hd)), y0 + speed * t * math.cos(math.radians(hd))
        lt = max(0.0, t - lag)
        lx, ly = x0 + speed * lt * math.sin(math.radians(hd)), y0 + speed * lt * math.cos(math.radians(hd))
        d = math.hypot(lx - cx, ly - cy)
        step = min(max(0.0, d - standoff), 15.0 * DT)
        if d > 1e-6:
            cx, cy = cx + (lx - cx) / d * step, cy + (ly - cy) / d * step
        heading = math.degrees(math.atan2(lx - cx, ly - cy)) % 360.0
        cz = float(city.ground(cx, cy)) + 40.0 - 0.3
        node.getField("translation").setSFVec3f([tx, ty, float(city.ground(tx, ty)) + lift])
        node.getField("rotation").setSFRotation([0, 0, 1, _yaw(hd)])
        rig.getField("translation").setSFVec3f([cx, cy, cz])
        rig.getField("rotation").setSFRotation(look_rotation(heading, pitch))
        for _ in range(5):  # 0.1 s of simulated time
            sup.step(20)
        im = Image.frombytes("RGBA", (w, h), bytes(cam.getImage()), "raw", "BGRA").convert("RGB")
        im.save(out / name / f"{k:03d}.jpg", quality=95)
        frames.append({"t": round(t, 1), "yaw": heading, "pitch": pitch,
                       "range": round(math.hypot(tx - cx, ty - cy), 1)})
    node.getField("translation").setSFVec3f(park)
    meta[name] = {"w": w, "h": h, "fov": cam.getFov(), "frames": frames}
    print(name, flush=True)
(out / "seq.json").write_text(json.dumps(meta))
