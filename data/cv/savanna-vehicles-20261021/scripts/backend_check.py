"""End-to-end check of WebotsBackend on the savanna world (private port): a Verifier flies to a
point with its camera on; read its frames with the stand-in and with detector_frames."""
import math
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / "src"))
from earshot.backend.webots import WebotsBackend  # noqa: E402
from earshot.city import City  # noqa: E402
from earshot.messages import Mode  # noqa: E402
from earshot.scenario import Scenario  # noqa: E402
from earshot.state import Command  # noqa: E402

sc = Scenario.load(repo / "scenarios/savanna.yaml")
city = City.load(repo / sc.city)
b = WebotsBackend(city, sc)
target = (-2400.0, -3250.0)  # ~120 m from drone 1's tower
cmd = {1: Command(Mode.VERIFY, goal=target, camera_on=True, look_at=target)}
for _ in range(80):
    states = b.step(0.1, cmd)
s = states[1]
f = b.camera_frame(1)
print("stand-in: drone 1 at", round(s.x), round(s.y), "dist", round(math.hypot(target[0] - s.x, target[1] - s.y)),
      "frame keys", sorted(f), "pitch deg", round(b._verify_pitch(), 1))
b.detector_frames = True
for _ in range(5):
    b.step(0.1, cmd)
f = b.camera_frame(1)
print("detector: pitch deg", round(math.degrees(f["pitch"]), 1), "height", round(f["height"], 1),
      "bgra bytes", len(f["bgra"]), "w x h", f["w"], f["h"], "yaw", round(f["yaw"]))
