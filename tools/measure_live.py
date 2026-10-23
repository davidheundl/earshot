"""Probe a running `earshot live` from outside: subscribes to the world's PUB socket.

    uv run python tools/measure_live.py --seconds 60 --skip 10 [--views] --label "webots, HUD"

Reports tick arrival jitter, sim-vs-wall real-time factor, how often a Verifier camera was on,
view-camera frame rates and JPEG sizes, GPU utilisation (ioreg, macOS) and CPU per component
(ps). Read-only, except --views: it sends the game screen's /view keepalive every 2 s, as an
open HUD does, so the four view cameras render without a browser.
"""

import argparse
import itertools
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.request

import msgpack
import zmq

from earshot import config

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--skip", type=float, default=0, help="seconds to wait before measuring")
ap.add_argument("--label", default="")
ap.add_argument("--views", action="store_true", help="keep the view cameras on (/view keepalive)")
a = ap.parse_args()


gpu, cpu = [], []
stop = False


def keepalive():
    url = f"http://{config.HOST}:{config.DASHBOARD_PORT}/view"
    while not stop:
        req = urllib.request.Request(url, method="POST",
                                     data=json.dumps({"views": None}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=1).read()
        except OSError:
            pass
        time.sleep(2.0)


if a.views:
    threading.Thread(target=keepalive, daemon=True).start()
    time.sleep(3)  # the cameras switch on at the next tick
time.sleep(a.skip)
# connect only now: a socket connected during the skip would queue its messages and read
# them back as a burst
ctx = zmq.Context()
sub = ctx.socket(zmq.SUB)
sub.setsockopt(zmq.LINGER, 0)
sub.connect(config.WORLD_PUB)
for t in (b"state.", b"cams."):
    sub.setsockopt(zmq.SUBSCRIBE, t)


def sampler():
    while not stop:
        out = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
                             check=False, capture_output=True, text=True).stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        if m:
            gpu.append(int(m.group(1)))
        ps = subprocess.run(["ps", "-Ao", "pcpu,args"], check=False, capture_output=True, text=True).stdout
        c = {}
        for line in ps.splitlines()[1:]:
            p, _, args = line.strip().partition(" ")
            name = ("webots" if "webots-bin" in args or "MacOS/webots" in args else
                    "agents" if "agent_proc" in args else "world" if "world_proc" in args else
                    "radio" if "radio_proc" in args else "dashboard" if "dashboard.server" in args
                    else None)
            if name:
                c[name] = c.get(name, 0.0) + float(p)
        cpu.append(c)
        time.sleep(1.0)


th = threading.Thread(target=sampler, daemon=True)
th.start()

ticks, cams = [], []  # (wall, sim t), (wall, frame keys, jpeg bytes)
t_end = time.monotonic() + a.seconds
while time.monotonic() < t_end:
    if not sub.poll(200):
        continue
    topic, payload = sub.recv_multipart()
    now = time.monotonic()
    msg = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    if topic == b"state.":
        ticks.append((now, msg["t"], any(d.get("mode") == "VERIFY" for d in msg["drones"])))
    else:
        fr = msg["frames"]
        cams.append((now, sorted(int(k) for k in fr), sum(len(f["jpeg"]) for f in fr.values())))
stop = True
th.join()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


print(f"== {a.label} ({a.seconds:.0f} s)")
if len(ticks) > 2:
    gaps = [(b[0] - a_[0]) * 1000 for a_, b in itertools.pairwise(ticks)]
    wall = ticks[-1][0] - ticks[0][0]
    sim = ticks[-1][1] - ticks[0][1]
    print(f"ticks {len(ticks)}  real-time factor {sim / wall:.3f}  "
          f"gap ms p50 {pct(gaps, .5):.0f} p95 {pct(gaps, .95):.0f} p99 {pct(gaps, .99):.0f} "
          f"max {max(gaps):.0f}  gaps>150ms {sum(g > 150 for g in gaps)}  "
          f"verifying {sum(v for *_, v in ticks) / len(ticks):.0%} of ticks")
else:
    print("no ticks: is a live run up?")
if cams:
    wall = cams[-1][0] - cams[0][0] or 1
    main = sum(0 in k for _, k, _ in cams) / wall
    tile = sum(1 in k for _, k, _ in cams) / wall
    print(f"cams: main {main:.1f} Hz, tile {tile:.1f} Hz, "
          f"{statistics.mean(b for *_, b in cams) / 1024:.0f} KiB/msg")
else:
    print("cams: none (no screen asking for views, or not Webots)")
if gpu:
    print(f"GPU util % mean {statistics.mean(gpu):.0f} max {max(gpu)}")
if cpu:
    keys = sorted({k for c in cpu for k in c})
    print("CPU % mean: " + "  ".join(f"{k} {statistics.mean(c.get(k, 0) for c in cpu):.0f}"
                                      for k in keys))
