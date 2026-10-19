"""Live world process: owns the clock, the backend and the ground truth.

    python -m earshot.runners.world_proc --scenario scenarios/demo.yaml --backend twin2d
    (backend webots: start via `earshot live --backend webots`, which sets the Webots env;
    backend gazebo: via `earshot live --backend gazebo`, which starts the Gazebo bridge)

Publishes "tick.<id>." (Sensed) to each agent and "state." (snapshot) to radio + dashboard;
with --audio also "audio." (each drone's newest 1 s of microphone sound) to the dashboard.
Pulls commands + telemetry from agents (a drone without a fresh command keeps its last one)
and operator events from the dashboard ({"op": ...}, see earshot/scenario.py). Every event
that took effect is appended to runs/live-<time>.events.jsonl, replayable with
`earshot fast --events <file>`.

The scripted Scenario ends after its `duration`; the live world keeps running (Sources end
as scripted, the swarm and operator Events go on) until stopped, or for --duration seconds.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import zmq

from earshot import config
from earshot.city import City
from earshot.runners.wire import command_from_dict, pack, sensed_to_dict, topic, unpack
from earshot.scenario import Scenario
from earshot.sensors import build_perception
from earshot.world import World

SAME_INTRUDER_M = 150.0  # Confirmed points this close count as one intruder (belief's follow radius)
PREVIEW_EVERY_S = 5.0  # the Verifier preview (only /classic shows it): big, so rarely


class VerifyCounts:
    """What the operator screens count, from the Agents' decision events. Every drone reports
    its own events, so a raw tally overstates: a task is dispatched again by the drone that
    outbids the first, and several drones confirm the same intruder. Hence
      dispatch   distinct verify tasks set off (by task id),
      verified   distinct verify tasks resolved, confirmed or dismissed (the "Verifications"),
      confirmed  distinct confirmed intruders (by task id and within SAME_INTRUDER_M),
      dismissed  distinct verify tasks that found nothing."""

    def __init__(self):
        self._tids: dict[str, set] = {"dispatch": set(), "confirmed": set(), "dismissed": set()}
        self._intruders: list[list[float]] = []  # last known position of each confirmed one

    def add(self, ev: dict) -> None:
        kind = ev.get("kind")
        if ev.get("task") != "verify" or kind not in self._tids or "tid" not in ev:
            return
        tid = ev["tid"]
        if tid in self._tids[kind]:
            return
        self._tids[kind].add(tid)
        if kind == "confirmed":
            x, y = float(ev.get("x", 0.0)), float(ev.get("y", 0.0))
            near = next((p for p in self._intruders
                         if (p[0] - x) ** 2 + (p[1] - y) ** 2 <= SAME_INTRUDER_M ** 2), None)
            if near is None:
                self._intruders.append([x, y])
            else:
                near[:] = [x, y]  # it moves; follow it

    def as_dict(self) -> dict:
        return {"dispatch": len(self._tids["dispatch"]),
                "verified": len(self._tids["confirmed"] | self._tids["dismissed"]),
                "confirmed": len(self._intruders), "dismissed": len(self._tids["dismissed"])}


def make_backend(name: str, city: City, scenario: Scenario):
    if name == "twin2d":
        from earshot.backend.twin2d import Kinematics, Twin2D

        return Twin2D(city, scenario.drones, Kinematics.for_scenario(scenario)), True
    if name == "webots":
        from earshot.backend.webots import WebotsBackend

        return WebotsBackend(city, scenario), False  # the backend paces Webots' steps
    if name == "gazebo":
        from earshot.backend.gazebo import GazeboBackend

        return GazeboBackend(city, scenario), False  # the bridge paces Gazebo's steps
    raise ValueError(name)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenarios/demo.yaml")
    ap.add_argument("--backend", default="twin2d", choices=["twin2d", "webots", "gazebo"])
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor (twin2d only)")
    ap.add_argument("--events", help="recorded session to replay (runs/live-*.events.jsonl)")
    ap.add_argument("--duration", type=float,
                    help="stop after this many sim seconds (default: run until stopped; the "
                         "scripted Scenario ends on time and the swarm keeps going)")
    ap.add_argument("--audio", action="store_true", help="render sound as waveforms")
    ap.add_argument("--sounds", help="recordings: <dir>/<class>/*.wav (default: the "
                    "Scenario's audio.sounds, else data/sounds)")
    ap.add_argument("--classifier", metavar="MOD:ATTR",
                    help="trained sound classifier (default: the Scenario's; 'none' for off)")
    ap.add_argument("--camera", choices=["yolo", "truth"],
                    help="Verifier camera (default: the Scenario's; webots only)")
    a = ap.parse_args(argv)

    scenario = Scenario.load(a.scenario)
    if a.duration:
        scenario.duration = a.duration
    if a.events:  # replayed as scripted events, so they are not recorded again
        recorded = [json.loads(line) for line in Path(a.events).read_text().splitlines()]
        replay = [{k: v for k, v in e.items() if k not in ("live", "note")} for e in recorded]
        scenario.events = sorted(scenario.events + replay, key=lambda e: e["t"])
    city = City.load(Path(scenario.city))
    backend, self_paced = make_backend(a.backend, city, scenario)
    audio, camera = build_perception(scenario, classifier=a.classifier, sounds=a.sounds,
                                     audio=a.audio, camera=a.camera,
                                     renders=a.backend == "webots",
                                     log=lambda m: print(m, flush=True))
    if audio is not None and audio.classifier is not None:
        print(f"[world] sound: {a.classifier or scenario.audio['classifier']}", flush=True)
    if camera is not None:
        print(f"[world] camera: YOLO {Path(camera.weights).name} on {camera.device}", flush=True)
    world = World(scenario, city, backend, audio, camera)

    ctx = zmq.Context.instance()
    ctx.setsockopt(zmq.LINGER, 0)  # never block exit on unsent messages
    pub = ctx.socket(zmq.PUB)
    pub.bind(config.WORLD_PUB)
    pull = ctx.socket(zmq.PULL)
    pull.bind(config.WORLD_PULL)
    time.sleep(0.3)  # let subscribers connect

    commands: dict = {}
    telemetry: dict = {}
    decisions: list[dict] = []
    counts = VerifyCounts()
    events_path = Path("runs") / time.strftime("live-%Y%m%d-%H%M%S.events.jsonl")
    events_file = None  # created on the first live event, so idle runs leave no file
    logged = 0

    def _stop(*_):
        raise SystemExit

    signal.signal(signal.SIGTERM, _stop)  # the launcher stops us with SIGTERM
    period = scenario.dt / a.speed
    next_t = time.monotonic()
    end = a.duration or None  # live: run until stopped, unless a length was asked for
    scripted = True  # the Scenario's own script still runs
    print(f"[world] {a.backend} running {scenario.name}, dt={scenario.dt}s", flush=True)
    try:
        while end is None or world.t < end:
            if scripted and end is None and world.t >= scenario.duration:
                scripted = False
                print(f"[world] the scripted Scenario ended at {scenario.duration:.0f} s; the "
                      "swarm keeps running until stopped", flush=True)
            while True:
                try:
                    msg = unpack(pull.recv(zmq.NOBLOCK))
                except zmq.Again:
                    break
                try:
                    if not isinstance(msg, dict):
                        raise TypeError(f"not a dict: {type(msg).__name__}")
                    if msg.get("op") == "view":  # game screen camera assignment, not an Event
                        if hasattr(backend, "set_views"):
                            backend.set_views(msg.get("views"))
                        continue
                    if "op" in msg:
                        refused = getattr(backend, "refuses", lambda _op: None)(msg)
                        if refused:  # e.g. Webots has no free node to show the Source;
                            # logged below like any Event, but not recorded (no "live" tag)
                            world.applied.append({**msg, "t": round(world.t, 3),
                                                  "note": f"refused {msg['op']}: {refused}"})
                            continue
                        world.control(msg)
                        continue
                    commands[msg["id"]] = command_from_dict(msg["command"])
                    telemetry[msg["id"]] = msg["telemetry"]
                    for d in msg.get("events", []):
                        decisions.append(d)
                        counts.add(d)
                    del decisions[:-50]  # the screens show the last 12
                except Exception as e:  # noqa: BLE001 - a bad message must not stop the world
                    print(f"[world] ignored message: {type(e).__name__}: {e}", file=sys.stderr,
                          flush=True)
            sensed = world.step(commands)
            for ev in world.applied[logged:]:
                if ev.get("live"):  # scripted and replayed events are not recorded again
                    if events_file is None:
                        events_path.parent.mkdir(exist_ok=True)
                        events_file = open(events_path, "a")  # noqa: SIM115 - process lifetime
                    events_file.write(json.dumps(ev) + "\n")
                    events_file.flush()
                print(f"[world] t={ev['t']:.1f} {ev['note']}", flush=True)
            logged = len(world.applied)
            for did, s in sensed.items():
                pub.send_multipart([topic("tick", did), pack(sensed_to_dict(s))])
            if world.audio:  # float32, RMS 1.0 = waveform.FULL_SCALE_DB
                pub.send_multipart([topic("audio"), pack({"t": world.t, "windows": {
                    str(w.drone): w.mix.astype("<f4").tobytes() for w in world.audio}})])
            snap = world.snapshot(telemetry)
            if round(world.t / scenario.dt) % round(PREVIEW_EVERY_S / scenario.dt) == 0:
                snap["camera"] = world.camera_preview()
            snap["decisions"] = decisions[-12:]
            snap["counts"] = counts.as_dict()
            pub.send_multipart([topic("state"), pack(snap)])
            if hasattr(backend, "view_frames"):  # main view every tick (10 Hz), tiles every 2nd
                tick = round(world.t / scenario.dt)
                frames = backend.view_frames([0] + ([1, 2, 3] if tick % 2 == 0 else []))
                if frames:
                    pub.send_multipart([topic("cams"), pack({"t": world.t, "frames": frames})])
            if self_paced:
                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
        print(f"[world] finished after {end:.0f} s (--duration)", flush=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        backend.close()
        if events_file is not None:
            events_file.close()
            print(f"[world] recorded live events to {events_path}", flush=True)
        print("[world] stopped", flush=True)


if __name__ == "__main__":
    main()
