"""Live world process: owns the clock, the backend and the ground truth.

    python -m earshot.runners.world_proc --scenario scenarios/demo.yaml --backend twin2d
    (backend webots: start via `earshot live --backend webots`, which sets the Webots env)

Publishes "tick.<id>." (Sensed) to each agent and "state." (snapshot) to radio + dashboard.
Pulls commands + telemetry from agents (a drone without a fresh command keeps its last one)
and operator events from the dashboard ({"op": ...}, see earshot/scenario.py). Every event
that took effect is appended to runs/live-<time>.events.jsonl, replayable with
`earshot fast --events <file>`.
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import zmq

from earshot import config
from earshot.city import City
from earshot.runners.wire import command_from_dict, pack, sensed_to_dict, topic, unpack
from earshot.scenario import Scenario
from earshot.world import World


def make_backend(name: str, city: City, scenario: Scenario):
    if name == "twin2d":
        from earshot.backend.twin2d import Twin2D

        return Twin2D(city, scenario.drones), True
    if name == "webots":
        from earshot.backend.webots import WebotsBackend

        return WebotsBackend(city, scenario), False  # Webots paces itself
    raise ValueError(name)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenarios/demo.yaml")
    ap.add_argument("--backend", default="twin2d", choices=["twin2d", "webots"])
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor (twin2d only)")
    ap.add_argument("--events", help="recorded session to replay (runs/live-*.events.jsonl)")
    ap.add_argument("--duration", type=float, help="override the scenario length (sim seconds)")
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
    world = World(scenario, city, backend)

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
    events_path = Path("runs") / time.strftime("live-%Y%m%d-%H%M%S.events.jsonl")
    events_file = None  # created on the first live event, so idle runs leave no file
    logged = 0

    def _stop(*_):
        raise SystemExit

    signal.signal(signal.SIGTERM, _stop)  # the launcher stops us with SIGTERM
    period = scenario.dt / a.speed
    next_t = time.monotonic()
    print(f"[world] {a.backend} running {scenario.name}, dt={scenario.dt}s", flush=True)
    try:
        while world.t < scenario.duration:
            while True:
                try:
                    msg = unpack(pull.recv(zmq.NOBLOCK))
                except zmq.Again:
                    break
                if "op" in msg:
                    world.control(msg)
                    continue
                commands[msg["id"]] = command_from_dict(msg["command"])
                telemetry[msg["id"]] = msg["telemetry"]
                decisions.extend(msg.get("events", []))
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
            snap = world.snapshot(telemetry)
            if round(world.t / scenario.dt) % round(1.0 / scenario.dt) == 0:
                snap["camera"] = world.camera_preview()  # once per second; frames are big
            snap["decisions"] = decisions[-12:]
            snap["counts"] = {k: sum(1 for d in decisions if d["kind"] == k
                                     and d.get("task") == "verify")
                              for k in ("dispatch", "confirmed", "dismissed")}
            pub.send_multipart([topic("state"), pack(snap)])
            if self_paced:
                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
        print(f"[world] scenario finished after {scenario.duration:.0f} s "
              "(use --duration to run longer)", flush=True)
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
