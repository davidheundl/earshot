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
    a = ap.parse_args(argv)

    scenario = Scenario.load(a.scenario)
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
    Path("runs").mkdir(exist_ok=True)
    events_path = Path("runs") / time.strftime("live-%Y%m%d-%H%M%S.events.jsonl")
    events_file = open(events_path, "a")  # noqa: SIM115 - lives as long as the process
    logged = 0
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
            sensed = world.step(commands)
            for ev in world.applied[logged:]:
                if ev.get("live"):  # scripted YAML events replay from the YAML itself
                    events_file.write(json.dumps(ev) + "\n")
                    events_file.flush()
                print(f"[world] t={ev['t']:.1f} {ev['note']}", flush=True)
            logged = len(world.applied)
            for did, s in sensed.items():
                pub.send_multipart([topic("tick", did), pack(sensed_to_dict(s))])
            pub.send_multipart([topic("state"), pack(world.snapshot(telemetry))])
            if self_paced:
                next_t += period
                time.sleep(max(0.0, next_t - time.monotonic()))
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        backend.close()
        events_file.close()
        if events_path.stat().st_size == 0:
            events_path.unlink()
        print("[world] stopped", flush=True)


if __name__ == "__main__":
    main()
