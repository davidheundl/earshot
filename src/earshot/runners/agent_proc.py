"""Live agent process: one per drone. A thin shell around the pure Agent.

    python -m earshot.runners.agent_proc --id 3 --scenario scenarios/demo.yaml

Talks to nobody but the world (sensing, commands) and the radio (frames).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import zmq

from earshot import config
from earshot.agent import Agent
from earshot.city import City
from earshot.messages import Sensor
from earshot.runners.wire import command_to_dict, pack, sensed_from_dict, topic, unpack
from earshot.scenario import Scenario
from earshot.state import PolicyParams, Sensed, SwarmConfig


def merge_ticks(ticks: list[Sensed]) -> Sensed:
    """The newest tick's state and camera frame, with the readings of all ticks (oldest
    first). Sensors report once per second per drone, so a dropped tick would lose them."""
    newest = ticks[-1]
    if len(ticks) == 1:
        return newest
    return Sensed(newest.state, [r for s in ticks for r in s.readings], newest.camera_frame)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--scenario", default="scenarios/demo.yaml")
    ap.add_argument("--array", help="sound from this microphone array (device name or index), "
                    "optionally @rotation_deg, instead of the simulated sound sensor")
    ap.add_argument("--classifier", metavar="MOD:ATTR",
                    help="with --array, a trained sound classifier decides the class")
    ap.add_argument("--params", help="PolicyParams JSON over the Scenario's (earshot tune)")
    a = ap.parse_args(argv)
    array = None
    if a.array:
        from earshot.audio.array import ArraySensor

        dev, _, rot = a.array.partition("@")
        clf = None
        if a.classifier:
            from earshot.sensors.waveform import load_classifier

            clf = load_classifier(a.classifier)
        array = ArraySensor(int(dev) if dev.isdigit() else dev, float(rot or 0.0),
                            classifier=clf)
        print(f"[agent {a.id}] sound from array {dev!r}, rotation {rot or 0} deg", flush=True)
    scenario = Scenario.load(a.scenario)
    params = dict(scenario.params)
    if a.params:  # earshot live --params FILE: over the Scenario's own
        d = json.loads(Path(a.params).read_text())
        params.update(d.get("params", d))
    swarm = SwarmConfig.from_specs(scenario.drones, scenario.policy,
                                   scenario.radio.get("scheme", "ours"), scenario.map_cell,
                                   PolicyParams.from_dict(params), scenario.threat)
    agent = Agent(a.id, City.load(Path(scenario.city)), config.KEY, swarm)

    ctx = zmq.Context.instance()
    ctx.setsockopt(zmq.LINGER, 0)  # never block exit on unsent messages
    world_sub = ctx.socket(zmq.SUB)
    world_sub.connect(config.WORLD_PUB)
    world_sub.setsockopt(zmq.SUBSCRIBE, topic("tick", a.id))
    radio_sub = ctx.socket(zmq.SUB)
    radio_sub.connect(config.RADIO_PUB)
    radio_sub.setsockopt(zmq.SUBSCRIBE, topic("rx", a.id))
    to_world = ctx.socket(zmq.PUSH)
    to_world.connect(config.WORLD_PULL)
    to_radio = ctx.socket(zmq.PUSH)
    to_radio.connect(config.RADIO_PULL)

    poller = zmq.Poller()
    poller.register(world_sub, zmq.POLLIN)
    poller.register(radio_sub, zmq.POLLIN)
    inbox: list = []  # [frame, margin_db] pairs

    def drain_radio() -> None:
        while True:
            try:
                _, payload = radio_sub.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                return
            inbox.extend(unpack(payload))

    try:
        while True:
            ev = dict(poller.poll(1000))
            if radio_sub in ev:
                drain_radio()
            if world_sub in ev:
                # if we fell behind, step once on the newest tick with every queued reading
                payloads = []
                while True:
                    try:
                        payloads.append(world_sub.recv_multipart(zmq.NOBLOCK)[1])
                    except zmq.Again:
                        break
                if not payloads:
                    continue
                sensed = merge_ticks([sensed_from_dict(unpack(p)) for p in payloads])
                if array is not None:  # real sound replaces the simulated sound readings
                    keep = [r for r in sensed.readings if r.sensor != Sensor.SOUND]
                    sensed = Sensed(sensed.state, keep + array.readings(), sensed.camera_frame)
                drain_radio()  # frames that arrived since the poll belong to this tick too
                out = agent.step(sensed.state.t, sensed, inbox)
                inbox.clear()
                telemetry = out.telemetry
                if array is not None:
                    telemetry = {**telemetry, "array": array.diagnostics}
                to_world.send(pack({"id": a.id, "command": command_to_dict(out.command),
                                    "telemetry": telemetry, "events": out.events}))
                if out.outbox:
                    to_radio.send(pack({"id": a.id, "frames": out.outbox}))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
