"""Live agent process: one per drone. A thin shell around the pure Agent.

    python -m earshot.runners.agent_proc --id 3 --scenario scenarios/demo.yaml

Talks to nobody but the world (sensing, commands) and the radio (frames).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import zmq

from earshot import config
from earshot.agent import Agent
from earshot.city import City
from earshot.runners.wire import command_to_dict, pack, sensed_from_dict, topic, unpack
from earshot.scenario import Scenario
from earshot.state import SwarmConfig


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--scenario", default="scenarios/demo.yaml")
    a = ap.parse_args(argv)
    scenario = Scenario.load(a.scenario)
    swarm = SwarmConfig.from_specs(scenario.drones, scenario.policy,
                                   scenario.radio.get("scheme", "ours"))
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
    try:
        while True:
            ev = dict(poller.poll(1000))
            if radio_sub in ev:
                while True:
                    try:
                        _, payload = radio_sub.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    inbox.extend(unpack(payload))
            if world_sub in ev:
                # only the newest tick matters if we fell behind
                payload = None
                while True:
                    try:
                        _, payload = world_sub.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                sensed = sensed_from_dict(unpack(payload))
                out = agent.step(sensed.state.t, sensed, inbox)
                inbox = []
                to_world.send(pack({"id": a.id, "command": command_to_dict(out.command),
                                    "telemetry": out.telemetry, "events": out.events}))
                if out.outbox:
                    to_radio.send(pack({"id": a.id, "frames": out.outbox}))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
