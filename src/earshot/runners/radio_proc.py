"""Live radio process: the only path between agents (ADR 0003).

Clocked by the world's "state." snapshots (positions, time, jam zones). Agents PUSH frames
here; deliveries go out as "rx.<id>.", channel statistics as "radio." for the dashboard.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import zmq

from earshot import config
from earshot.radio.medium import RadioConfig, RadioMedium
from earshot.runners.wire import pack, topic, unpack
from earshot.scenario import Scenario


def _skipped(what: str) -> None:
    """Log a message the radio could not handle; the run goes on without it."""
    print(f"[radio] skipped {what}:\n{traceback.format_exc()}", file=sys.stderr, flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="scenarios/demo.yaml")
    a = ap.parse_args(argv)
    scenario = Scenario.load(a.scenario)
    medium = RadioMedium(
        RadioConfig.from_dict(scenario.radio), np.random.default_rng(scenario.seed + 2)
    )

    ctx = zmq.Context.instance()
    ctx.setsockopt(zmq.LINGER, 0)  # never block exit on unsent messages
    sub = ctx.socket(zmq.SUB)
    sub.connect(config.WORLD_PUB)
    sub.setsockopt(zmq.SUBSCRIBE, topic("state"))
    pull = ctx.socket(zmq.PULL)
    pull.bind(config.RADIO_PULL)
    pub = ctx.socket(zmq.PUB)
    pub.bind(config.RADIO_PUB)

    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)
    poller.register(pull, zmq.POLLIN)
    outgoing: dict[int, list[bytes]] = {}
    last_stats = -1.0
    print("[radio] up", flush=True)
    try:
        while True:
            ev = dict(poller.poll(1000))
            if pull in ev:
                while True:
                    try:
                        payload = pull.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    try:
                        msg = unpack(payload)
                        frames = msg["frames"]
                        if not all(isinstance(f, bytes) for f in frames):
                            raise ValueError("frames must be bytes")
                        outgoing.setdefault(msg["id"], []).extend(frames)
                    except Exception:  # noqa: BLE001 - skip it, keep the channel up
                        _skipped("frames from an agent")
            if sub in ev:
                _, payload = sub.recv_multipart()
                try:
                    snap = unpack(payload)
                    t = snap["t"]
                    pos = {d["id"]: (d["x"], d["y"], d["z"]) for d in snap["drones"]}
                    jam = [SimpleNamespace(center=(j["x"], j["y"]), radius=j["r"])
                           for j in snap["jamming"]]
                    delivered = medium.step(t, pos, outgoing, jam)
                    for r, frames in delivered.items():
                        pub.send_multipart([topic("rx", r), pack(frames)])
                    if t - last_stats >= 0.5:
                        last_stats = t
                        stats = medium.stats()
                        stats["links"] = [[s, r] for (s, r), lt in medium.links.items()
                                          if t - lt < 3]
                        pub.send_multipart([topic("radio"), pack(stats)])
                except Exception:  # noqa: BLE001 - skip this tick, keep the channel up
                    _skipped("a world state tick")
                finally:
                    outgoing = {}  # never retry frames that may have caused the failure
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
