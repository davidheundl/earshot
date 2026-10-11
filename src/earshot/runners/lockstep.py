"""Lockstep runner (fast mode): world, radio and all agents in one process, one shared clock.

Reproducible by construction: same scenario + seed -> identical run (ADR 0001). This is what
the 100-run evaluation and the tests use. The live runner drives the same Agent class.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from earshot import config
from earshot.agent import Agent
from earshot.backend.twin2d import Twin2D
from earshot.city import City
from earshot.radio.medium import RadioConfig, RadioMedium
from earshot.scenario import Scenario
from earshot.world import World


@dataclass
class RunResult:
    scenario: str
    seed: int
    steps: int
    radio: dict
    agents: dict


def build(scenario: Scenario, root: Path = Path(".")):
    city = City.load(root / scenario.city)
    backend = Twin2D(city, scenario.drones)
    world = World(scenario, city, backend)
    radio = RadioMedium(
        RadioConfig.from_dict(scenario.radio), np.random.default_rng(scenario.seed + 2)
    )
    agents = {
        d.id: Agent(d.id, city, config.KEY, d.home or (d.x, d.y)) for d in scenario.drones
    }
    return city, world, radio, agents


def run(scenario: Scenario, log: Path | None = None, root: Path = Path(".")) -> RunResult:
    """Run one scenario. With `log`, writes 1 Hz snapshots to `log` and every sensor
    detection with its true source to `<log stem>.detections.jsonl`."""
    _, world, radio, agents = build(scenario, root)
    commands: dict = {}
    inbox: dict[int, list[bytes]] = {}
    telemetry: dict[int, dict] = {}
    steps = round(scenario.duration / scenario.dt)
    log_every = round(1.0 / scenario.dt)  # 1 Hz
    det_log = log.with_name(log.stem + ".detections.jsonl") if log else None
    with (
        open(log, "w") if log else contextlib.nullcontext() as fh,
        open(det_log, "w") if det_log else contextlib.nullcontext() as dh,
    ):
        for i in range(steps):
            sensed = world.step(commands)
            t = world.t
            outgoing: dict[int, list[bytes]] = {}
            for did, agent in agents.items():
                out = agent.step(t, sensed[did], inbox.get(did, []))
                commands[did] = out.command
                outgoing[did] = out.outbox
                telemetry[did] = out.telemetry
            inbox = radio.step(t, world.positions(), outgoing, scenario.jam_zones(t))
            if dh:
                for d in world.detections:
                    dh.write(json.dumps(d) + "\n")
            if fh and i % log_every == 0:
                snap = world.snapshot(telemetry)
                snap["radio"] = radio.stats()
                fh.write(json.dumps(snap) + "\n")
    return RunResult(scenario.name, scenario.seed, steps, radio.stats(), telemetry)
