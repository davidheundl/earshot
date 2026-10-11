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
from earshot.state import SwarmConfig
from earshot.world import World


@dataclass
class RunResult:
    scenario: str
    seed: int
    steps: int
    radio: dict
    agents: dict
    events: list  # Agents' decisions (claim, dispatch, confirmed, dismissed, ...) with sim time
    truth: list  # every Source of the run, including ones added by events
    battery: dict  # drone -> battery left


def build(scenario: Scenario, root: Path = Path(".")):
    city = City.load(root / scenario.city)
    backend = Twin2D(city, scenario.drones)
    world = World(scenario, city, backend)
    radio = RadioMedium(
        RadioConfig.from_dict(scenario.radio), np.random.default_rng(scenario.seed + 2)
    )
    swarm = SwarmConfig.from_specs(scenario.drones, scenario.policy,
                                   scenario.radio.get("scheme", "ours"))
    agents = {d.id: Agent(d.id, city, config.KEY, swarm) for d in scenario.drones}
    return city, world, radio, agents


def run(scenario: Scenario, log: Path | None = None, root: Path = Path(".")) -> RunResult:
    """Run one scenario. With `log`, writes 1 Hz snapshots to `log` and every sensor
    detection with its true source to `<log stem>.detections.jsonl`."""
    _, world, radio, agents = build(scenario, root)
    commands: dict = {}
    inbox: dict[int, list[tuple[bytes, float]]] = {}
    telemetry: dict[int, dict] = {}
    steps = round(scenario.duration / scenario.dt)
    log_every = round(1.0 / scenario.dt)  # 1 Hz
    det_log = log.with_name(log.stem + ".detections.jsonl") if log else None
    events: list[dict] = []
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
                events.extend(out.events)
            inbox = radio.step(t, world.positions(), outgoing, scenario.jam_zones(t))
            if dh:
                for d in world.detections:
                    dh.write(json.dumps(d) + "\n")
            if fh and i % log_every == 0:
                snap = world.snapshot(_strip(telemetry))
                snap["radio"] = radio.stats()
                fh.write(json.dumps(snap) + "\n")
    truth = [{"id": s.id, "cls": s.cls.name, "start": s.start, "end": s.end,
              "path": [list(p) for p in s.path]} for s in scenario.sources]
    battery = {d: round(st.battery, 4) for d, st in world.states.items()}
    return RunResult(scenario.name, scenario.seed, steps, radio.stats(), _strip(telemetry),
                     events, truth, battery)


def _strip(telemetry: dict) -> dict:
    """Telemetry without the heatmap bytes (JSON logs, comparisons)."""
    return {d: {k: v for k, v in tm.items() if k != "heatmap"} for d, tm in telemetry.items()}
