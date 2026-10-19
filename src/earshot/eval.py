"""Evaluation over many seeded runs: concept section 17 metrics, policies side by side.

    earshot eval --scenario scenarios/demo.yaml --runs 20 [--policy earshot nearest no_perch]

Per run, from the lockstep RunResult (decision events + ground truth):
    t_confirm      s from a drone Source appearing to the first camera confirmation within
                   CONFIRM_MATCH of it (None if never)
    t_dispatch     s from a drone Source appearing to the first Verifier dispatched within
                   FALSE_RADIUS of it (None if never)
    loc_err        m between the Suspicion and the true drone when a Verifier is dispatched
    false_verify   Verifier dispatches with no drone Source within FALSE_RADIUS
    missed         drone Sources never confirmed
    busy_max       most drones away from their default post at once (the Gap)
    load           radio channel load
    clean_rx       share of receptions not lost to collisions
    t_agree        s after a jam zone ends until all drones' top Suspicions agree within
                   AGREE_RADIUS (None if no jam, or never)
    battery_used   mean battery fraction used per drone
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass

from earshot.messages import TargetClass
from earshot.runners.lockstep import RunResult, run
from earshot.scenario import Scenario, Source

CONFIRM_MATCH = 100.0
FALSE_RADIUS = 150.0
AGREE_RADIUS = 100.0


@dataclass
class RunMetrics:
    seed: int
    policy: str
    t_confirm: float | None
    t_dispatch: float | None
    loc_err: float | None
    false_verify: int
    verify_flights: int
    missed: int
    busy_max: int
    load: float
    clean_rx: float
    t_agree: float | None
    battery_used: float


def _sources(r: RunResult) -> list[Source]:
    return [Source(s["id"], TargetClass[s["cls"]], s["start"], s["end"],
                   [tuple(p) for p in s["path"]], 0.0) for s in r.truth]


def _nearest_drone(sources, x, y, t) -> float:
    d = [math.hypot(p[0] - x, p[1] - y) for s in sources if s.cls == TargetClass.DRONE
         for p in [s.position(t)] if p is not None]
    return min(d, default=math.inf)


def metrics(r: RunResult, policy: str) -> RunMetrics:
    sources = _sources(r)
    drones = [s for s in sources if s.cls == TargetClass.DRONE]
    dispatches = [e for e in r.events if e["kind"] == "dispatch" and e["task"] == "verify"]
    confirms = [e for e in r.events if e["kind"] == "confirmed"]

    t_conf, missed = [], 0
    for s in drones:
        hits = [e["t"] for e in confirms if e["t"] >= s.start
                and (p := s.position(e["t"])) is not None
                and math.hypot(p[0] - e["x"], p[1] - e["y"]) <= CONFIRM_MATCH]
        if hits:
            t_conf.append(min(hits) - s.start)
        else:
            missed += 1
    t_disp = [min(hits) - s.start for s in drones
              if (hits := [e["t"] for e in dispatches if e["t"] >= s.start
                           and (p := s.position(e["t"])) is not None
                           and math.hypot(p[0] - e["x"], p[1] - e["y"]) <= FALSE_RADIUS])]
    errs = [_nearest_drone(sources, e["x"], e["y"], e["t"]) for e in dispatches]
    true_errs = [d for d in errs if d <= FALSE_RADIUS]
    return RunMetrics(
        seed=r.seed, policy=policy,
        t_confirm=min(t_conf) if t_conf else None,
        t_dispatch=min(t_disp) if t_disp else None,
        loc_err=statistics.median(true_errs) if true_errs else None,
        false_verify=sum(1 for d in errs if d > FALSE_RADIUS),
        verify_flights=len(dispatches),
        missed=missed,
        busy_max=r.busy_max,
        load=r.radio["channel_load"],
        clean_rx=r.radio["delivered"] / max(1, r.radio["delivered"] + r.radio["collided"]),
        t_agree=_time_to_agree(r),
        battery_used=1.0 - statistics.mean(r.battery.values()),
    )


def _time_to_agree(r: RunResult) -> float | None:
    if not r.jams:
        return None
    end = max(e for _, e in r.jams)
    for t, s in r.spread:
        if t >= end and s is not None and s <= AGREE_RADIUS:
            return t - end
    return None


def load_scenario(scenario_path: str, seed: int) -> Scenario:
    """A scenario YAML at `seed`, or with "random" a generated world for that seed."""
    if scenario_path == "random":
        from earshot.scenario_gen import generate

        return generate(Scenario.load("scenarios/demo.yaml"), seed)
    return Scenario.load(scenario_path, seed=seed)


def sound_audio(classifier: str | None, sounds: str = "data/sounds"):
    """sensors.Audio with a trained sound classifier ("pkg.module:attr"), or None for the
    CONFUSION stand-in. Built once per classifier: loading a model is the slow part."""
    if not classifier or classifier == "none":
        return None
    from pathlib import Path

    from earshot.sensors import Audio
    from earshot.sensors import waveform as wf

    return Audio(wf.Mixer(wf.ClipBank(Path(sounds))), wf.load_classifier(classifier))


def run_one(scenario_path: str, seed: int, policy: str, scheme: str | None = None,
            bw_khz: float | None = None, params: dict | None = None,
            audio=None) -> RunMetrics:
    """`audio`: sensors.Audio (see sound_audio) to classify sound with a trained model."""
    sc = load_scenario(scenario_path, seed)
    sc.policy = policy
    if params is not None:  # state.PolicyParams overrides, on top of the scenario's own
        sc.params = {**sc.params, **params}
    if scheme:
        sc.radio = {**sc.radio, "scheme": scheme}
    if bw_khz:
        sc.radio = {**sc.radio, "bw_khz": bw_khz}
    return metrics(run(sc, audio=audio), policy)


def evaluate(scenario_path: str, runs: int, seed0: int, policy: str,
             scheme: str | None = None, bw_khz: float | None = None,
             params: dict | None = None, audio=None) -> list[RunMetrics]:
    return [run_one(scenario_path, seed0 + k, policy, scheme, bw_khz, params, audio)
            for k in range(runs)]


def summarize(rows: list[RunMetrics], label: str | None = None) -> dict:
    def stat(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-"
        return f"{statistics.median(vals):.1f} [{min(vals):.0f}-{max(vals):.0f}]"

    n = len(rows)
    return {
        "policy": label or rows[0].policy,
        "runs": n,
        "confirmed": f"{sum(r.t_confirm is not None for r in rows)}/{n}",
        "t_dispatch_s": stat([r.t_dispatch for r in rows]),
        "t_confirm_s": stat([r.t_confirm for r in rows]),
        "loc_err_m": stat([r.loc_err for r in rows]),
        "false_verify": stat([r.false_verify for r in rows]),
        "verify_flights": stat([r.verify_flights for r in rows]),
        "missed": stat([r.missed for r in rows]),
        "busy_max": stat([r.busy_max for r in rows]),
        "load_%": stat([100 * r.load for r in rows]),
        "clean_rx_%": stat([100 * r.clean_rx for r in rows]),
        "t_agree_s": stat([r.t_agree for r in rows]),
        "battery_%": stat([100 * r.battery_used for r in rows]),
    }


def as_json(rows: list[RunMetrics]) -> list[dict]:
    return [asdict(r) for r in rows]
