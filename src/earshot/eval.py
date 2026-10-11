"""Evaluation over many seeded runs: concept section 17 metrics, policies side by side.

    earshot eval --scenario scenarios/demo.yaml --runs 20 [--policy earshot nearest no_perch]

Per run, from the lockstep RunResult (decision events + ground truth):
    t_confirm      s from a drone Source appearing to the first camera confirmation within
                   CONFIRM_MATCH of it (None if never)
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
    errs = [_nearest_drone(sources, e["x"], e["y"], e["t"]) for e in dispatches]
    true_errs = [d for d in errs if d <= FALSE_RADIUS]
    return RunMetrics(
        seed=r.seed, policy=policy,
        t_confirm=min(t_conf) if t_conf else None,
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


def evaluate(scenario_path: str, runs: int, seed0: int, policy: str,
             scheme: str | None = None, bw_khz: float | None = None) -> list[RunMetrics]:
    out = []
    base = Scenario.load("scenarios/demo.yaml") if scenario_path == "random" else None
    for k in range(runs):
        if base is not None:
            from earshot.scenario_gen import generate

            sc = generate(base, seed0 + k)
        else:
            sc = Scenario.load(scenario_path, seed=seed0 + k)
        sc.policy = policy
        if scheme:
            sc.radio = {**sc.radio, "scheme": scheme}
        if bw_khz:
            sc.radio = {**sc.radio, "bw_khz": bw_khz}
        out.append(metrics(run(sc), policy))
    return out


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
