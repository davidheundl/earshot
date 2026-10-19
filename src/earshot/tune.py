"""Tune the decision layer's numbers (state.PolicyParams) with CMA-ES over seeded runs.

    earshot tune [--gens 30] [--pop 12] [--train 10] [--holdout 20] [--workers 8]

The structure stays hand-written (auction, Claims, leases, Sectors); only the thresholds in
GENES move. Each candidate runs the earshot policy on the same training seeds (generated
worlds, scenario "random"), so candidates differ only by their parameters; a run is
deterministic, so the objective is too. Cost per run, in seconds of delay:

    t_confirm                       mean over the drone Sources (eval.metrics: one never
                                    confirmed counts as the scenario's duration), so a
                                    second intruder weighs as much as the first
    + W_MISS * missed               a drone Source never confirmed
    + W_FALSE * false_verify        a Verifier sent where no drone is
    + W_GAP * busy_max              drones away from their post at once (the Gap)

Generation 0 is the hand-tuned default. At the end, default and best run on held-out seeds
the optimiser never saw, compared seed by seed; the tuned set counts as a win only if it
misses no more drones there than the default. Everything lands in runs/tune-<time>/:
history.jsonl (one line per candidate, with its cost components), best.json (`--params`
for eval and fast) and holdout.json.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

from earshot.eval import RunMetrics, run_one
from earshot.state import PolicyParams

# gene -> (low, high): bounds keep the protocol safe, not just plausible
GENES = {
    "verify_min_strength": (0.5, 10.0),  # 0 floods
    "verify_min_confidence": (0.2, 0.9),
    "top_suspicions": (1.0, 3.999),  # floored: 1-3
    "confirmed_shadow": (50.0, 400.0),
    "settle": (2.0, 8.0),  # below ~2 s rival bids arrive too late: two drones fly
    "same_task_radius": (100.0, 400.0),
    "arrive": (45.0, 150.0),  # the standoff is 40 m: below it a Verifier never arrives
    "dismiss_after": (3.0, 20.0),
}
W_FALSE = 10.0
W_GAP = 10.0
W_MISS = 600.0  # a missed drone is the one unacceptable outcome: no trade buys it back
TRAIN_SEED0 = 0
TRAIN_RUNS = 10  # default --train: seeds TRAIN_SEED0 .. + TRAIN_RUNS - 1
HOLDOUT_SEED0 = 1000


def to_params(x) -> dict:
    """A point in [0, 1]^len(GENES) -> PolicyParams overrides."""
    out = {}
    for (name, (lo, hi)), v in zip(GENES.items(), x):
        out[name] = lo + min(1.0, max(0.0, float(v))) * (hi - lo)
    out["top_suspicions"] = int(out["top_suspicions"] + 1e-6)  # the default 2 maps back to 2
    return out


def to_unit(params: PolicyParams) -> list[float]:
    return [(getattr(params, name) - lo) / (hi - lo) for name, (lo, hi) in GENES.items()]


def run_cost(m: RunMetrics) -> dict:
    t_c = m.t_confirm if m.t_confirm is not None else 0.0  # None: no drone Source to confirm
    return {"cost": t_c + W_MISS * m.missed + W_FALSE * m.false_verify + W_GAP * m.busy_max,
            "t_confirm": t_c, "false_verify": m.false_verify, "busy_max": m.busy_max,
            "missed": m.missed, "verify_flights": m.verify_flights,
            "battery": m.battery_used}


def _job(scenario: str, seed: int, params: dict) -> dict:
    """One (candidate, seed) run in a worker process: returns its cost components."""
    return run_cost(run_one(scenario, seed, "earshot", params=params))


def _mean(rows: list[dict]) -> dict:
    return {k: statistics.mean(r[k] for r in rows) for k in rows[0]}


def evaluate_all(pool, scenario: str, candidates: list[dict], seeds: list[int]) -> list[list[dict]]:
    futures = [[pool.submit(_job, scenario, s, c) for s in seeds] for c in candidates]
    return [[f.result() for f in row] for row in futures]


def tune(scenario: str = "random", gens: int = 30, pop: int = 12, train: int = TRAIN_RUNS,
         holdout: int = 20, workers: int | None = None, sigma0: float = 0.25,
         seed: int = 1, out: Path | None = None) -> Path:
    import warnings

    warnings.filterwarnings("ignore", message="Could not import matplotlib")
    import cma

    out = out or Path("runs") / time.strftime("tune-%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    train_seeds = list(range(TRAIN_SEED0, TRAIN_SEED0 + train))
    hold_seeds = list(range(HOLDOUT_SEED0, HOLDOUT_SEED0 + holdout))
    default = {f.name: getattr(PolicyParams(), f.name) for f in fields(PolicyParams)}
    t0 = time.time()
    history = (out / "history.jsonl").open("w")

    def log(gen, params, rows):
        rec = {"gen": gen, "params": params, **_mean(rows), "t": round(time.time() - t0)}
        history.write(json.dumps(rec) + "\n")
        history.flush()
        return rec

    with ProcessPoolExecutor(workers or os.cpu_count()) as pool:
        best = base = log(0, default, evaluate_all(pool, scenario, [default], train_seeds)[0])
        print(f"gen  0  default            cost {best['cost']:7.1f}", flush=True)
        es = cma.CMAEvolutionStrategy(to_unit(PolicyParams()), sigma0, {
            "bounds": [0.0, 1.0], "popsize": pop, "seed": seed, "verbose": -9})
        for gen in range(1, gens + 1):
            xs = es.ask()
            cands = [to_params(x) for x in xs]
            recs = [log(gen, c, rows) for c, rows in
                    zip(cands, evaluate_all(pool, scenario, cands, train_seeds))]
            es.tell(xs, [r["cost"] for r in recs])
            gen_best = min(recs, key=lambda r: r["cost"])
            if gen_best["cost"] < best["cost"]:
                best = gen_best
                (out / "best.json").write_text(json.dumps(best, indent=1))
            print(f"gen {gen:2d}  best {gen_best['cost']:7.1f}  overall {best['cost']:7.1f}"
                  f"  mean {statistics.mean(r['cost'] for r in recs):7.1f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
        if not (out / "best.json").exists():
            (out / "best.json").write_text(json.dumps(best, indent=1))

        rows_d, rows_b = evaluate_all(pool, scenario, [default, best["params"]], hold_seeds)
    history.close()
    report = holdout_report(hold_seeds, rows_d, rows_b)
    report.update(default=default, best=best["params"], train_cost_default=base["cost"],
                  train_cost_best=best["cost"])
    (out / "holdout.json").write_text(json.dumps(report, indent=1))
    print_report(report)
    print(f"-> {out}  ({time.time() - t0:.0f}s)")
    return out


def holdout_report(seeds, rows_d: list[dict], rows_b: list[dict]) -> dict:
    """Paired per seed: on how many held-out seeds is the tuned set better, and by how much."""
    deltas = [b["cost"] - d["cost"] for d, b in zip(rows_d, rows_b)]
    return {
        "seeds": seeds,
        "better": sum(x < 0 for x in deltas), "worse": sum(x > 0 for x in deltas),
        "median_delta": statistics.median(deltas),
        "missed_default": sum(d["missed"] for d in rows_d),
        "missed_best": sum(b["missed"] for b in rows_b),
        "default_mean": _mean(rows_d), "best_mean": _mean(rows_b),
        "per_seed": [{"seed": s, "default": d, "best": b}
                     for s, d, b in zip(seeds, rows_d, rows_b)],
    }


def print_report(r: dict) -> None:
    n = len(r["seeds"])
    print(f"held-out: tuned better on {r['better']}/{n} seeds, worse on {r['worse']}/{n},"
          f" median cost change {r['median_delta']:+.1f};"
          f" missed drones {r['missed_default']} -> {r['missed_best']}")
    keys = list(r["default_mean"])
    print("         " + "  ".join(f"{k:>14}" for k in keys))
    for label in ("default", "best"):
        m = r[f"{label}_mean"]
        print(f"{label:8} " + "  ".join(f"{m[k]:14.2f}" for k in keys))
    print("params:  " + ", ".join(f"{k}={v:.3g}" for k, v in r["best"].items()))
