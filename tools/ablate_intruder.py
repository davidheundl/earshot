"""Which sensor family finds the intruder: random worlds whose intruder has an RF link, run as
generated, with every intruder's RF stripped, and with every intruder's sound stripped
(sound power 0 dB). Paired by seed, so each arm sees the same world.

    uv run python tools/ablate_intruder.py --seed 1000 --runs 40 --jobs 8 [--out rows.json]

Per arm: medians of loc_err and its range (radial) and bearing (tangential) parts (eval.py),
confirmed drone Sources, and in how many runs any drone ever held a Suspicion of the Threat
Class (RunResult.spread not None) and any Verifier was dispatched.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from concurrent.futures import ProcessPoolExecutor

from earshot.eval import load_scenario, metrics
from earshot.runners.lockstep import run

ARMS = ("both", "no_rf", "no_sound")


def one(seed: int, arm: str) -> dict:
    sc = load_scenario("random", seed)
    if arm == "no_rf":
        sc.sources = [dataclasses.replace(s, rf_bands=()) if s.intruder else s
                      for s in sc.sources]
    elif arm == "no_sound":
        sc.sources = [dataclasses.replace(s, level_db=0.0) if s.intruder else s
                      for s in sc.sources]
    r = run(sc)
    m = metrics(r, "earshot", sc.duration)
    return {"seed": seed, "arm": arm, "loc_err": m.loc_err, "radial": m.loc_radial,
            "tangential": m.loc_tangential, "confirmed": m.confirmed, "drones": m.drones,
            "t_confirm": m.t_confirm, "false_verify": m.false_verify,
            "flights": m.verify_flights,
            "suspicion_s": sum(1 for _, s in r.spread or [] if s is not None)}


def summary(rows: list[dict], arm: str) -> str:
    rs = [r for r in rows if r["arm"] == arm]

    def med(k):
        v = [r[k] for r in rs if r[k] is not None and r[k] != float("inf")]
        return f"{statistics.median(v):.0f}" if v else "-"

    return (f"{arm:9} runs {len(rs)}  loc_err {med('loc_err')}  radial {med('radial')}  "
            f"tangential {med('tangential')}  confirmed "
            f"{sum(r['confirmed'] for r in rs)}/{sum(r['drones'] for r in rs)}  "
            f"t_confirm {med('t_confirm')}  flights {med('flights')}  "
            f"runs with a Suspicion {sum(r['suspicion_s'] > 0 for r in rs)}  "
            f"with a dispatch {sum(r['flights'] > 0 for r in rs)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--runs", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    seeds = [s for s in range(a.seed, a.seed + a.runs)
             if any(x.intruder and x.rf_bands for x in load_scenario("random", s).sources)]
    print(f"{len(seeds)} of {a.runs} seeds have an RF intruder", flush=True)
    jobs = [(s, arm) for arm in ARMS for s in seeds]
    with ProcessPoolExecutor(max(1, a.jobs)) as pool:
        rows = list(pool.map(one, *zip(*jobs)))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)
    for arm in ARMS:
        print(summary(rows, arm))


if __name__ == "__main__":
    main()
