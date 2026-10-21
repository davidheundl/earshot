"""Evaluation over many seeded runs: concept section 17 metrics, policies side by side.

    earshot eval --scenario scenarios/demo.yaml --runs 20 [--policy earshot nearest no_perch]
                 [--jobs N]

Per run, from the lockstep RunResult (decision events + ground truth), with the scenario's
`duration` as the censoring value for anything that never happened. "Drone Sources" below
are the intruder Sources (Scenario `intruder:`, by default those of the Threat Class): the
drones at the airport, the poachers' vehicles in a Reserve:
    t_confirm      mean over drone Sources of s from the Source appearing to the first camera
                   confirmation within CONFIRM_MATCH of it; a never-confirmed Source counts as
                   `duration` (censored, as tune's cost does). None: no drone Source in the run
    t_dispatch     the same for the first Verifier dispatched within FALSE_RADIUS of it
    drones         drone Sources in the run; confirmed: how many of them were confirmed
    loc_err        median over all Verifier dispatches of m from the dispatch target to the
                   nearest drone Source in the air (inf if none was); None: no dispatch
    false_verify   Verifier dispatches with no drone Source within FALSE_RADIUS
    missed         drone Sources never confirmed
    busy_max       most drones away from their default post at once (the Gap)
    load           radio channel load
    clean_rx       share of receptions not lost to collisions
    t_agree        s after the last jam zone ends until all drones' top Suspicions agree within
                   AGREE_RADIUS; never re-agreed: `duration` - jam end (censored, agreed False).
                   None (and agreed None): no jam in the run, or it lasts past its end
    battery_used   mean battery fraction used per drone
"""

from __future__ import annotations

import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field

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
    duration: float = 0.0
    drones: int = 0
    confirmed: int = 0
    agreed: bool | None = None
    t_confirm_each: list = field(default_factory=list)  # per drone Source, None: never
    t_dispatch_each: list = field(default_factory=list)


def censored_mean(vals: list[float | None], cap: float) -> float | None:
    """Mean with every None (it never happened) counted as `cap`; None for no values."""
    return statistics.mean(cap if v is None else v for v in vals) if vals else None


def _sources(r: RunResult) -> list[Source]:
    return [Source(s["id"], TargetClass[s["cls"]], s["start"], s["end"],
                   [tuple(p) for p in s["path"]], 0.0,
                   intruder=s.get("intruder", s["cls"] == "DRONE")) for s in r.truth]


def _nearest_intruder(sources, x, y, t) -> float:
    d = [math.hypot(p[0] - x, p[1] - y) for s in sources if s.intruder
         for p in [s.position(t)] if p is not None]
    return min(d, default=math.inf)


def _first(events, s: Source, radius: float) -> float | None:
    """s from `s` appearing to the first event within `radius` of it, or None."""
    hits = [e["t"] for e in events if e["t"] >= s.start
            and (p := s.position(e["t"])) is not None
            and math.hypot(p[0] - e["x"], p[1] - e["y"]) <= radius]
    return min(hits) - s.start if hits else None


def metrics(r: RunResult, policy: str, duration: float) -> RunMetrics:
    """`duration`: the scenario's length, the value a censored latency counts as."""
    sources = _sources(r)
    drones = [s for s in sources if s.intruder]
    dispatches = [e for e in r.events if e["kind"] == "dispatch" and e["task"] == "verify"]
    confirms = [e for e in r.events if e["kind"] == "confirmed"]

    t_conf = [_first(confirms, s, CONFIRM_MATCH) for s in drones]
    t_disp = [_first(dispatches, s, FALSE_RADIUS) for s in drones]
    errs = [_nearest_intruder(sources, e["x"], e["y"], e["t"]) for e in dispatches]
    t_agree, agreed = _time_to_agree(r, duration)
    confirmed = sum(t is not None for t in t_conf)
    return RunMetrics(
        seed=r.seed, policy=policy,
        t_confirm=censored_mean(t_conf, duration),
        t_dispatch=censored_mean(t_disp, duration),
        loc_err=statistics.median(errs) if errs else None,
        false_verify=sum(1 for d in errs if d > FALSE_RADIUS),
        verify_flights=len(dispatches),
        missed=len(drones) - confirmed,
        busy_max=r.busy_max,
        load=r.radio["channel_load"],
        clean_rx=r.radio["delivered"] / max(1, r.radio["delivered"] + r.radio["collided"]),
        t_agree=t_agree,
        battery_used=1.0 - statistics.mean(r.battery.values()),
        duration=duration, drones=len(drones), confirmed=confirmed, agreed=agreed,
        t_confirm_each=t_conf, t_dispatch_each=t_disp,
    )


def _time_to_agree(r: RunResult, duration: float) -> tuple[float | None, bool | None]:
    """(s from the last jam's end to agreement, agreed): (None, None) without a jam or when
    it outlasts the run; (duration - jam end, False) if the drones never agree again."""
    if not r.jams or max(e for _, e in r.jams) >= duration:
        return None, None
    end = max(e for _, e in r.jams)
    for t, s in r.spread or []:
        if t >= end and s is not None and s <= AGREE_RADIUS:
            return t - end, True
    return duration - end, False


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
            audio=None, duration: float | None = None) -> RunMetrics:
    """`audio`: sensors.Audio (see sound_audio) to classify sound with a trained model.
    `duration`: override the scenario's length (sim seconds)."""
    sc = load_scenario(scenario_path, seed)
    sc.policy = policy
    if params is not None:  # state.PolicyParams overrides, on top of the scenario's own
        sc.params = {**sc.params, **params}
    if scheme:
        sc.radio = {**sc.radio, "scheme": scheme}
    if bw_khz:
        sc.radio = {**sc.radio, "bw_khz": bw_khz}
    if duration:
        sc.duration = duration
    return metrics(run(sc, audio=audio), policy, sc.duration)


def evaluate(scenario_path: str, runs: int, seed0: int, policy: str,
             scheme: str | None = None, bw_khz: float | None = None,
             params: dict | None = None, audio=None, jobs: int = 1,
             duration: float | None = None) -> list[RunMetrics]:
    """Seeds seed0 .. seed0 + runs - 1, in order. `jobs` > 1: that many processes (each run
    is seeded and independent, so the rows are the same as serial); not with `audio`, a
    loaded model that does not travel to a worker."""
    args = [(scenario_path, seed0 + k, policy, scheme, bw_khz, params, audio, duration)
            for k in range(runs)]
    if jobs <= 1:
        return [run_one(*x) for x in args]
    if audio is not None:
        raise ValueError("jobs > 1 cannot share a sound classifier: run serially")
    with ProcessPoolExecutor(jobs) as pool:
        futures = [pool.submit(run_one, *x) for x in args]
        return [f.result() for f in futures]


def summarize(rows: list[RunMetrics], label: str | None = None) -> dict:
    """One table row: median [min-max] over runs. Latencies are already censored per run
    (never happened = the scenario's duration, see the module docstring), so no run drops
    out of t_dispatch_s / t_confirm_s / t_agree_s; "-" only where none applies (no drone
    Source, no jam). Counts beside them:
        confirmed    drone Sources confirmed / drone Sources, summed over runs
        agreed       jammed runs whose drones agreed again / jammed runs
    loc_err_m is the median per run over all Verifier dispatches (see metrics)."""
    def stat(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "-"
        return f"{statistics.median(vals):.1f} [{min(vals):.0f}-{max(vals):.0f}]"

    n = len(rows)
    jammed = [r for r in rows if r.agreed is not None]
    return {
        "policy": label or rows[0].policy,
        "runs": n,
        "confirmed": f"{sum(r.confirmed for r in rows)}/{sum(r.drones for r in rows)}",
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
        "agreed": f"{sum(bool(r.agreed) for r in jammed)}/{len(jammed)}" if jammed else "-",
        "battery_%": stat([100 * r.battery_used for r in rows]),
    }


def as_json(rows: list[RunMetrics]) -> list[dict]:
    return [asdict(r) for r in rows]
