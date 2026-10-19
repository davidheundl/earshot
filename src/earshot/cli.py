"""earshot <command>

    earshot fast   --scenario S [--runs N] [--seed K] [--log] [--events F]   lockstep, no sockets
                   [--audio DIR [--audio-beams]] [--classifier MOD:ATTR]       sound as waveforms
    earshot live   --scenario S [--backend twin2d|webots] [--events F|latest] [--no-dashboard]
                   [--audio] [--classifier MOD:ATTR]
    earshot doctor                                                pre-flight checks (run offline)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from earshot import config

WEBOTS_HOME = os.environ.get("WEBOTS_HOME", "/Applications/Webots.app")


def webots_env() -> dict:
    env = dict(os.environ)
    env["WEBOTS_HOME"] = WEBOTS_HOME
    env["PYTHONPATH"] = f"{WEBOTS_HOME}/Contents/lib/controller/python" + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["DYLD_LIBRARY_PATH"] = f"{WEBOTS_HOME}/Contents/lib/controller"
    env["PYTHONIOENCODING"] = "UTF-8"
    env["WEBOTS_CONTROLLER_URL"] = f"ipc://{config.WEBOTS_PORT}/supervisor"
    return env


# -- fast ---------------------------------------------------------------------------------


def cmd_fast(a) -> None:
    from earshot.runners.lockstep import run
    from earshot.scenario import Scenario

    events_file = _resolve_events(a.events) if a.events else None
    results = []
    t0 = time.time()
    for k in range(a.runs):
        sc = Scenario.load(a.scenario, seed=a.seed + k)
        if a.params:
            sc.params = {**sc.params, **_params(a)}
        if a.duration:
            sc.duration = a.duration
        if a.events:  # replay a recorded live session on top of the scenario
            recorded = [json.loads(line) for line in events_file.read_text().splitlines()]
            sc.events = sorted(sc.events + recorded, key=lambda e: e["t"])
        log = None
        if a.log:
            log = Path("runs") / f"{sc.name}-seed{sc.seed}.jsonl"
            log.parent.mkdir(exist_ok=True)
        audio, audio_out = _audio(a, sc)
        r = run(sc, log=log, audio=audio, audio_out=audio_out)
        if audio_out is not None:
            audio_out.close()
            print(f"audio: {audio_out.root}")
        results.append({"seed": r.seed, "radio": r.radio})
        print(f"seed {r.seed}: sent {r.radio['sent']} delivered {r.radio['delivered']} "
              f"collided {r.radio['collided']} load {r.radio['channel_load']:.1%}")
    print(f"{a.runs} run(s) in {time.time() - t0:.1f}s")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))


def _resolve_events(arg: str) -> Path:
    """A recorded session file, or 'latest' for the newest runs/live-*.events.jsonl."""
    if arg == "latest":
        found = sorted(p for p in Path("runs").glob("live-*.events.jsonl") if p.stat().st_size)
        if not found:
            sys.exit("no recorded sessions in runs/ yet: run `earshot live` and press some keys")
        print(f"replaying {found[-1]}")
        return found[-1]
    p = Path(arg)
    if not p.exists():
        have = sorted(Path("runs").glob("live-*.events.jsonl"))
        hint = "\n  ".join(str(h) for h in have[-5:]) or "(none yet)"
        sys.exit(f"{arg} not found. Recorded sessions:\n  {hint}\nor use --events latest")
    return p


def stale_webots() -> list[str]:
    """Webots instances an earlier Earshot run left on our port (never the user's own Webots,
    which has no --port flag). While one lives, a new instance silently moves to the next
    port and our controller would connect to the stale one."""
    out = subprocess.run(["pgrep", "-f", f"webots.* --port={config.WEBOTS_PORT}"],
                         capture_output=True, text=True, check=False).stdout
    return out.split()


def cmd_arrays(_a) -> None:
    from earshot.audio.array import list_arrays

    found = list_arrays()
    for i, name, ch in found:
        print(f"{i}: {name} ({ch} channels)")
    if not found:
        print("no input with 4+ channels: plug in the ReSpeaker, use the 6-channel firmware")


def _audio(a, sc):
    """(sensors.Audio, sensors.export.AudioExport) as the flags ask, or Nones."""
    if not (a.audio or a.classifier):
        return None, None
    from earshot.sensors import Audio
    from earshot.sensors import waveform as wf
    from earshot.sensors.export import AudioExport

    mixer = wf.Mixer(wf.ClipBank(Path(a.sounds)))
    clf = wf.load_classifier(a.classifier) if a.classifier else None
    out = None
    if a.audio:
        out = AudioExport(Path(a.audio) / f"{sc.name}-seed{sc.seed}", mixer, a.audio_beams)
    return Audio(mixer, clf), out


# -- eval ---------------------------------------------------------------------------------


def _params(a) -> dict | None:
    """--params FILE: a JSON object of state.PolicyParams overrides (`earshot tune` writes
    best.json with them under "params")."""
    if not a.params:
        return None
    d = json.loads(Path(a.params).read_text())
    return d.get("params", d)


def cmd_eval(a) -> None:
    from earshot.eval import as_json, evaluate, summarize

    t0 = time.time()
    table, raw = [], {}
    from earshot.eval import sound_audio

    combos = [(p, s, c) for c in (a.classifier or [None]) for p in a.policy
              for s in (a.scheme or [None])]
    audios = {c: sound_audio(c, a.sounds) for c in (a.classifier or [None])}
    for policy, scheme, c in combos:
        label = policy + (f"/{scheme}" if scheme else "") + (f"/{a.bw:g}kHz" if a.bw else "")
        if c:
            label += "/" + c.rpartition(":")[2]
        rows = evaluate(a.scenario, a.runs, a.seed, policy, scheme, a.bw, _params(a), audios[c])
        raw[label] = as_json(rows)
        table.append(summarize(rows, label))
    cols = list(table[0])
    width = {c: max(len(c), *(len(str(row[c])) for row in table)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    for row in table:
        print("  ".join(str(row[c]).ljust(width[c]) for c in cols))
    print(f"{a.runs} run(s) x {len(combos)} configurations in {time.time() - t0:.0f}s"
          "  (median [min-max] over seeds)")
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": table, "runs": raw}, indent=1))


def cmd_tune(a) -> None:
    from earshot.tune import tune

    tune(a.scenario, a.gens, a.pop, a.train, a.holdout, a.workers, a.sigma,
         out=Path(a.out) if a.out else None)


# -- live ---------------------------------------------------------------------------------


def cmd_live(a) -> None:
    from earshot.scenario import Scenario

    sc = Scenario.load(a.scenario)
    py = sys.executable
    procs: list[subprocess.Popen] = []
    # everything that can fail on bad arguments, before the first process starts
    events = _resolve_events(a.events) if a.events else None
    bad = [x for x in (a.array or []) if "=" not in x]
    if bad:
        sys.exit(f"--array takes <drone id>=<device>[@rotation], got {bad}")
    arrays = dict(x.split("=", 1) for x in (a.array or []))

    def spawn(args, env=None):
        procs.append(subprocess.Popen([py, "-m", *args], env=env))

    def _stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _stop)  # explicit: background jobs inherit SIGINT=ignore
    signal.signal(signal.SIGTERM, _stop)
    try:
        _launch(a, sc, py, procs, spawn, events, arrays)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 2.0
        for p in procs:
            try:
                p.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()


def _launch(a, sc, py, procs, spawn, events, arrays) -> None:
    """Start Webots (optional), radio, agents, dashboard and world; return when the world
    ends. A crashed helper is reported, not fatal (the dashboard never stops a run)."""
    if a.backend == "webots":
        world_file = Path(f"webots/worlds/{sc.name}.wbt")
        subprocess.run([py, "tools/build_webots_world.py", a.scenario, str(world_file)], check=True)
        if not a.no_webots_launch:
            stale = stale_webots()
            if stale:
                sys.exit(f"An earlier Earshot Webots is still open on port {config.WEBOTS_PORT} "
                         f"(pid {', '.join(stale)}). Close its window, or run: kill {' '.join(stale)}")
            procs.append(subprocess.Popen(
                [f"{WEBOTS_HOME}/Contents/MacOS/webots", "--mode=" + ("fast" if a.no_webots_view else "realtime"), "--batch",
                 f"--port={config.WEBOTS_PORT}",
                 *(["--no-rendering"] if a.no_webots_view else []),
                 "--stdout", "--stderr", str(world_file)]
            ))
            time.sleep(3)

    spawn(["earshot.runners.radio_proc", "--scenario", a.scenario])
    for d in sc.drones:
        extra = ["--array", arrays[str(d.id)]] if str(d.id) in arrays else []
        if extra and a.classifier:
            extra += ["--classifier", a.classifier]
        spawn(["earshot.runners.agent_proc", "--id", str(d.id), "--scenario", a.scenario, *extra])
    if not a.no_dashboard:
        spawn(["earshot.dashboard.server", "--scenario", a.scenario])
        print(f"dashboard: http://{config.HOST}:{config.DASHBOARD_PORT}")
    time.sleep(1.0)
    world_args = ["earshot.runners.world_proc", "--scenario", a.scenario, "--backend", a.backend,
                  "--speed", str(a.speed)]
    if events:
        world_args += ["--events", str(events)]
    if a.duration:
        world_args += ["--duration", str(a.duration)]
    if a.audio:
        world_args += ["--audio", "--sounds", a.sounds]
    if a.classifier:
        world_args += ["--classifier", a.classifier]
    spawn(world_args, env=webots_env() if a.backend == "webots" else None)

    world, reported = procs[-1], set()
    while world.poll() is None:  # the world ends the run
        for p in procs[:-1]:
            if p.poll() is not None and p.pid not in reported:
                reported.add(p.pid)
                print(f"[live] {' '.join(map(str, p.args[-4:]))} exited with {p.returncode}",
                      flush=True)
        time.sleep(0.5)
    if world.returncode:
        print(f"[live] world exited with {world.returncode}", flush=True)


# -- doctor -------------------------------------------------------------------------------


def cmd_doctor(a) -> None:
    ok = True

    def check(name, cond, hint=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + ("" if cond else f"  -> {hint}"))

    print("earshot doctor")
    check("python 3.11", sys.version_info[:2] == (3, 11), "uv python pin 3.11 && uv sync")
    check("Webots installed", Path(f"{WEBOTS_HOME}/Contents/MacOS/webots").exists(),
          "see README: install from the GitHub release")
    r = subprocess.run([sys.executable, "-c", "from controller import Supervisor"],
                       env=webots_env(), capture_output=True, check=False)
    check("Webots Python controller importable", r.returncode == 0, r.stderr.decode()[-200:])
    for name, ep in [("world pub", config.WORLD_PUB), ("world pull", config.WORLD_PULL),
                     ("radio pull", config.RADIO_PULL), ("radio pub", config.RADIO_PUB)]:
        port = int(ep.rsplit(":", 1)[1])
        with socket.socket() as s:
            check(f"port {port} free ({name})", s.connect_ex((config.HOST, port)) != 0,
                  "an earlier run is still alive: pkill -f earshot")
    with socket.socket() as s:
        check(f"port {config.DASHBOARD_PORT} free (dashboard)",
              s.connect_ex((config.HOST, config.DASHBOARD_PORT)) != 0, "pkill -f earshot")
    stale = stale_webots()
    check(f"no leftover Earshot Webots on port {config.WEBOTS_PORT}", not stale,
          f"close its window or: kill {' '.join(stale)}")
    check("city data present", Path("data/cities/synthetic.json").exists(),
          "uv run python tools/make_synthetic_city.py")
    check("default HMAC key replaced", "EARSHOT_KEY" in os.environ,
          "export EARSHOT_KEY=... (fine to skip for dev)")
    try:
        import sounddevice as sd

        n_in = [d for d in sd.query_devices() if d["max_input_channels"] >= 4]
        check("4+ channel input device (ReSpeaker)", n_in, "plug in the array; grant mic access")
    except Exception as e:  # noqa: BLE001
        check("audio extras installed", False, f"uv sync ({e})")
    print("all good" if ok else "some checks failed (non-fatal ones are fine during dev)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="earshot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fast", help="lockstep runs without Webots or sockets")
    f.add_argument("--scenario", default="scenarios/demo.yaml")
    f.add_argument("--runs", type=int, default=1)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--log", action="store_true", help="write runs/<name>-seed<k>.jsonl")
    f.add_argument("--out", help="write summary JSON here")
    f.add_argument("--duration", type=float, help="override the scenario length (sim seconds)")
    f.add_argument("--events", help="replay a recorded session: a runs/live-*.events.jsonl file, or 'latest'")
    f.add_argument("--audio", metavar="DIR",
                   help="render sound as waveforms and write them to DIR/<name>-seed<k>/")
    f.add_argument("--audio-beams", action="store_true",
                   help="with --audio, also write each Source's beam (what a classifier gets)")
    f.add_argument("--sounds", default="data/sounds",
                   help="recordings per class: <dir>/<class>/*.wav (else synthetic)")
    f.add_argument("--classifier", metavar="MOD:ATTR",
                   help="a trained sound classifier decides the class probabilities")
    f.add_argument("--params", help="policy parameters: a JSON file, e.g. a tune run's best.json")
    f.set_defaults(fn=cmd_fast)

    lv = sub.add_parser("live", help="multi-process run with radio, agents and dashboard")
    lv.add_argument("--scenario", default="scenarios/demo.yaml")
    lv.add_argument("--backend", default="twin2d", choices=["twin2d", "webots"])
    lv.add_argument("--speed", type=float, default=1.0)
    lv.add_argument("--no-dashboard", action="store_true")
    lv.add_argument("--events", help="replay a recorded session: a file, or 'latest'")
    lv.add_argument("--duration", type=float, help="run this many sim seconds (default: scenario)")
    lv.add_argument("--array", action="append", metavar="ID=DEVICE[@ROT]",
                    help="drone ID hears through a real microphone array (see `earshot arrays`)")
    lv.add_argument("--no-webots-launch", action="store_true", help="Webots already open")
    lv.add_argument("--audio", action="store_true",
                    help="render sound as waveforms; the dashboard's Listen button plays them")
    lv.add_argument("--sounds", default="data/sounds",
                    help="recordings per class: <dir>/<class>/*.wav (else synthetic)")
    lv.add_argument("--classifier", metavar="MOD:ATTR",
                    help="a trained sound classifier decides the class probabilities")
    lv.add_argument("--no-webots-view", action="store_true",
                    help="don't draw Webots' own 3D window (the drone cameras still render)")
    lv.set_defaults(fn=cmd_live)

    ev = sub.add_parser("eval", help="metrics over seeded runs, policies side by side")
    ev.add_argument("--scenario", default="scenarios/demo.yaml",
                    help="a scenario YAML, or 'random' for a generated world per seed")
    ev.add_argument("--runs", type=int, default=10)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--policy", nargs="+", default=["earshot"],
                    choices=["earshot", "nearest", "no_perch"])
    ev.add_argument("--scheme", nargs="+", choices=["ours", "flood", "unicast3"])
    ev.add_argument("--bw", type=float, help="LoRa bandwidth in kHz (default: scenario's)")
    ev.add_argument("--out", help="write all metrics as JSON")
    ev.add_argument("--params", help="policy parameters: a JSON file, e.g. a tune run's best.json")
    ev.add_argument("--classifier", nargs="+", metavar="MOD:ATTR",
                    help="sound classifiers side by side; 'none' is the CONFUSION stand-in")
    ev.add_argument("--sounds", default="data/sounds",
                    help="recordings per class for --classifier: <dir>/<class>/*.wav")
    ev.set_defaults(fn=cmd_eval)

    tu = sub.add_parser("tune", help="CMA-ES over the policy parameters, judged on held-out seeds")
    tu.add_argument("--scenario", default="random",
                    help="a scenario YAML, or 'random' for a generated world per seed")
    tu.add_argument("--gens", type=int, default=30)
    tu.add_argument("--pop", type=int, default=12, help="candidates per generation")
    tu.add_argument("--train", type=int, default=10, help="training seeds per candidate")
    tu.add_argument("--holdout", type=int, default=20, help="held-out seeds for the final check")
    tu.add_argument("--workers", type=int, help="processes (default: all cores)")
    tu.add_argument("--sigma", type=float, default=0.25, help="CMA-ES step size in [0, 1] units")
    tu.add_argument("--out", help="output directory (default: runs/tune-<time>)")
    tu.set_defaults(fn=cmd_tune)

    ar = sub.add_parser("arrays", help="list microphone arrays (inputs with 4+ channels)")
    ar.set_defaults(fn=cmd_arrays)

    d = sub.add_parser("doctor", help="pre-flight checks")
    d.set_defaults(fn=cmd_doctor)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
