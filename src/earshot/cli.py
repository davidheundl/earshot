"""earshot <command>

    earshot fast   --scenario S [--runs N] [--seed K] [--log] [--events F]   lockstep, no sockets
    earshot live   --scenario S [--backend twin2d|webots] [--events F|latest] [--no-dashboard]
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
        if a.duration:
            sc.duration = a.duration
        if a.events:  # replay a recorded live session on top of the scenario
            recorded = [json.loads(line) for line in events_file.read_text().splitlines()]
            sc.events = sorted(sc.events + recorded, key=lambda e: e["t"])
        log = None
        if a.log:
            log = Path("runs") / f"{sc.name}-seed{sc.seed}.jsonl"
            log.parent.mkdir(exist_ok=True)
        r = run(sc, log=log)
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
    out = subprocess.run(["pgrep", "-f", f"webots --port={config.WEBOTS_PORT}"],
                         capture_output=True, text=True, check=False).stdout
    return out.split()


def cmd_arrays(_a) -> None:
    from earshot.audio.array import list_arrays

    found = list_arrays()
    for i, name, ch in found:
        print(f"{i}: {name} ({ch} channels)")
    if not found:
        print("no input with 4+ channels: plug in the ReSpeaker, use the 6-channel firmware")


# -- eval ---------------------------------------------------------------------------------


def cmd_eval(a) -> None:
    from earshot.eval import as_json, evaluate, summarize

    t0 = time.time()
    table, raw = [], {}
    combos = [(p, s) for p in a.policy for s in (a.scheme or [None])]
    for policy, scheme in combos:
        label = policy + (f"/{scheme}" if scheme else "") + (f"/{a.bw:g}kHz" if a.bw else "")
        rows = evaluate(a.scenario, a.runs, a.seed, policy, scheme, a.bw)
        raw[label] = as_json(rows)
        table.append(summarize(rows, label))
    cols = list(table[0])
    width = {c: max(len(c), *(len(str(row[c])) for row in table)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    for row in table:
        print("  ".join(str(row[c]).ljust(width[c]) for c in cols))
    print(f"{a.runs} run(s) x {len(a.policy)} policies in {time.time() - t0:.0f}s"
          "  (median [min-max] over seeds)")
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": table, "runs": raw}, indent=1))


# -- live ---------------------------------------------------------------------------------


def cmd_live(a) -> None:
    from earshot.scenario import Scenario

    sc = Scenario.load(a.scenario)
    py = sys.executable
    procs: list[subprocess.Popen] = []

    def spawn(args, env=None):
        procs.append(subprocess.Popen([py, "-m", *args], env=env))

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
    arrays = dict(x.split("=", 1) for x in (a.array or []))
    for d in sc.drones:
        extra = ["--array", arrays[str(d.id)]] if str(d.id) in arrays else []
        spawn(["earshot.runners.agent_proc", "--id", str(d.id), "--scenario", a.scenario, *extra])
    if not a.no_dashboard:
        spawn(["earshot.dashboard.server", "--scenario", a.scenario])
        print(f"dashboard: http://{config.HOST}:{config.DASHBOARD_PORT}")
    time.sleep(1.0)
    world_args = ["earshot.runners.world_proc", "--scenario", a.scenario, "--backend", a.backend,
                  "--speed", str(a.speed)]
    if a.events:
        world_args += ["--events", str(_resolve_events(a.events))]
    if a.duration:
        world_args += ["--duration", str(a.duration)]
    spawn(world_args, env=webots_env() if a.backend == "webots" else None)

    def _stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _stop)  # explicit: background jobs inherit SIGINT=ignore
    signal.signal(signal.SIGTERM, _stop)
    try:
        procs[-1].wait()  # the world ends the run
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
    ev.set_defaults(fn=cmd_eval)

    ar = sub.add_parser("arrays", help="list microphone arrays (inputs with 4+ channels)")
    ar.set_defaults(fn=cmd_arrays)

    d = sub.add_parser("doctor", help="pre-flight checks")
    d.set_defaults(fn=cmd_doctor)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
