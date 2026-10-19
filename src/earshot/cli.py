"""earshot <command>

    earshot fast   --scenario S [--runs N] [--seed K] [--log] [--events F]   lockstep, no sockets
                   [--audio DIR [--audio-beams]] [--classifier MOD:ATTR]       sound as waveforms
    earshot live   --scenario S [--backend twin2d|webots|gazebo] [--events F|latest]
                   [--no-dashboard] [--audio] [--classifier MOD:ATTR]
    earshot doctor                                                pre-flight checks (run offline)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


GZ_VENV = Path("gazebo/.venv")


def gazebo_env() -> dict:
    """gz-transport on loopback, in our own partition (server, GUI and bridge alike)."""
    env = dict(os.environ)
    env["GZ_IP"] = config.HOST
    env["GZ_PARTITION"] = config.GZ_PARTITION
    return env


def gazebo_python() -> str | None:
    """Homebrew's newest Python that has Gazebo's bindings (gz.sim8), or None."""
    prefix = Path(os.environ.get("HOMEBREW_PREFIX", "/opt/homebrew"))
    for site in sorted(prefix.glob("lib/python3.*/site-packages"),
                       key=lambda p: int(p.parent.name.split(".")[1]), reverse=True):
        exe = prefix / "bin" / site.parent.name
        if exe.exists() and list(site.glob("gz/sim8*")):
            return str(exe)
    return None


def ensure_gazebo_venv() -> str:
    """The bridge's interpreter: a venv on Homebrew's Python that sees its gz packages,
    plus gazebo/requirements.txt. Made on first use; returns its python."""
    py = GZ_VENV / "bin" / "python"
    if py.exists():
        return str(py)
    base = gazebo_python()
    if base is None:
        sys.exit("Gazebo's Python bindings not found: brew tap osrf/simulation && "
                 "brew install gz-harmonic")
    print(f"making {GZ_VENV} on {base} (once)")
    subprocess.run(["uv", "venv", str(GZ_VENV), "--python", base, "--system-site-packages",
                    "-q"], check=True)
    subprocess.run(["uv", "pip", "install", "-q", "--python", str(py), "-r",
                    "gazebo/requirements.txt"], check=True)
    return str(py)


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


def busy_ports(gazebo: bool = False) -> list[int]:
    """The local ports a live run binds that something already listens on."""
    eps = [config.WORLD_PUB, config.WORLD_PULL, config.RADIO_PULL, config.RADIO_PUB]
    eps += [config.GAZEBO_BRIDGE] if gazebo else []
    ports = [int(ep.rsplit(":", 1)[1]) for ep in eps] + [config.DASHBOARD_PORT]
    out = []
    for port in ports:
        with socket.socket() as sk:
            if sk.connect_ex((config.HOST, port)) == 0:
                out.append(port)
    return out


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
    """(sensors.Audio, sensors.export.AudioExport) as the flags and the Scenario's `audio:`
    ask, or Nones. Fast mode renders no images, so the camera stays the stand-in."""
    from earshot.sensors import build_perception
    from earshot.sensors.export import AudioExport

    audio, _ = build_perception(sc, classifier=a.classifier, sounds=a.sounds,
                                audio=bool(a.audio))
    out = None
    if a.audio and audio is not None:
        out = AudioExport(Path(a.audio) / f"{sc.name}-seed{sc.seed}", audio.mixer, a.audio_beams)
    return audio, out


# -- eval ---------------------------------------------------------------------------------


def _params(a) -> dict | None:
    """--params FILE: a JSON object of state.PolicyParams overrides (`earshot tune` writes
    best.json with them under "params")."""
    if not a.params:
        return None
    d = json.loads(Path(a.params).read_text())
    return d.get("params", d)


def _eval_seed(a) -> int:
    """--seed, else the held-out seeds when --params (a tuned set) is judged, else 0. Warns
    when a tuned set is judged on its own training seeds."""
    from earshot.tune import HOLDOUT_SEED0, TRAIN_RUNS, TRAIN_SEED0

    seed = a.seed if a.seed is not None else (HOLDOUT_SEED0 if a.params else 0)
    if a.params and seed < TRAIN_SEED0 + TRAIN_RUNS and TRAIN_SEED0 < seed + a.runs:
        print(f"warning: seeds {seed}-{seed + a.runs - 1} overlap tune's training seeds "
              f"{TRAIN_SEED0}-{TRAIN_SEED0 + TRAIN_RUNS - 1}: --params was fit to them "
              f"(held-out seeds start at {HOLDOUT_SEED0})", file=sys.stderr)
    return seed


def cmd_eval(a) -> None:
    from earshot.eval import as_json, evaluate, summarize

    if a.jobs > 1 and any(c != "none" for c in a.classifier or []):
        sys.exit("--jobs > 1 cannot share a loaded sound classifier: drop --jobs or --classifier")
    seed = _eval_seed(a)
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
        rows = evaluate(a.scenario, a.runs, seed, policy, scheme, a.bw, _params(a), audios[c],
                        a.jobs, a.duration)
        raw[label] = as_json(rows)
        table.append(summarize(rows, label))
    cols = list(table[0])
    width = {c: max(len(c), *(len(str(row[c])) for row in table)) for c in cols}
    print("  ".join(c.ljust(width[c]) for c in cols))
    for row in table:
        print("  ".join(str(row[c]).ljust(width[c]) for c in cols))
    print(f"{a.runs} run(s) (seeds {seed}-{seed + a.runs - 1}) x {len(combos)} configurations"
          f" in {time.time() - t0:.0f}s  (median [min-max] over seeds; latencies censored at"
          " the scenario's duration; confirmed: drone Sources, agreed: jammed runs)")
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


def _wait_for_webots(proc: subprocess.Popen, timeout: float = 60.0) -> None:
    """Wait until Webots has loaded the world and offers the RIG's extern-controller socket
    (a big world such as Munich takes a while), at most `timeout` seconds. Webots keeps that
    socket at /tmp/webots/<user>/<port>/ipc/<robot>/extern; if its instance folder never
    shows up while the port is open, the layout differs from what we expect and we go on
    (the controller library itself retries for a while)."""
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "default"
    instance = Path("/tmp/webots") / user / str(config.WEBOTS_PORT)
    robot = os.environ.get("WEBOTS_ROBOT_NAME") or "supervisor"  # webots_env's controller URL
    sock = instance / "ipc" / robot / "extern"
    start = time.monotonic()
    port_up: float | None = None
    print(f"[live] waiting for Webots to load {Path(proc.args[-1]).name} (up to "
          f"{timeout:.0f} s)", flush=True)
    said = 0
    while (waited := time.monotonic() - start) < timeout:
        if proc.poll() is not None:
            sys.exit(f"Webots exited with {proc.returncode} while loading the world")
        if sock.exists():
            print(f"[live] Webots ready after {waited:.0f} s", flush=True)
            return
        if port_up is None:
            with socket.socket() as sk:
                if sk.connect_ex((config.HOST, config.WEBOTS_PORT)) == 0:
                    port_up = time.monotonic()
        elif not instance.exists() and time.monotonic() - port_up > 10.0:
            print("[live] Webots is up, its controller socket not where expected; "
                  "connecting anyway", flush=True)
            return
        if waited // 10 > said:
            said = int(waited // 10)
            print(f"[live] still waiting for Webots ({waited:.0f} s)", flush=True)
        time.sleep(0.5)
    print(f"[live] Webots not ready after {timeout:.0f} s; connecting anyway (the world may "
          "fail to start: check the Webots window)", flush=True)


def _launch(a, sc, py, procs, spawn, events, arrays) -> None:
    """Start Webots (optional), radio, agents, dashboard and world; return when the world
    ends. A crashed helper is reported, not fatal (the dashboard never stops a run)."""
    taken = busy_ports(gazebo=a.backend == "gazebo")
    if taken:  # another run: our agents would steer its world
        sys.exit(f"Port(s) {', '.join(map(str, taken))} taken: another Earshot run (or a "
                 "leftover one) is live. Stop it first, e.g. pkill -f 'earshot live'")
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
            _wait_for_webots(procs[-1])

    if a.backend == "gazebo":
        world_file = Path(f"gazebo/worlds/{sc.name}.sdf")
        subprocess.run([py, "tools/build_gazebo_world.py", a.scenario, str(world_file)], check=True)
        bridge_py = ensure_gazebo_venv()
        procs.append(subprocess.Popen([bridge_py, "src/earshot/backend/gz_bridge.py",
                                       str(world_file), "--endpoint", config.GAZEBO_BRIDGE],
                                      env=gazebo_env()))
        if not a.no_gazebo_view:  # macOS: the GUI is its own process, joined over gz-transport
            procs.append(subprocess.Popen(["gz", "sim", "-g"], env=gazebo_env()))

    spawn(["earshot.runners.radio_proc", "--scenario", a.scenario])
    for d in sc.drones:
        extra = ["--array", arrays[str(d.id)]] if str(d.id) in arrays else []
        if extra and a.classifier:
            extra += ["--classifier", a.classifier]
        if a.params:
            extra += ["--params", a.params]
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
        world_args += ["--audio"]
    if a.sounds:
        world_args += ["--sounds", a.sounds]
    if a.classifier:
        world_args += ["--classifier", a.classifier]
    if a.camera:
        world_args += ["--camera", a.camera]
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

    def check(name, cond, hint="", fatal=True):
        """fatal=False: a run works without it (dev setups, optional hardware/backends)."""
        nonlocal ok
        ok &= bool(cond) or not fatal
        tag = "ok  " if cond else ("FAIL" if fatal else "warn")
        print(f"  {tag} {name}" + ("" if cond else f"  -> {hint}"))

    print("earshot doctor")
    check("python 3.11", sys.version_info[:2] == (3, 11), "uv python pin 3.11 && uv sync")
    check("Webots installed", Path(f"{WEBOTS_HOME}/Contents/MacOS/webots").exists(),
          "see README: install from the GitHub release")
    r = subprocess.run([sys.executable, "-c", "from controller import Supervisor"],
                       env=webots_env(), capture_output=True, check=False)
    check("Webots Python controller importable", r.returncode == 0, r.stderr.decode()[-200:])
    check("Gazebo installed (gz sim)", shutil.which("gz") is not None,
          "brew tap osrf/simulation && brew install gz-harmonic", fatal=False)
    check("Gazebo Python bindings (Homebrew)", gazebo_python() is not None,
          "brew install gz-harmonic", fatal=False)
    if (GZ_VENV / "bin" / "python").exists():
        r = subprocess.run([str(GZ_VENV / "bin" / "python"), "-c",
                            "import gz.sim8, gz.transport13, gz.msgs10.image_pb2, zmq, msgpack"],
                           capture_output=True, check=False)
        check("Gazebo bridge venv importable", r.returncode == 0, r.stderr.decode()[-200:],
              fatal=False)
    else:
        check("Gazebo bridge venv (made by the first `earshot live --backend gazebo`)", True)
    for name, ep in [("world pub", config.WORLD_PUB), ("world pull", config.WORLD_PULL),
                     ("radio pull", config.RADIO_PULL), ("radio pub", config.RADIO_PUB),
                     ("gazebo bridge", config.GAZEBO_BRIDGE)]:
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
          "export EARSHOT_KEY=... (fine to skip for dev)", fatal=False)
    try:
        import sounddevice as sd

        n_in = [d for d in sd.query_devices() if d["max_input_channels"] >= 4]
        check("4+ channel input device (ReSpeaker)", n_in, "plug in the array; grant mic access",
              fatal=False)
    except Exception as e:  # noqa: BLE001
        check("audio extras installed", False, f"uv sync ({e})", fatal=False)
    if not ok:
        sys.exit("some checks FAILed: fix them before a run (warn lines are fine during dev)")
    print("ready (warn lines are fine during dev)")


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
    f.add_argument("--sounds",
                   help="recordings per class: <dir>/<class>/*.wav (default: the Scenario's "
                        "audio.sounds, else data/sounds; synthetic when empty)")
    f.add_argument("--classifier", metavar="MOD:ATTR",
                   help="a trained sound classifier decides the class probabilities")
    f.add_argument("--params", help="policy parameters: a JSON file, e.g. a tune run's best.json")
    f.set_defaults(fn=cmd_fast)

    lv = sub.add_parser("live", help="multi-process run with radio, agents and dashboard")
    lv.add_argument("--scenario", default="scenarios/demo.yaml")
    lv.add_argument("--backend", default="twin2d", choices=["twin2d", "webots", "gazebo"])
    lv.add_argument("--speed", type=float, default=1.0)
    lv.add_argument("--no-dashboard", action="store_true")
    lv.add_argument("--events", help="replay a recorded session: a file, or 'latest'")
    lv.add_argument("--duration", type=float,
                    help="stop after this many sim seconds (default: run until Ctrl-C)")
    lv.add_argument("--array", action="append", metavar="ID=DEVICE[@ROT]",
                    help="drone ID hears through a real microphone array (see `earshot arrays`)")
    lv.add_argument("--no-webots-launch", action="store_true", help="Webots already open")
    lv.add_argument("--audio", action="store_true",
                    help="render sound as waveforms; the dashboard's Listen button plays them")
    lv.add_argument("--sounds",
                    help="recordings per class: <dir>/<class>/*.wav (default: the Scenario's "
                         "audio.sounds, else data/sounds; synthetic when empty)")
    lv.add_argument("--classifier", metavar="MOD:ATTR",
                    help="a trained sound classifier decides the class probabilities "
                         "(default: the Scenario's audio.classifier; 'none' for the stand-in)")
    lv.add_argument("--camera", choices=["yolo", "truth"],
                    help="Verifier camera: the trained detector or the ground-truth stand-in "
                         "(default: the Scenario's camera.detector; yolo needs --backend webots)")
    lv.add_argument("--params", help="policy parameters: a JSON file, e.g. a tune run's best.json")
    lv.add_argument("--no-webots-view", action="store_true",
                    help="don't draw Webots' own 3D window (the drone cameras still render)")
    lv.add_argument("--no-gazebo-view", action="store_true",
                    help="don't open Gazebo's window (the drone cameras still render)")
    lv.set_defaults(fn=cmd_live)

    ev = sub.add_parser("eval", help="metrics over seeded runs, policies side by side")
    ev.add_argument("--scenario", default="scenarios/demo.yaml",
                    help="a scenario YAML, or 'random' for a generated world per seed")
    ev.add_argument("--runs", type=int, default=10)
    ev.add_argument("--seed", type=int,
                    help="first seed (default: tune's held-out seeds, 1000, with --params; "
                         "else 0); warns if --params is judged on tune's training seeds")
    ev.add_argument("--jobs", type=int, default=1,
                    help="processes; same rows as serial (not with --classifier)")
    ev.add_argument("--duration", type=float, help="override the scenario length (sim seconds)")
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
