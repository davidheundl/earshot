# Earshot

A listening swarm: cheap drones perch on flat roofs to hear, fuse sound and RF energy into a shared probability map of threats, and decide among themselves who flies out to verify by camera. SF Hackathon, Oct 24, 2026.

## Setup (macOS, Apple Silicon)

1. Install [uv](https://docs.astral.sh/uv/) and Webots R2025a. The Homebrew cask is disabled, so download `webots-R2025a.dmg` from the [GitHub release](https://github.com/cyberbotics/webots/releases/tag/R2025a) and drag Webots.app into /Applications.
2. Install the environment and run the pre-flight check:

```bash
uv sync
```

```bash
uv run earshot doctor
```

## Run

Fast mode: one process, no Webots, reproducible. Use it for tests and the 100-run evaluation.

```bash
uv run earshot fast --runs 10 --log
```

Live on the 2D twin (dashboard at http://127.0.0.1:8000):

```bash
uv run earshot live
```

In the dashboard, keys act at the mouse position: `d` drone, `v` vehicle, `t` train, `n` fan, `j`/`J` jam / clear, `f`/`F` sensor fault / clear, `b` day/night, `e` ego-noise filter. Each live session is recorded to `runs/live-*.events.jsonl`; replay the newest one in fast mode (or pass a specific file):

```bash
uv run earshot fast --events latest
```

To watch a recorded session instead, replay it live (add `--backend webots` to see it in Webots):

```bash
uv run earshot live --backend webots --events latest
```

Live with Webots (opens its own Webots instance on port 1240):

```bash
uv run earshot live --backend webots
```

When the dashboard is the show, skip Webots' own 3D window (the drone cameras still render); on an M3 this keeps the valley at real time:

```bash
uv run earshot live --backend webots --no-webots-view
```

Live with Gazebo Sim instead of Webots (ADR 0008): the same swarm behind the same Backend seam, switched by `--backend`. Install Gazebo Harmonic once: `brew tap osrf/simulation && brew install gz-harmonic`. The first run makes the bridge's venv in `gazebo/.venv`, then:

```bash
uv run earshot live --backend gazebo
```

`--no-gazebo-view` skips Gazebo's own window (the drone cameras still render). `uv run earshot doctor` checks the install.

Munich Airport, 1:1 from open data (LoD2 buildings, OSM runways and rail, the aerial photo as ground). The valley stays as the backup map:

```bash
uv run earshot live --scenario scenarios/muc.yaml --backend webots
```

Munich with the full perception (`scenarios/muc_full.yaml`): the trained sound classifier (EfficientAT) and a YOLO drone detector on the Verifier camera. Install the model groups (`ml`: torch for sound, `cv`: ultralytics for the camera) and fetch the assets that are not committed, the sound backbone and the recordings:

```bash
uv sync --group ml --group cv
uv run python tools/fetch_efficientat.py
uv run python tools/fetch_sound_data.py
```

On the 2D twin (sound classifier on, the camera stays the ground-truth stand-in):

```bash
uv run earshot live --scenario scenarios/muc_full.yaml
```

In Webots, YOLO on a 1280x768 Verifier view (the first world build takes about 3 min). A missing model falls back to its stand-in with a warning; `--camera truth` and `--classifier none` turn them off:

```bash
uv run earshot live --scenario scenarios/muc_full.yaml --backend webots
```

The airport files in `data/cities/` are committed. To rebuild them from the open data (needs internet once; downloads ~850 MB into `data/raw/muc/`, the surroundings included):

```bash
uv run python tools/muc_fetch.py
```

```bash
uv run python tools/make_muc.py
```

Anti-poaching in a 10 x 10 km game reserve in the South African lowveld (`scenarios/savanna.yaml`, [docs/savanna.md](docs/savanna.md)): twenty drones on lookout towers hunt vehicles instead of drones (`threat: vehicle`). Poachers drive cross-country to a rhino and an elephant herd, rangers and a game drive keep to the tracks, the game sounds like engines half the time; a Verifier that confirms a poacher follows it and the Threat Log records what its camera saw and where the poacher went (`runs/*.threats.jsonl`, and on the dashboard):

```bash
uv run earshot live --scenario scenarios/savanna.yaml
```

In Webots (bushveld, acacias, the game as herds from primitives, Webots' Range Rover for the vehicles: fetched once on first open). The Verifier camera is the stand-in: COCO YOLO sees the Range Rovers as cars only 50-160 m ahead and missed the moving poacher in a live run ([docs/savanna.md](docs/savanna.md#the-verifier-camera-in-webots-what-coco-yolo-sees); its settings are a commented block in the scenario):

```bash
uv run earshot live --scenario scenarios/savanna.yaml --backend webots --no-webots-view
```

The reserve is generated, deterministic: `uv run python tools/make_savanna.py` rewrites `data/cities/savanna.json`.

Evaluation over seeded runs, policies side by side (concept section 17). `--scenario random` generates a new world per seed; `--scheme ours flood unicast3` and `--bw 125` compare radio set-ups:

```bash
uv run earshot eval --scenario random --runs 30 --policy earshot nearest no_perch
```

Microphone arrays in the room (drones 0 and 1 in `scenarios/room.yaml`):

```bash
uv run earshot arrays
```

```bash
uv run earshot live --scenario scenarios/room.yaml --array 0=DEVICE@ROTATION --array 1=DEVICE@ROTATION
```

Tests:

```bash
uv run pytest
```

## Docs

- [docs/functionality.md](docs/functionality.md): how the system works end to end (sensors, Probability Map, policy, radio, evaluation, how to run Munich), limitations, licences.
- [docs/architecture.md](docs/architecture.md): processes, module interfaces, where each teammate plugs in.
- [docs/savanna.md](docs/savanna.md): the second world, anti-poaching in a game reserve: what changes when the swarm hunts vehicles, results, limits.
- [CONTEXT.md](CONTEXT.md): the project's vocabulary.
- [docs/adr/](docs/adr/): decisions and why.
- [DECLARATION.md](DECLARATION.md): what was built before the event.
