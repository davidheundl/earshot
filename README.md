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

- [docs/architecture.md](docs/architecture.md): processes, module interfaces, where each teammate plugs in.
- [CONTEXT.md](CONTEXT.md): the project's vocabulary.
- [docs/adr/](docs/adr/): decisions and why.
- [DECLARATION.md](DECLARATION.md): what was built before the event.
