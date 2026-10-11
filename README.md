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

Live with Webots (opens its own Webots instance on port 1240):

```bash
uv run earshot live --backend webots
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
