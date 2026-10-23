# Earshot

Hackathon project (Oct 24, 2026): a perching drone swarm fuses sound + RF into a Probability Map, auctions who flies out, and a Verifier's camera (YOLO) confirms. Two worlds: Munich Airport (Threat Class drone) and a savanna Reserve (poachers' vehicles). Two-person team: David (agents, radio, belief, decisions), Nikolas (Webots world, LoD2/OSM city, live arrays, camera detector, dashboard).

Read before changing code:
- [docs/architecture.md](docs/architecture.md) — processes, module interfaces, gotchas.
- [docs/functionality.md](docs/functionality.md) — one second of sim time end to end, with constants and file:line.
- [CONTEXT.md](CONTEXT.md) — use its terms exactly in code, comments and messages (Perch Site, Verifier, Evidence vs. Track, ...).
- [docs/adr/](docs/adr/) — before reversing an architectural choice.

## Rules

- `earshot/agent/` stays pure: time arrives as the `t` argument, frames as `inbox`, randomness through injected seeded generators. `tests/test_lockstep.py` checks same seed → same run.
- Agents receive readings only: audio, images and ground truth stay in the world process (ADR 0001), and agents reach each other only through the radio process (ADR 0003).
- Radio frames stay ≤ 40 B; `earshot/messages.py` is their single source of truth.
- The Threat Class comes from the Scenario's `threat:` (`SwarmConfig.threat`, ADR 0009); code that means "the intruder" reads it rather than `TargetClass.DRONE`.
- Simulators sit behind the Backend seam (`earshot/backend/`): `twin2d` (fast mode, eval, tune), `webots` (the demo, and the only backend whose images feed YOLO), `gazebo` (alternative renderer, ADR 0008), picked by `--backend`. Nothing above the seam may know which one runs.
- A model that cannot load (no torch or ultralytics, missing weights) is reported and replaced by its stand-in; a run never stops for it.
- Judge behaviour changes with `uv run earshot eval` over seeds, one change at a time; a single run is noise. Tuned sets from `earshot tune` live in `scenarios/params/` and load only with `--params`; a tuned set replaces the code defaults only if it misses no more drones on held-out seeds.
- `prototype` is the working branch (it contains `main` and `integrate`). Anything new built before Oct 24 gets a line in DECLARATION.md.
- Generated Webots worlds may use EXTERNPROTOs (ADR 0005 superseded: the venue has WLAN); open each world once online so Webots caches them. Keep DEF names `DRONE_<id>`, `SRC_<id>`, `RIG`.

## Verify

`uv run pytest` and `uv run ruff check src tests tools`. The sound-classifier and camera tests skip without `uv sync --group ml --group cv` and the EfficientAT weights (`tools/fetch_efficientat.py`), so run them with those after touching `audio/` or `sensors/camera.py`. For runtime changes, also `uv run earshot fast` and a short `uv run earshot live`; one live run per Mac (ports 5550-5554, 8000, 1240; `uv run earshot doctor` checks them).
