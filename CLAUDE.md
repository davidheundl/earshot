# Earshot

Hackathon project (Oct 24, 2026): perching drone swarm that fuses sound + RF into a probability map. Two-person team: David (agents, radio, belief, decisions), Nikolas (Webots world, LoD2/OSM city, live arrays, camera detector, dashboard).

Read before changing code:
- [docs/architecture.md](docs/architecture.md) — processes, module interfaces, gotchas.
- [CONTEXT.md](CONTEXT.md) — use its terms exactly in code, comments and messages (Perch Site, Verifier, Evidence vs. Track, ...).
- [docs/adr/](docs/adr/) — before reversing an architectural choice.

## Rules

- `earshot/agent/` stays pure: time arrives as the `t` argument, frames as `inbox`, randomness through injected seeded generators. `tests/test_lockstep.py` checks same seed → same run.
- Radio frames stay ≤ 40 B; `earshot/messages.py` is their single source of truth.
- `main` holds the working version of everything, judged logic included, built before the event; anything new built before Oct 24 gets a line in DECLARATION.md.
- Simulators sit behind the Backend seam (`earshot/backend/`): `twin2d`, `webots`, `gazebo`, picked by `--backend`. Nothing above the seam may know which one runs.
- Judge behaviour changes with `uv run earshot eval` over seeds, one change at a time; a single run is noise.
- Generated Webots worlds use base nodes only (ADR 0005); keep DEF names `DRONE_<id>`, `SRC_<id>`, `RIG`.

## Verify

`uv run pytest` and `uv run ruff check src tests tools`. For runtime changes, also `uv run earshot fast` and a short `uv run earshot live`.
