# Pre-built declaration

Everything below existed before the hackathon day (Oct 24, 2026) and goes on the declaration slide. Everything else in the demo was built on the day. Keep this list current: add a line whenever pre-event work lands.

## Infrastructure (pre-built)

- Repository, Python 3.11 environment (uv), CLI (`earshot fast | live | doctor`).
- Radio message formats (≤ 40 B binary frames with HMAC tags), codec and tests.
- Simulated radio medium: LoRa airtime, range with shadowing, half duplex, collisions with capture, duty-cycle budget, jam zones.
- Process layout: world, radio, one process per drone, dashboard; ZeroMQ wiring.
- Lockstep fast-mode runner with fixed seeds.
- 2D twin backend (kinematic drones, perching on flat roofs, battery drain).
- Webots bridge: puppeted drones, a single camera rig with segmentation, world generator from City JSON.
- Placeholder synthetic city and a demo scenario.
- Dashboard skeleton (map, drones, modes, radio links, ground truth).
- Message store with duplicate filter and plain hop-limited flooding (baseline).
- Placeholder sensor models (fixed hearing ranges); stub belief and duty-cycle policy.

## Third-party

- Webots R2025a (Apache 2.0). Python packages per `pyproject.toml` / `uv.lock`.
- Planned: EfficientAT (MIT), DroneAudioset (MIT), Bavarian LoD2 (check licence), OpenStreetMap (ODbL).

## Built on the day

Probability map and measurement models, explaining away, faulty-sensor detection, mode decisions, perch-site selection, verification value vs. gap cost, claim auction with leases, sector re-partitioning, relevance-limited forwarding, partition healing, evaluation over 100 runs.
