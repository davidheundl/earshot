# Pre-built declaration

Everything below existed before the hackathon day (Oct 24, 2026) and goes on the declaration slide. Everything else in the demo was built on the day. Keep this list current: add a line whenever pre-event work lands.

## Infrastructure (pre-built)

- Repository, Python 3.11 environment (uv), CLI (`earshot fast | live | doctor`).
- Radio message formats (≤ 40 B binary frames with HMAC tags), codec and tests.
- Simulated radio medium: LoRa airtime, range with shadowing, half duplex, collisions with capture, duty-cycle budget, jam zones.
- Process layout: world, radio, one process per drone, dashboard; ZeroMQ wiring.
- Lockstep fast-mode runner with fixed seeds.
- 2D twin backend (kinematic drones, perching on flat roofs, battery drain).
- Webots bridge: puppeted drones, a single camera rig with recognition, world generator from City JSON.
- Webots performance: backend paces Webots (fast mode) to the wall clock, Verifier camera rendered once per tick, drone view cameras only while a screen shows them, tree meshes tiled for culling, `--no-webots-view`.
- Placeholder synthetic city and a demo scenario.
- Munich Airport city, 1:1 from open data (`tools/muc_fetch.py`, `tools/make_muc.py`): 1,303 LoD2 buildings with their measured walls and roofs, Perch Sites on flat roofs, OSM runways, roads and the S-Bahn surface track, the aerial photo as Webots ground and dashboard basemap; scenario `scenarios/muc.yaml` (drones on the terminal roofs, intruder over the apron).
- Munich Airport look in Webots: the aerial photo draped over the LoD2 roofs, a curtain-wall facade texture on the walls, ground photo at 0.5 m/px (dashboard basemap 4096 px).
- Drone models from Webots primitives (swarm in signal orange, a larger dark intruder), closer chase and pilot cameras, view cameras with anti-aliasing and ambient occlusion (main view 854x480).
- Head-up display over the main drone view: speed and height tapes, vertical speed, horizon and pitch ladder, flight path marker, heading tape, target bracket and threat box, battery, position, REC.
- Airliners parked nose-in on the OSM stands at Munich (generated narrow- and wide-body models, generic liveries); `stands` key in muc.json.
- Dashboard skeleton (map, drones, modes, radio links, ground truth).
- Game screen motion: drones glide between states on the map and flight display (60 fps), static map layer cached.
- Message store with duplicate filter and plain hop-limited flooding (baseline).
- Sensor models: sound propagation (spherical spreading, per-class air absorption and building loss), ego-noise by motor speed with a filter gain, masking, SNR-based detection, bearing error and outliers, placeholder classifier confusion (fan mistaken for drone); passive RF energy per band (2.4 / 5.8 GHz, 915 MHz) with log-distance path loss and shadowing; camera stand-in with an error rate. Calibrated to the concept's range tables.
- Detection log with ground-truth source per measurement (for the evaluation metrics).
- Sound as waveforms (infrastructure only, no trained model): per-class clip bank (recordings in `data/sounds/`, synthetic stand-ins otherwise), per-drone 1 s mixes calibrated to the level model, per-Source beams, a classifier hook, training-data export (`earshot fast --audio`) and live playback in the dashboard (`earshot live --audio`, Listen).
- Scenario events (add source, jam, fault, day/night, ego filter): scripted in YAML or sent live from dashboard hotkeys; live sessions recorded and replayable in fast mode.
- Stub belief and duty-cycle policy.

## Prototype of the judged logic (branch `prototype`, built before the event)

Built to rehearse and measure. If any of it is used on the day, it goes on the declaration slide as pre-built; otherwise it is rebuilt on the day from `main`.

- Probability Map: bounded log-likelihood evidence per drone and sensor, effective-sample cap, rail/road priors (explaining away), camera confirmation and dismissal, track extraction with uncertainty ellipses, overlapping Suspicions merged at their likelihood-weighted mean.
- Faulty Sensor detection against what the other drones agree on.
- Decisions: one Verifier per danger, the nearest drone (Claim auction by place with distance bids, leases and settle window, a claim held while its Suspicion lasts), everyone else holding their post, Voronoi sectors with gap filling.
- Radio protocol: priority queue, contention-based relaying, Digest/Request healing, track suppression.
- Evaluation: section 17 metrics, random scenarios, baselines "nearest drone always flies" and "no perching".
- Dashboard: probability-map heatmap, Suspicions, tasks, sectors, decision log, camera panel.
- Live arrays: SRP-PHAT bearing, array sensor, fixed drones.
- Agent hot path cached between changes (merged Tracks, telemetry, summary; cheap overlap reject): no behaviour change, eval identical bit for bit.

## Third-party

- Webots R2025a (Apache 2.0). Python packages per `pyproject.toml` / `uv.lock`.
- Munich Airport map data: buildings (LoD2) and aerial photo (DOP) © Bayerische Vermessungsverwaltung, www.geodaten.bayern.de, CC BY 4.0; © OpenStreetMap contributors, ODbL. Both need this attribution on the slide.
- Planned: EfficientAT (MIT), DroneAudioset (MIT).

## Built on the day

Probability map and measurement models, explaining away, faulty-sensor detection, mode decisions, perch-site selection, verification value vs. gap cost, claim auction with leases, sector re-partitioning, relevance-limited forwarding, partition healing, evaluation over 100 runs.
