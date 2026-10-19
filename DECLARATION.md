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
- Webots performance: backend paces Webots (fast mode) to the wall clock, Verifier camera rendered once per tick, drone view cameras only while a screen shows them, tree meshes tiled for culling, `--no-webots-view`; detail falls off with distance from the City (trees outside it without trunks and with coarser crowns, villages 2 km out as LoD1 blocks), parked cars without separate tyres, fence posts every 8 m.
- Placeholder synthetic city and a demo scenario.
- Munich Airport city, 1:1 from open data (`tools/muc_fetch.py`, `tools/make_muc.py`): 1,303 LoD2 buildings with their measured walls and roofs, Perch Sites on flat roofs, OSM runways, roads and the S-Bahn surface track, the aerial photo as Webots ground and dashboard basemap; scenario `scenarios/muc.yaml` (drones on the terminal roofs, intruder over the apron).
- Munich Airport look in Webots: the aerial photo draped over the LoD2 roofs, a curtain-wall facade texture on the walls, ground photo at 0.5 m/px (dashboard basemap 4096 px).
- Drone models from Webots primitives (swarm in signal orange, a larger dark intruder), closer chase and pilot cameras, view cameras with anti-aliasing and ambient occlusion (main view 854x480).
- Head-up display over the main drone view: speed and height tapes, vertical speed, horizon and pitch ladder, flight path marker, heading tape, target bracket and threat box, battery, position, REC.
- Airliners parked nose-in on the OSM stands at Munich (generated narrow- and wide-body models, generic liveries); `stands` key in muc.json.
- Munich surroundings: 4 km of aerial photo, LoD2 villages, forests and power lines around the map; 3D trees, tree rows, hedges, fences, walls, floodlight and radio masts, cars on the surface and street-side car parks, jet bridges and ground equipment (tugs, belt loaders, baggage carts, fuel trucks) at the parked aircraft, with the flat aircraft and cars cleaned out of the photo underneath.
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
- Policy tuning: the decision thresholds as `PolicyParams` (scenario `params:`, `--params`), `earshot tune` (CMA-ES over seeded random worlds, paired check on held-out seeds); a first tuned set (`scenarios/params/tuned-2026-10-18.json`, notes in `docs/tune-2026-10-18.md`) and the `two_drones` scenario.
- Agent hot path cached between changes (merged Tracks, telemetry, summary; cheap overlap reject): no behaviour change, eval identical bit for bit.
- Sound classifier spike (`src/earshot/audio/classifier.py`, `tools/*sound*`, results in `docs/sound-classifier-spike.md`): EfficientAT mn10 (AudioSet-pretrained, vendored, frozen) with a small trained head over fine classes (drone, plane, helicopter, vehicle, train, machine, insect, ambient, other) folded onto the 4 Target Classes; a zero-shot AudioSet baseline; `earshot eval --classifier` side by side, `--classifier` for the live arrays. Trained on ESC-50 and DroneAudioSet, tested on ESC-50 fold 5 and a drone the head never heard; the trained head (`data/models/sound_head.pt`) is committed. Built to judge whether it is worth rebuilding and retraining on the day.
- Second-drone blind spot: a confirmed drone follows one map peak per update at most 15 m/s (no hopping from peak to peak), other peaks beside it are Explained side lobes, Confirmed Tracks merge only with Confirmed ones; `scenario_gen` adds a second intruder 250-500 m from the first in 30 % of random worlds (own random stream, other worlds unchanged).
- Second tuned policy set trained with two-intruder worlds (`scenarios/params/tuned-2026-10-18b.json`, notes in `docs/tune-2026-10-18b.md`): fewer missed drones on held-out random worlds, worse on `two_drones.yaml`'s late RF-only second drone.

## Third-party

- Webots R2025a (Apache 2.0). Python packages per `pyproject.toml` / `uv.lock`; `cma` (pycma, BSD-3) for `earshot tune`.
- Munich Airport map data: buildings (LoD2) and aerial photo (DOP) © Bayerische Vermessungsverwaltung, www.geodaten.bayern.de, CC BY 4.0; © OpenStreetMap contributors, ODbL. Both need this attribution on the slide.
- EfficientAT (MIT, github.com/fschmid56/EfficientAT): model code vendored in `src/earshot/audio/efficientat/`, mn10_as AudioSet weights downloaded; AudioSet label names (CC BY 4.0, Google). `torch`/`torchvision` (BSD) in the optional `ml` group.
- Sound datasets, downloaded, not redistributed: DroneAudioSet (MIT, Gupta et al., NeurIPS 2025, huggingface.co/datasets/ahlab-drone-project/DroneAudioSet), ESC-50 (CC BY-NC 3.0, K. J. Piczak, github.com/karolpiczak/ESC-50; clips from freesound.org).

## Built on the day

Probability map and measurement models, explaining away, faulty-sensor detection, mode decisions, perch-site selection, verification value vs. gap cost, claim auction with leases, sector re-partitioning, relevance-limited forwarding, partition healing, evaluation over 100 runs.
