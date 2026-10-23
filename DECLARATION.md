# Pre-built declaration

Everything below existed before the hackathon day (Oct 24, 2026) and goes on the declaration slide. Everything else in the demo was built on the day. Keep this list current: add a line whenever pre-event work lands.

## Infrastructure (pre-built)

- Offline visual localization (DINOv2 retrieval, SIFT/PnP with a reference RGB-D map)
  and Depth Anything V2 outdoor metric-depth/proximity demos, with held-out simulator
  query poses and aligned depth truth for evaluation; no flight-control integration.
  Localization also includes iterative pose hypotheses, cross-reference consensus and
  ambiguity rejection, with a separate withheld-reference evaluation split.

- Offline CV demos and controlled Webots Verifier datasets: pretrained drone detection,
  resolution comparison and moving-sequence tracking; demo verification requires repeated
  same-track detections and returns INCONCLUSIVE on misses (Nikolas, `tools/cv_demos.py`,
  `tools/cv_navigation.py`, `data/cv/`). Since Oct 18 the detector is also in the loop: see
  "Integration" below.

- Repository, Python 3.11 environment (uv), CLI (`earshot fast | live | doctor`).
- Radio message formats (≤ 40 B binary frames with HMAC tags), codec and tests.
- Simulated radio medium: LoRa airtime, range with shadowing, half duplex, collisions with capture, duty-cycle budget, jam zones.
- Process layout: world, radio, one process per drone, dashboard; ZeroMQ wiring.
- Lockstep fast-mode runner with fixed seeds.
- 2D twin backend (kinematic drones, perching on flat roofs, battery drain).
- Webots bridge: puppeted drones, a single camera rig with recognition, world generator from City JSON.
- Webots performance: backend paces Webots (fast mode) to the wall clock, Verifier camera rendered once per tick, drone view cameras only while a screen shows them, tree meshes tiled for culling, `--no-webots-view`; detail falls off with distance from the City (trees outside it without trunks and with coarser crowns, villages 2 km out as LoD1 blocks), parked cars without separate tyres, fence posts every 8 m.
- Gazebo Sim backend, interchangeable with Webots behind the Backend seam: world generator from City JSON (SDF, single-mesh bodies), a bridge process embedding the Gazebo server in lockstep, a bounding-box camera with class labels as the Verifier's detector, view cameras for the game screen, `earshot live --backend gazebo`.
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

## Prototype of the judged logic (built before the event, on `main` and `integrate`)

Built to rehearse and measure, and to learn what to rebuild better on the day. Whatever of it runs in the demo goes on the declaration slide as pre-built; what is rebuilt on the day goes under "Built on the day".

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

## Integration (Oct 18-19, branch `integrate`)

- Branches merged: `prototype` (the swarm), `main` (Gazebo backend), `feat/cv-applications` (Nikolas' offline CV).
- Perception in the loop: the trained EfficientAT sound head decides each beam's class, and a YOLO Verifier camera detector (`src/earshot/sensors/camera.py`, Nikolas' drone-yolo-v1 weights, ByteTrack streaks, "nothing" only after empty windows while holding still) reads the rendered Webots Verifier view at 1280x768; both chosen per Scenario (`audio:`, `camera:`, `build_perception`), with stand-ins when a model cannot load; `scenarios/muc_full.yaml`.
- Fixes from the pre-event review (`docs/review-2026-10-18.md`): censored eval metrics per drone Source, `eval --jobs`; radio shadowing drawn once per frame and receiver, summed interference; Confirmed only from a camera, capped camera evidence, Tracks per origin, bounded task drift, Claim re-sent within settle, outbid cooldown, `verify_timeout`; wrap-safe message store; live world outlives its script, Webots-quit fallback to the twin, deduplicated dashboard counters, map-first HUD without cameras; the Webots world cache stamp.
- Docs: `docs/functionality.md` (end to end), `docs/architecture.md` updated.

## Second world: anti-poaching in a game reserve (Oct 20, branch `savanna`)

- A 10 x 10 km bushveld reserve, generated (`tools/make_savanna.py`, `data/cities/savanna.json`): terrain with a sand river, granite kopjes and waterholes, grass, thornveld, thickets and riverine forest, dirt tracks and a fence road, a ranger camp, a lodge, outposts, 20 lookout towers as Perch Sites.
- Its Webots look (`tools/build_webots_world.py`, `biome: savanna`): golden grass and red tracks, flat-crowned acacias, shrubs, riverine trees and boulders, steel lookout towers, herds of elephants, giraffes, rhinos, buffalo, zebra, impala, lions and hyenas from primitives, Webots' Range Rover for poachers, rangers and the game drive (Oct 21; pickups from primitives before).
- `scenarios/savanna.yaml`: 20 drones, two poacher vehicles (one with a UHF handheld), a ranger patrol and a game drive on the tracks, eight kinds of game as Nuisances.
- Judged logic, generalised: the Threat Class per Scenario (`threat:`; the swarm hunts vehicles off the tracks in a reserve; ADR 0009), the UHF band (handhelds, push-to-talk) ringing the Vehicle class, Follow (a Verifier stays with what it confirmed, `follow_for`), Explained Tracks cheap on air and kept out of the Suspicions' places, a vehicle Explained by a road through its ellipse; `intruder:` and `kind:` per Source; the camera stand-in names what it saw and knows the reserve's own vehicles; the YOLO detector maps any model's labels to Target Classes (COCO wired for the reserve, but not used: it sees the primitive pickups as "bird"); for twenty drones, `max_verify_distance` (no bids on far tasks) and relevance-limited relaying (`radio.relay_radius_m`), both off by default; healing only Claims and Tracks that are not Explained.
- Follow with a Quarry (Oct 21, branch `follow-pursuit`): the Follower's own camera hits and the Confirmed Track fused into a position and velocity (`agent/follow.py`); it faces the Quarry and adapts its speed (`Command.speed`, `follow_gap`, `follow_gain`, `follow_max_speed`, `Kinematics.max_speed`); the camera ranges a drone by its box size for the Follower only (`SensorReading.range`, `camera.box_range`); Follow on at the airport (`muc.yaml`, `muc_full.yaml`: `follow_for` 300). Guide (Oct 22): a demo mock, `scenarios/muc_demo.yaml` (`guide: true`, `Sensed.guide`): after the honest detection and dispatch, the Verifier is told where the intruder really is and flies straight to it.
- Threat Log (`src/earshot/threat_log.py`): per checked danger what the camera recognised, when, which Verifier, the followed path; `runs/*.threats.jsonl` and a dashboard panel, the dashboard reading the Threat Class from the run.
- Performance measurement (Oct 21): `earshot live --profile` (`src/earshot/runners/profile.py`: where each world tick's wall time goes, Webots step and pacing split out) and `tools/measure_live.py` (tick jitter, real-time factor, camera rates, GPU and CPU of a live run); results in `docs/perf-2026-10-21.md`. Timing only, no change to a run.
- Live Webots speed-up (Oct 21, `src/earshot/backend/webots.py`, `tools/build_webots_world.py`): the last basic step of a tick runs in Webots while the world works (`stepBegin`/`stepEnd`, camera data copied before the step opens), pose fields written only when they change, `basicTimeStep` 50 instead of 20, view JPEGs encoded on a thread of their own, one view camera instead of four (the HUD hides the empty tiles); savanna cut to ten drones. Before/after in `docs/perf-2026-10-21.md` (0.72x -> 1.00x real time), measured on the Webots R2025a nightly of 21 Oct 2026 on macOS 27.0.1 (the release does not start there, Webots issue #6976).
- One Observation per drone and sensor window (Oct 21): the readings a drone takes in one 1 s window share a 40 B frame, up to two (time, position and Mode once, 8 B per reading), each still fused exactly once; the reserve's channel load 68 -> 63.5 %.
- Shots (Oct 21): a rifle fired is a `shot` Event every tower within a few km hears once, with a sharp bearing (`Sensor.SHOT`, `acoustics.SHOT`), Evidence for the Threat Class in its own Belief family; a Source may stop on its path (`[x, y, s]`). poacher2 stops at its kill and fires: found on 4 of 4 seeds instead of none (`docs/savanna.md`).
- The reserve's Verifier camera in Webots (Oct 21), measured on Webots renders and two live runs (`data/cv/savanna-vehicles-20261021/`, docs/savanna.md): the camera was tilted for a drone and looked over a vehicle at its standoff, and COCO YOLO11n read the primitive pickups as airplane or chair. Now a detector's camera aims at the hunted class's height at the standoff, YOLO hits are geolocated from the box's bottom edge and the camera's height, the reserve's vehicles are Webots' Range Rover (a car to COCO at 50-160 m), and `EARSHOT_CV_DUMP` saves what the detector saw. COCO YOLO11s, downloaded for a comparison (not committed), did no better. The reserve keeps the stand-in: live, YOLO cleared all four checks near the moving poacher; its measured settings (standoff 120 m among them) are a commented block in `scenarios/savanna.yaml`.
- The HUD's camera boxes (Oct 22): the main view camera has its own Webots Recognition, and each frame carries the boxes of the Sources it recognises, so the dashboard frames the intruder in the same image Webots does instead of placing it from the Track; the Track's bracket now uses the camera's own (smoothed) yaw.

## Third-party

- Webots R2025a (Apache 2.0); the reserve world uses its Range Rover Sport SVR model (Webots assets licence, cyberbotics.com/webots_assets_license). COCO YOLO11n (`yolo11n.pt`, Ultralytics, AGPL-3.0) for the reserve's Verifier camera in Webots; YOLO11s (same source and licence) only for an offline comparison, not in the repository. Python packages per `pyproject.toml` / `uv.lock`; `cma` (pycma, BSD-3) for `earshot tune`.
- Munich Airport map data: buildings (LoD2) and aerial photo (DOP) © Bayerische Vermessungsverwaltung, www.geodaten.bayern.de, CC BY 4.0; © OpenStreetMap contributors, ODbL. Both need this attribution on the slide.
- EfficientAT (MIT, github.com/fschmid56/EfficientAT): model code vendored in `src/earshot/audio/efficientat/`, mn10_as AudioSet weights downloaded; AudioSet label names (CC BY 4.0, Google). `torch`/`torchvision` (BSD) in the optional `ml` group.
- Camera detector: drone-yolo-v1 (TomSmail, Hugging Face, CC BY 4.0) via `ultralytics` (AGPL-3.0, optional `cv` group; a closed product needs an Enterprise licence or another detector). Offline CV: DINOv2-small (Apache-2.0), Depth Anything V2 Metric Outdoor Small (Apache-2.0), OpenCV, transformers (`cv-navigation` group).
- Sound datasets, downloaded, not redistributed: DroneAudioSet (MIT, Gupta et al., NeurIPS 2025, huggingface.co/datasets/ahlab-drone-project/DroneAudioSet), ESC-50 (CC BY-NC 3.0, K. J. Piczak, github.com/karolpiczak/ESC-50; clips from freesound.org).

## Built on the day

To fill in on Oct 24 with what was really rebuilt or added that day, each with its before/after `earshot eval`. Not yet built anywhere, so candidates that are genuinely new: perch-site selection by expected information (crossing angle, SNR, flight and Gap cost), position uncertainty (flying drones drift, perched ones are surveyed; bearings weighted by it), a corroboration gate that lets fix A of the review (second drone keeps its Track) back in without dispatching Ghosts, verification value vs. Gap cost. (Relevance-limited forwarding was built on Oct 20 for the reserve, see above.)
