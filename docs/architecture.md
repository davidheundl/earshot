# Architecture

Vocabulary: [CONTEXT.md](../CONTEXT.md). Decisions: [docs/adr/](adr/). Concept: section numbers below refer to the full concept note (Horchschwarm / Earshot).

## Processes (live mode)

```
                       ┌──────────────────────────────┐
                       │ Webots (optional, port 1240)  │
                       │ DEF DRONE_<id>, SRC_<id>, RIG │
                       └──────────────▲───────────────┘
                                      │ Supervisor API (extern controller)
┌─────────────────────────────────────┴───────────────────────────────────┐
│ world_proc   clock · Backend (Twin2D | Webots) · Scenario truth · sensors│
│   PUB :5550  "tick.<id>." Sensed → agent <id>     "state." → radio, dash │
│   PULL :5551 ← command + telemetry from agents, Events from the dashboard │
└──────────▲───────────────────────────┬──────────────────────┬───────────┘
           │ command                   │ tick                 │ state (positions, t, jam)
┌──────────┴───────────┐     ┌─────────▼──────────┐   ┌───────▼────────────┐
│ agent_proc × 6        │     │ (same agents)      │   │ radio_proc         │
│ pure Agent.step(...)  │────▶│ PUSH :5552 frames  │──▶│ RadioMedium        │
│                       │◀────────────────────────────│ PUB :5553 "rx.<id>."│
└───────────────────────┘                             │       "radio." stats│
                                                      └───────┬────────────┘
                     ┌────────────────────────────────────────▼───────────┐
                     │ dashboard  FastAPI :8000, WebSocket /ws, read-only  │
                     └────────────────────────────────────────────────────┘
```

`earshot live` starts all of these and stops them together. `earshot fast` runs the same World, RadioMedium and Agents in one process in lockstep, with no sockets (ADR 0001).

## Modules and their seams

| Module | Interface | Notes |
|---|---|---|
| `messages` | `encode(msg, key) -> bytes`, `decode(frame, key) -> msg` | The radio contract, ≤ 40 B per frame (ADR 0004). |
| `city` | `City.load(json)`, `buildings_between`, `distance_to_rail/road`, `grid` | Static prior. Schema in the module docstring. |
| `scenario` | `Scenario.load(yaml, seed)`, `active_sources(t)`, `jam_zones(t)`, `faults_for`, `apply(op, t)` | The only place ground truth lives. Every Event (YAML, hotkey, replay) goes through `apply`; op list in the module docstring. |
| `backend` | `step(dt, commands) -> states`, `camera_frame(id)`, `close()` | Adapters: `twin2d.Twin2D`, `webots.WebotsBackend` (ADR 0002). |
| `sensors` | `SensorSuite.sense(state, t, frame) -> [Detection]` | Physics in `acoustics.py` / `rf.py` (pure), calibrated to concept 5.1/5.2 by `tests/test_sensors.py`. A Detection = reading + truth; truth never reaches Agents. With a rendered frame (Webots), the recogniser decides what the camera sees. |
| `sensors.waveform` | `ClipBank(dir)`, `Mixer.window(...) -> AudioWindow`, `Mixer.beam(win, i)`, `load_classifier("mod:attr")` | Off unless asked (`--audio`, `--classifier`). Each 1 s sound window rendered as 16 kHz audio: recordings from `data/sounds/<drone\|vehicle\|train\|nothing\|ego\|background>/*.wav`, synthetic loops for classes without any; every part scaled to the level `acoustics` computes, so waveform SNR = logged SNR. A classifier (`classify(audio, sr) -> 4 probs` by TargetClass) gets one *beam* per detected Source (it + the others 6 dB down + noise), not the raw mix, and replaces the `CONFUSION` stand-in. Draws no random numbers: with audio on, everything else in the run stays the same (a classifier aside). `sensors.export.AudioExport` writes training data (`earshot fast --audio DIR [--audio-beams]`): `drone<k>.wav`, `labels.jsonl`, `beams/`, `meta.json`. |
| `world` | `World.step(commands) -> {id: Sensed}`, `detections`, `audio`, `snapshot()` | Backend + truth + sensors. Runners own the clock. `detections` feeds logs and metrics. |
| `radio.medium` | `RadioMedium.step(t, positions, outgoing, jam) -> {rx: [(frame, margin_db)]}`, `stats()` | LoRa airtime, shadowing, half duplex, capture, duty cycle, per-sender priority queue (Claim > Request > Track > Observation > Digest). |
| `agent` | `Agent.step(t, sensed, inbox) -> StepOut` | Pure. Composes `MessageStore`, `Belief`, `Policy`. |
| `agent.belief` | `add_evidence`, `tracks(t)`, `check_faults(t)`, `posterior()` | Probability Map: bounded log-likelihood per (origin, sensor family), at most N_EFF readings' worth each (ADR 0006); rail/road priors explain trains and trucks; camera confirms or dismisses; overlapping drone Suspicions merge at their likelihood-weighted mean (`merge_overlapping`, also applied to received Tracks); Faulty Sensor check. |
| `agent.policy` | `decide(t, state, tracks, belief, claims) -> Decision` | One verify task per strong Suspicion, taken by the nearest drone (bid by distance, auction by place within 300 m); everyone else holds their post; Claim lease and settle window; Voronoi Sectors and Fill Gap; baselines `nearest`, `no_perch`; fixed (array) drones never fly. |
| `agent.store` | `receive(frame, margin, t)`, `due(t)`, `on_digest`, `on_request` | Duplicate filter; contention-based relaying (far receivers first, cancel on overhearing, skip when heard strongly); Digest/Request healing of Tracks and Claims; schemes `ours` / `flood` / `unicast3`. |
| `eval` | `evaluate(scenario, runs, seed, policy, scheme, bw)`, `summarize` | Section 17 metrics from decision events + ground truth; `scenario_gen` makes a random world per seed. |
| `audio` | `ArraySensor(device, rotation).readings()`, `doa.estimate_bearing` | Real microphone arrays as a perched drone's sound sensor (SRP-PHAT, 500-4000 Hz). |
| `runners.*` | processes and lockstep | Thin shells. `wire.py` holds the msgpack shapes. |
| `dashboard` | `GET /`, `GET /city`, `WS /ws`, `WS /ws/audio`, `POST /control` | Operator console: hotkeys send Events to the world. Agents never see it; a crash never stops a run. With `earshot live --audio`, Listen (`l`) plays the selected drone's microphone, and the Evidence screen graphs every drone's frequencies (sound per band from the snapshot's `spectrum`, RF dBm per band from the readings). |

## Pre-built vs. on the day

Rule (concept section 14): infrastructure is built beforehand and declared ([DECLARATION.md](../DECLARATION.md)); the judged intelligence is built on Oct 24. Branch `main` holds the infrastructure with the judged logic as stubs. Branch `prototype` holds a working version of the judged logic, built before the event to rehearse and measure: using it on the day means declaring it (DECLARATION.md lists it), or rebuilding it on the day from `main`.

Open items worth doing on the day (prototype):
- Bearings that point at an already confirmed drone should explain away the ghost crossings they create (the source of most remaining false verify flights).
- One aggregated Observation per drone per second, so the swarm also runs on a 125 kHz LoRa channel (it needs 500 kHz now).
- EfficientAT instead of the coarse class stand-in for the arrays; a trained detector instead of the Webots recogniser.

## Where Nikolas plugs in

| Area | Contract |
|---|---|
| City from LoD2 + OSM | Done for Munich Airport: `tools/muc_fetch.py` caches the open data, `tools/make_muc.py` writes `data/cities/muc.json` (optional keys `ground_tiles`, `basemap`, `building_meshes`: the Webots builder and dashboard use them when present). Otherwise: write a City JSON (schema in `earshot/city.py`) into `data/cities/<name>.json`; point a scenario's `city:` at it. Local metric frame, metres, x = east, y = north. |
| Webots world | Either extend `tools/build_webots_world.py` or hand-build a world; it must keep `DEF DRONE_<id>` (Pose), `DEF SRC_<source id>` (Solid with `recognitionColors`) and `DEF RIG` (Robot, `supervisor TRUE`, `controller "<extern>"`, Camera named `camera` with a Recognition node). Base nodes only, or pre-cache PROTOs (ADR 0005). |
| Camera detector | `Sensed.camera_frame` = `{"w","h","objects"}` on the Verifier (the image only via `camera_image()`, for the dashboard preview). Now: Webots' recogniser reports each Source Solid's `model` (its class) and `SensorSuite._camera` turns that into a reading. Replace with a trained detector at the same spot. |
| Live arrays | `scenarios/room.yaml`: drones 0 and 1 are `fixed` (never leave their roof). `earshot arrays` lists devices; `earshot live --scenario scenarios/room.yaml --array 0=<dev>@<rot> --array 1=<dev>@<rot>` replaces their simulated sound with real bearings. `rot` = the compass bearing the array's forward points to in the miniature city (1 m room = 100 m city). On arrival: check the mic-channel order (`audio/doa.py`) and calibrate `calibration_db`. |
| Dashboard | `src/earshot/dashboard/static/index.html`, one file, no build step, no CDN. Data: the `state` and `radio` dicts on `/ws`. |

## Events and replay

Live Events are applied at the next world tick, tagged `live`, and recorded to `runs/live-<time>.events.jsonl`. `earshot fast --events <file>` replays a recorded session on top of its scenario, deterministically, so a demo rehearsal can become a regression run or an evaluation scenario.

## Gotchas

- A ZeroMQ SUB socket returns the oldest queued message. Drain with `NOBLOCK` to get the newest (see `agent_proc.py`).
- Every context sets `LINGER 0`, or processes hang on exit waiting for unsent messages.
- Webots needs `WEBOTS_HOME`, `PYTHONPATH` and `DYLD_LIBRARY_PATH` (`cli.webots_env()`); our instance runs on port 1240 so an already open Webots does not grab the controller.
- Regenerate the Webots world after changing the City or scenario sources: `earshot live --backend webots` does this automatically.
- `WebotsBackend` sleeps each basic step to its wall-clock deadline. With `--no-webots-view`, `earshot live` starts Webots in fast mode, so our per-tick work fits into that slack (1.0x); with the 3D window it stays in realtime mode, where our work comes on top of every step.
- Each Webots camera renders the whole scene. The view cameras run only while a game screen sends its `/view` keepalive (every 2 s, not from the evidence screen or a hidden tab); `--no-webots-view` skips Webots' own 3D window, the largest remaining cost.
