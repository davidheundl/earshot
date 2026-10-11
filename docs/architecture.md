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
│   PULL :5551 ← command + telemetry from agents                           │
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
| `scenario` | `Scenario.load(yaml, seed)`, `active_sources(t)`, `jam_zones(t)`, `faults_for` | The only place ground truth lives. |
| `backend` | `step(dt, commands) -> states`, `camera_frame(id)`, `close()` | Adapters: `twin2d.Twin2D`, `webots.WebotsBackend` (ADR 0002). |
| `sensors` | `SensorSuite.sense(state, t) -> [Detection]` | Physics in `acoustics.py` / `rf.py` (pure), calibrated to concept 5.1/5.2 by `tests/test_sensors.py`. A Detection = reading + truth; truth never reaches Agents. |
| `world` | `World.step(commands) -> {id: Sensed}`, `detections`, `snapshot()` | Backend + truth + sensors. Runners own the clock. `detections` feeds logs and metrics. |
| `radio.medium` | `RadioMedium.step(t, positions, outgoing, jam) -> delivered`, `stats()` | LoRa airtime, shadowing, half duplex, capture, duty cycle. |
| `agent` | `Agent.step(t, sensed, inbox) -> StepOut` | Pure. Composes `MessageStore`, `Belief`, `Policy`. |
| `runners.*` | processes and lockstep | Thin shells. `wire.py` holds the msgpack shapes. |
| `dashboard` | `GET /`, `GET /city`, `WS /ws` | Observer only; can crash without hurting a run. |

## Pre-built vs. on the day

Rule (concept section 14): infrastructure is built beforehand and declared ([DECLARATION.md](../DECLARATION.md)); the judged intelligence is built on Oct 24. In code, stubs to be filled on the day are marked `ON THE DAY` or `STUB`:

- `agent/belief.py`: measurement models, fusion, explaining away, Faulty Sensor, Track extraction.
- `agent/policy.py`: Perch Site choice by crossing angle, verification value vs. Gap cost, Claim auction with leases, Sector re-partitioning.
- `agent/store.py`: relevance radius, contention-based forwarding, Digest/Request healing.

## Where Nikolas plugs in

| Area | Contract |
|---|---|
| City from LoD2 + OSM | Write a City JSON (schema in `earshot/city.py`) into `data/cities/<name>.json`; point a scenario's `city:` at it. Local metric frame, metres, x = east, y = north. |
| Webots world | Either extend `tools/build_webots_world.py` or hand-build a world; it must keep `DEF DRONE_<id>` (Pose), `DEF SRC_<source id>` (Solid with `recognitionColors`) and `DEF RIG` (Robot, `supervisor TRUE`, `controller "<extern>"`, Camera named `camera` with a Recognition node). Base nodes only, or pre-cache PROTOs (ADR 0005). |
| Camera detector | Consumes `Sensed.camera_frame` = `{"w","h","bgra","seg","objects"}` on the Verifier and produces a `SensorReading(Sensor.CAMERA, ...)`. Hook point: `World.step`, where `camera_frame` is attached. |
| Live arrays | Planned: two Drones in the scenario marked as permanently perched at the Perch Sites the room maps to (1 m room = 100 m city). Their agent processes take readings from a microphone-array sensor instead of `SensorSuite`, so the arrays enter the swarm as ordinary Observations over the simulated radio. |
| Dashboard | `src/earshot/dashboard/static/index.html`, one file, no build step, no CDN. Data: the `state` and `radio` dicts on `/ws`. |

## Gotchas

- A ZeroMQ SUB socket returns the oldest queued message. Drain with `NOBLOCK` to get the newest (see `agent_proc.py`).
- Every context sets `LINGER 0`, or processes hang on exit waiting for unsent messages.
- Webots needs `WEBOTS_HOME`, `PYTHONPATH` and `DYLD_LIBRARY_PATH` (`cli.webots_env()`); our instance runs on port 1240 so an already open Webots does not grab the controller.
- Regenerate the Webots world after changing the City or scenario sources: `earshot live --backend webots` does this automatically.
