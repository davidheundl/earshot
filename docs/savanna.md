# Savanna: the swarm against poachers

The same swarm that guards Munich Airport against drones, flown over a 10 x 10 km private game reserve in the South African lowveld. Vocabulary: [CONTEXT.md](../CONTEXT.md) (Reserve, Threat Class, Poacher, Follow, Threat Log). Decision: [ADR 0009](adr/0009-threat-class-per-scenario.md).

## The idea

A perching swarm is not tied to an airport. Twenty drones sit on lookout towers 2-2.5 km apart and listen to the bush at night; together they cover 100 km². Engines and handheld radios are what gives poachers away: their pickup drives cross-country, where nobody else drives, and the crew talks on a UHF handheld. The swarm fuses sound bearings and radio energy into its Probability Map, the nearest drone flies out with its camera, the camera tells what it is (a poacher's pickup, an elephant herd, the rangers' own Land Cruiser), and a confirmed poacher is followed and logged. What the rangers do then is theirs.

```bash
uv run earshot live --scenario scenarios/savanna.yaml
uv run earshot live --scenario scenarios/savanna.yaml --backend webots --no-webots-view
uv run earshot fast --scenario scenarios/savanna.yaml --log     # runs/savanna-seed0.threats.jsonl
```

## The reserve (`tools/make_savanna.py` -> `data/cities/savanna.json`)

Generated and deterministic (seed 24), a City like Munich or the valley, with `biome: savanna`:

| Feature | What it does in the simulation |
|---|---|
| Terrain, 25 m grid, 328-408 m | a plain falling to the east; the sand river cut ~11 m in; seven granite kopjes 18-44 m high that block sound and radio (`City.obstacles_between` counts terrain) |
| Land use (2048 px) | grass, thornveld, thickets, riverine forest, sand, rock, track, camp: the dashboard's basemap and the Webots ground and trees; the agents never use it |
| 10 tracks (`roads`, `road_surfaces`) | the fence road 120 m inside the boundary, the gravel spine west to east, eight dirt tracks between the camp, the lodge, the waterholes and the outposts. A vehicle on or beside one is Explained |
| 7 waterholes (`water`) | where the game drinks |
| Ranger HQ, a lodge, three outposts | huts under corrugated iron and thatch; their flat roofs are Perch Sites |
| 20 lookout towers | 12 m steel towers on the highest ground of a 5 x 4 grid: each drone's home Perch Site |

## The scenario (`scenarios/savanna.yaml`)

- `threat: vehicle`: what the swarm hunts. Off the tracks a vehicle gets the threat prior (-5, as a drone does at the airport) instead of -8.
- poacher1 cuts the south fence at 60 s, drives cross-country (8 m/s) to a white rhino in the thicket, circles it and leaves through the fence; its crew is on a UHF handheld. poacher2 comes through the east fence at 280 s for the elephant herd by the river, without a radio.
- ranger1 patrols the gravel spine in radio silence (a poacher's scanner would hear it too), safari1 (a game drive) the lodge's track: vehicles, but `intruder: false`.
- Game as Nuisances with a `kind`: an elephant herd (105 dB(A)), a lion pride roaring (106), hyenas, buffalo, zebra, impala, a giraffe and the rhino (72). `confusion:` makes the sound stand-in hear them as an engine 45 % of the time: a classifier trained on European data has never heard a lion.
- `params`: `same_task_radius` 1200 m (a poacher drives on while the Verifier flies out), `confirmed_shadow` 400 m (the followed vehicle's trail is no second poacher), `follow_for` 900 s, `max_verify_distance` 3000 m (a nearer tower is always there).
- `radio.relay_radius_m` 4000: Tracks and Claims are relayed only near their place.
- `camera.range` 300 m: at night a Verifier looks with a thermal camera; a vehicle's engine shows far beyond the 100 m a quadcopter takes, and the gimbal geolocates it from the drone's height and the depression angle.
- `map_cell` 50 m (200 x 200 cells per drone), `radio.nominal_range_m` 3500, `environment.background_db` 35 (bush at night).

## What had to change, and what it costs the airport

| Change | Why | Where |
|---|---|---|
| Threat Class per Scenario | the swarm hunted drones only | `scenario.threat`, `SwarmConfig.threat`, belief, policy, core, eval (ADR 0009) |
| UHF band, `Band.emitter`, `duty` | poachers' handhelds; an RF ring raises the class its band's emitters are | `sensors/rf.py`, `Sensor.RF_UHF` |
| `intruder:`, `kind:` per Source | rangers are vehicles but no intruders; the camera and Webots need to know what a Source is | `scenario.py` |
| Camera labels | "what is it?": the stand-in names the Source's `kind`, YOLO its class names; a recognised Nuisance or one of the reserve's own vehicles is no threat | `sensors/__init__.py`, `sensors/camera.py` (`classes`) |
| Follow | stay with a confirmed poacher | `policy.py` (`follow_for`, events `following`, `lost`, `handed_off`) |
| Threat Log | log every checked danger with the camera's label and the followed path | `threat_log.py`, both runners, the dashboard |
| Task ids for 12.8 km | 100 km² has 10 000 cells of 100 m | `policy.TASK_STRIDE` 16384 |
| Explained Tracks cheap on air | a ranger's radio left a trail of Explained peaks that filled each drone's six Track places and half the channel | `core.py` (one hop, one per 300 m per 30 s), `belief.tracks` (their own six places) |
| A road through a vehicle's ellipse explains it | a vehicle heard from 2 km lands a few hundred metres beside its road | `belief._status` |
| Healing only what is worth healing | a drone 10 km away never hears another's one-hop frames and kept asking for them: half the channel | `store.healable` |
| Geolocated camera hits | the confirmed point sat on the Verifier while the vehicle drove on 300 m ahead; Followers lost it | `belief._camera`, the stand-in with `camera.range` |
| No bids beyond `max_verify_distance` | with Claims lost on a busy channel, drones from the far side of the reserve flew 7 km to the same poacher | `policy._tasks` (PolicyParams, off by default) |
| Relevance-limited relaying | every Track went three hops through all twenty drones: 5.6 relays per Track, a broadcast storm | `store._relevant`, `radio.relay_radius_m` (off by default) |
| One Observation per drone and sensor window | a drone hearing two things sent two 35 B frames; the channel was at 68 % | `messages.Observation` (up to two readings), `core.py` step 2, `belief.add_evidence` |

The airport, re-evaluated with the same seeds (`earshot eval`, medians [min-max]):

| Set | Variant | Confirmed | t_confirm s | False verifies | Flights | Drones away (max) | Load % | Agreed after a jam |
|---|---|---|---|---|---|---|---|---|
| Munich (`muc.yaml`, seeds 1000-1007) | before | 8/8 | 95 [85-267] | 21 | 31.5 | 3 | 26.3 | 8/8 |
| | after | 8/8 | 87 [84-198] | 24.5 | 35 | 5 | 23.8 | 8/8 |
| | one Observation per window | 8/8 | 87 [84-116] | 17.5 | 27.5 | 3 | 20.2 | 7/8 |
| Random worlds (seeds 1000-1019) | before | 16/26 | 170 | 19 | 29.5 | 4 | 20.8 | 6/9 |
| | after | 17/26 | 148 | 18.5 | 34 | 5 | 21.4 | 8/9 |
| | one Observation per window | 16/26 | 189 | 15.5 | 27.5 | 5 | 18.6 | 7/9 |
| Demo (`demo.yaml`, seeds 100-111) | before | 12/12 | 26 | 16 | 33.5 | 4 | 23.6 | 12/12 |
| | after | 12/12 | 20 | 16 | 35.5 | 4 | 23.3 | 12/12 |
| | one Observation per window | 12/12 | 22 | 14.5 | 34.5 | 4 | 21.5 | 12/12 |

"Before" is `prototype` at 1b70338; stand-in sound and camera (lockstep), default PolicyParams. Reading it: the airport confirms its drone sooner and agrees after a jam more often, with a quieter channel; it flies a little more and sends up to one or two more drones away at once. Single seeds swing both ways (Munich seed 1001: 225 -> 198 s, seed 1006: 87 -> 119 s). An intermediate state without the healing change missed Munich's drone on one seed and agreed after only 6 of 8 jams: one-hop Explained Tracks made far drones ask for frames they could never get.

"One Observation per window" (2026-10-21) is "after" plus a sensor window's readings sharing their frames; "after" re-run on `prototype` 348bfac gives the row above exactly. The channel is quieter in every set (load lower on 54 of 60 random worlds and 19 of 24 Munich seeds, clean receptions up 1-2 points), and there are fewer flights and false verifies. Confirmations and agreement after a jam moved by one run in the small sets, so both were run on more seeds: over 40 more random worlds (1020-1059) 35 -> 38 of 48 drones confirmed and 19/21 agreed both times; over 16 more Munich seeds (1008-1023) 15 -> 16 of 16 confirmed and 9 -> 14 of 16 agreed. All seeds together, paired per drone Source: Munich confirms sooner (mean 124 -> 104 s, 23 -> 24 of 24) and agrees after 21 instead of 17 of 24 jams; the random worlds confirm 54 instead of 52 of 74 drones (mean 148 -> 143 s, sooner in 29, later in 27) and agree after 26 instead of 27 of 30 jams. On a 125 kHz channel (`--bw 125`, demo seeds 100-111) it is no better: both variants run into the 10 % duty cycle (load 50 %), confirm 9 of 12 and never agree again after the jam.

## Results in the reserve

`uv run earshot eval --scenario scenarios/savanna.yaml --runs 4 --seed 1000 --jobs 4` (lockstep: the stand-in camera at 300 m, the sound stand-in with the game's confusion; about 20 min). Per seed, poacher1's time from cutting the fence to the first camera confirmation:

| Variant | poacher1 confirmed | its t_confirm s | poacher2 | Flights | False verifies | Drones away (max) | Load % | Clean rx % |
|---|---|---|---|---|---|---|---|---|
| without the twenty-drone measures | 2/4 | 131, 122 | 0/4 | 56.5 | 53.5 | 4 | 83 | 40 |
| `max_verify_distance` 3 km, `relay_radius_m` 4 km | **4/4** | 267, 103, 152, 113 | 0/4 | 12 | 11.5 | 2 | 68 | 48 |
| also holding back irrelevant healing (dropped) | 2/4 | 199, 240 | 0/4 | 12.5 | 11 | 2.5 | 59 | 52 |
| **shipped**: also one Observation per drone and sensor window (2026-10-21) | **4/4** | 118, 91, 266, 123 | 0/4 | 9.5 | 9.5 | 3 | 63.5 | 49 |

Reading it:
- **The poacher with a radio is found every time**, one and a half to four and a half minutes after it cuts the fence, confirmed by the camera as "poacher pickup", and followed. Seed 0 (`uv run earshot fast --scenario scenarios/savanna.yaml --log`, `runs/savanna-seed0.threats.jsonl`): at 213 s drone 1 confirms the pickup and follows it to the end of the run, the point it follows (the Confirmed Track, reported once a second) a median 118 m from the vehicle (20-300 m); a second drone confirms a vehicle at 476 s, 300 m off, as an entry of its own, and yields after 4 s. The same seed before 2026-10-21 (`prototype` 348bfac): a first check at 244 s finds nothing, drone 1 confirms at 275 s and follows at a median 124 m (7-260 m), and a second drone confirming it at 464 s joins the same entry.
- **The silent poacher is never found.** poacher2 drives to the elephants between towers 13 and 14, 1-1.5 km from each; nothing else is within 2 km. Two engine bearings that wide (σ 20-27° at SNR -1 to -7 dB) give about 3 log-odds where a Suspicion needs about 4 against the threat prior. Twenty towers over 100 km² leave such gaps for a quiet vehicle without a radio: a third tower within 1.5 km, a gunshot, or a phone would close it. This is the honest limit of sound alone at this density.
- **False verifies are mostly flights toward the poacher.** The metric counts a dispatch with no intruder within 150 m at that moment; a vehicle at 8 m/s is 150-500 m ahead of its Suspicion, whose evidence fades over 30 s. In seed 1000 (before 2026-10-21) every one of the first 11 dispatches went toward poacher1.
- **Before the twenty-drone measures** Claims lost on the busy channel sent drones from the far side of the reserve (7 km away) to the same poacher, and each Track was relayed 5.6 times.
- **One Observation per drone and sensor window** quiets the channel from 68 to 63.5 % (63-65 % on every seed, 66-70 % before) and lets a little more through (49 % of receptions clean instead of 48 %), with fewer flights. It can do no more: in seed 1000 (900 s) Observations were about a quarter of the frames and of the airtime, and 78 % of the drone-seconds in which a drone heard anything it heard one thing (sound 3548, UHF 484, camera 342 of 5627); two readings shared a second 1528 times, three 225 times. The rest of that run's 33 871 frames is what is left to quiet: healing (6306 resends and 3653 Requests, 29 %), relays (22 %) and own Tracks (19 %, half of them Explained).

## Limits

- **A poacher on a track is Explained**, like the game drive, and never verified; a ranger off the tracks is verified. Only where a vehicle drives separates them. The rangers' own radios report GPS positions in many reserves; as City knowledge they would explain the rangers wherever they drive, and every other vehicle could be suspicious.
- **The camera is the stand-in, in Webots too** (`camera.detector: truth`). The COCO detector (`yolo11n.pt`, `camera.classes`) is wired, but in a live Webots run on 2026-10-20 it saw the primitive pickups as "bird" and dismissed both checks near the poacher; it would need real vehicle models (the Range Rover PROTO failed to download, see the builder) or a detector trained on these renders. The stand-in also knows the reserve's own vehicles (ground truth `intruder: false`), as a classifier trained on their livery would.
- **The game is a Source with a constant level.** Real calls come in bursts (a lion roars for a minute, then not for an hour); elephants' rumble is mostly infrasound. Animals use the Nuisance acoustics of a fan (3.7 dB/km absorption).
- **UHF calibration**: 50 % detection at 2.5 km in the open and 0.7 km behind a kopje, chosen, not measured; push-to-talk as a 40 % chance per second of any transmission.
- **No people on foot.** Poachers who walk in from a road make little sound; gunshots (loud, short, heard 5 km away) would be the cue, as a fifth sound class or a short Source.
- **Twenty drones on one LoRa channel** load it heavily (63.5 %, half the receptions collide), even with a window's readings in one Observation. Healing (29 % of the frames) and relays (22 %) are what is left to quiet; holding back healing of far-away Tracks quieted it, but lost poachers (table above).
- **A silent poacher between two towers is not found** (results above). Gunshots are not modelled.
- **Followers are not checked against battery**: `follow_for` 900 s is within a 25-minute flight, but a long chase from a far tower is not.
- **Belief size**: 50 m cells over 10 km, 20 origins per drone; a lockstep run of 900 s takes several minutes.
