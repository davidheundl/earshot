# Savanna: the swarm against poachers

The same swarm that guards Munich Airport against drones, flown over a 10 x 10 km private game reserve in the South African lowveld. Vocabulary: [CONTEXT.md](../CONTEXT.md) (Reserve, Threat Class, Poacher, Shot, Follow, Threat Log). Decision: [ADR 0009](adr/0009-threat-class-per-scenario.md).

## The idea

A perching swarm is not tied to an airport. Twenty drones sit on lookout towers 2-2.5 km apart and listen to the bush at night; together they cover 100 km². Engines, handheld radios and rifle shots are what give poachers away: their pickup drives cross-country, where nobody else drives, the crew talks on a UHF handheld, and a shot is heard for kilometres, as acoustic gunshot detection in real reserves uses. The swarm fuses sound bearings and radio energy into its Probability Map, the nearest drone flies out with its camera, the camera tells what it is (a poacher's pickup, an elephant herd, the rangers' own Land Cruiser), and a confirmed poacher is followed and logged. What the rangers do then is theirs.

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
- poacher1 cuts the south fence at 60 s, drives cross-country (8 m/s) to a white rhino in the thicket, circles it and leaves through the fence; its crew is on a UHF handheld. poacher2 comes through the east fence at 280 s for the elephant herd by the river, without a radio; at 560 s it stops 160 m from the herd, fires at 570 s (a Shot, `events:`) and stays five minutes at the kill (`[x, y, 300]` on its path).
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
| Shots | a poacher without a radio between two towers was never found; a rifle shot is heard for kilometres with a sharp bearing, as acoustic gunshot detection in real reserves uses. As a sound bearing it vanished among the tower's thirty engine and animal bearings | `shot` Event (`Scenario.shots`, no Source), `acoustics.SHOT`, `SensorSuite._shots`, `Sensor.SHOT`, the Belief's `shot` family |
| Stops on a path | a poacher that shoots stays at its kill; driving on at 9 m/s it was 450 m from the shot when the Verifier got there | `Source.waits` (`[x, y, s]`), lockstep `truth`, `eval` |

The airport, re-evaluated with the same seeds (`earshot eval`, medians [min-max]):

| Set | Variant | Confirmed | t_confirm s | False verifies | Flights | Drones away (max) | Load % | Agreed after a jam |
|---|---|---|---|---|---|---|---|---|
| Munich (`muc.yaml`, seeds 1000-1007) | before | 8/8 | 95 [85-267] | 21 | 31.5 | 3 | 26.3 | 8/8 |
| | after | 8/8 | 87 [84-198] | 24.5 | 35 | 5 | 23.8 | 8/8 |
| Random worlds (seeds 1000-1019) | before | 16/26 | 170 | 19 | 29.5 | 4 | 20.8 | 6/9 |
| | after | 17/26 | 148 | 18.5 | 34 | 5 | 21.4 | 8/9 |
| Demo (`demo.yaml`, seeds 100-111) | before | 12/12 | 26 | 16 | 33.5 | 4 | 23.6 | 12/12 |
| | after | 12/12 | 20 | 16 | 35.5 | 4 | 23.3 | 12/12 |

"Before" is `prototype` at 1b70338; stand-in sound and camera (lockstep), default PolicyParams. Reading it: the airport confirms its drone sooner and agrees after a jam more often, with a quieter channel; it flies a little more and sends up to one or two more drones away at once. Single seeds swing both ways (Munich seed 1001: 225 -> 198 s, seed 1006: 87 -> 119 s). An intermediate state without the healing change missed Munich's drone on one seed and agreed after only 6 of 8 jams: one-hop Explained Tracks made far drones ask for frames they could never get.

Shots and stops on a path (2026-10-21) change nothing at the airport: the Munich and demo lockstep logs (snapshots, detections, Threat Log, seed 0) stay byte-identical.

## Results in the reserve

`uv run earshot eval --scenario scenarios/savanna.yaml --runs 4 --seed 1000 --jobs 4` (lockstep: the stand-in camera at 300 m, the sound stand-in with the game's confusion; 20-35 min). Per seed, each poacher's time from coming through the fence to the first camera confirmation:

| Variant | poacher1 confirmed | its t_confirm s | poacher2 | its t_confirm s | Flights | False verifies | Drones away (max) | Load % | Clean rx % |
|---|---|---|---|---|---|---|---|---|---|
| without the twenty-drone measures | 2/4 | 131, 122 | 0/4 | | 56.5 | 53.5 | 4 | 83 | 40 |
| `max_verify_distance` 3 km, `relay_radius_m` 4 km (`prototype` at 348bfac) | 4/4 | 267, 103, 152, 113 | 0/4 | | 12 | 11.5 | 2 | 68 | 48 |
| also holding back irrelevant healing (dropped) | 2/4 | 199, 240 | 0/4 | | 12.5 | 11 | 2.5 | 59 | 52 |
| poacher2 stops at its kill, no Shot (control) | 4/4 | 267, 103, 152, 113 | 0/4 | | 6.5 | 6 | 2 | 69 | 48 |
| **shipped**: it stops and fires (a Shot) | **4/4** | 267, 103, 152, 113 | **4/4** | 349, 358, 350, 343 | 15 | 13.5 | 4 | 71 | 47 |

Reading it:
- **The poacher with a radio is found every time**, two to four minutes after it cuts the fence, confirmed by the camera as "poacher pickup", and followed. Seed 0 (`uv run earshot fast --scenario scenarios/savanna.yaml --log`, `runs/savanna-seed0.threats.jsonl`): a first check at 244 s finds nothing; at 275 s drone 1 confirms the pickup and follows it to the fence, the Confirmed Track 4-90 m from the vehicle all the way; a second drone confirming it at 464 s joins the same Threat Log entry and yields.
- **The silent poacher is found once it fires, never before.** poacher2 drives to the elephants between towers 13 and 14, 0.8-1.5 km from each; nothing else is within 2 km. Two engine bearings that wide (σ 20-27° at SNR -1 to -7 dB) give about 3 log-odds where a Suspicion needs about 4 against the threat prior, driving or parked (the control). The Shot at 570 s is heard by about eleven towers, the eight within 3.5 km almost surely and some out to 6 km, 3-18 s later, with bearings of 2-6°; a drone is dispatched to it 8-36 s after it, and the "poacher bakkie" is confirmed 53-68 s after it, where it stands at the kill, and followed. Seed 1000: drone 19 (2.4 km) claims first and takes off, drone 13 (0.8 km) outbids it 5 s after its Claim, flies at 583 s and confirms at 629 s, the Confirmed Track 5 m from the bakkie. The Shot comes after the kill: it saves no elephant, it gives the rangers the poachers at the carcass and the way they leave.
- **Without the stop the Shot is not enough**: a poacher that fires and drives on at 9 m/s was 450 m from the shot when drone 13 got there (seed 1000, before the stop), beyond the 300 m camera; the check found only the herd and was dismissed.
- **What the Shot costs**: up to three more drones away at once (median 2 -> 4: drone 13 follows poacher2 for the rest of the run while poacher1's checks go on) and about 3 points more channel load (68 -> 71 %). Flights and false verifies swing with the run's course after 560 s more than with the Shot: the stop alone took them from 12 to 6.5. In seed 1000 the Shot's two dispatches went to within 20 m of the bakkie; every later one was near poacher1's trail (0.2-2 km from it), 6 km from poacher2.
- **False verifies are mostly flights toward the poacher.** The metric counts a dispatch with no intruder within 150 m at that moment; a vehicle at 8 m/s is 150-500 m ahead of its Suspicion, whose evidence fades over 30 s. In seed 1000 every one of the first 11 dispatches went toward poacher1.
- **Before the twenty-drone measures** Claims lost on the busy channel sent drones from the far side of the reserve (7 km away) to the same poacher, and each Track was relayed 5.6 times. The channel is still busy (68 %): one aggregated Observation per drone per second is the next step.

## Limits

- **A poacher on a track is Explained**, like the game drive, and never verified; a ranger off the tracks is verified. Only where a vehicle drives separates them. The rangers' own radios report GPS positions in many reserves; as City knowledge they would explain the rangers wherever they drive, and every other vehicle could be suspicious.
- **The camera is the stand-in, in Webots too** (`camera.detector: truth`). The COCO detector (`yolo11n.pt`, `camera.classes`) is wired, but in a live Webots run on 2026-10-20 it saw the primitive pickups as "bird" and dismissed both checks near the poacher; it would need real vehicle models (the Range Rover PROTO failed to download, see the builder) or a detector trained on these renders. The stand-in also knows the reserve's own vehicles (ground truth `intruder: false`), as a classifier trained on their livery would.
- **The game is a Source with a constant level.** Real calls come in bursts (a lion roars for a minute, then not for an hour); elephants' rumble is mostly infrasound. Animals use the Nuisance acoustics of a fan (3.7 dB/km absorption).
- **UHF calibration**: 50 % detection at 2.5 km in the open and 0.7 km behind a kopje, chosen, not measured; push-to-talk as a 40 % chance per second of any transmission.
- **No people on foot.** Poachers who walk in from a road make little sound until they fire; a Shot would tell where, but nobody on foot is modelled for the camera to find.
- **Shots are calibrated, not measured**, like UHF: the level is a 1 s window's sound exposure (134 dB(A)), 2 dB/km, a kopje 15 dB; a tower hears one almost surely at 3 km in the open, half the time at 6 km; bearings 2-6 deg. One impulse, heard once per tower after distance / 343 m/s; no echoes off the kopjes, no other bangs (thunder, a ranger's warning shot): every Shot counts for the Threat Class.
- **The kill is not modelled**: the poacher's engine keeps its level while it stands, the herd walks on, no carcass.
- **Twenty drones on one LoRa channel** load it heavily (68 %, half the receptions collide); one aggregated Observation per drone per second (an open item since Munich) would help. Holding back healing of far-away Tracks quieted it further but lost poachers (table above).
- **A silent poacher between two towers is found only when it fires** (results above), and only if it stays near where it fired until a Verifier is there, about a minute.
- **Followers are not checked against battery**: `follow_for` 900 s is within a 25-minute flight, but a long chase from a far tower is not.
- **Belief size**: 50 m cells over 10 km, 20 origins per drone; a lockstep run of 900 s takes several minutes.
