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
- `camera.range` 300 m: at night a Verifier looks with a thermal camera; a vehicle's engine shows far beyond the 100 m a quadcopter takes, and the gimbal geolocates it from the drone's height and the depression angle. `camera.detector: truth`: the stand-in, in Webots too; the COCO detector is a commented block (below, "The Verifier camera in Webots").
- `camera.standoff` 30 m: the Verifier holds 30 m off the place it checks. In Webots the stand-in's camera keeps the airport's tilt (18 deg down) and sees out to its 300 m; a trained detector's camera aims at the ground at the standoff instead. For YOLO the commented block holds 120 m off (18 deg down from 40 m; at 30 m the camera would look 53 deg down, where a car is a phone or an airplane); with the stand-in, 120 m lost poacher1 in one seed of four (results below).
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
| A detector's camera aimed at the hunted class | the Webots camera was tilted for a drone 10 m below; a vehicle on the ground at the standoff passed under a detector's view | `backend/webots.py` (`_verify_pitch`, `SOURCE_Z`, with `detector_frames`; the airport's and the stand-in's tilt are unchanged) |
| YOLO hits geolocated too | a detector's bearing alone puts the confirmed point on the Verifier | `sensors/camera.py` (`ground_range`, the frame's `height`), `sensors._detected` |
| No bids beyond `max_verify_distance` | with Claims lost on a busy channel, drones from the far side of the reserve flew 7 km to the same poacher | `policy._tasks` (PolicyParams, off by default) |
| Relevance-limited relaying | every Track went three hops through all twenty drones: 5.6 relays per Track, a broadcast storm | `store._relevant`, `radio.relay_radius_m` (off by default) |

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

## The Verifier camera in Webots: what COCO YOLO sees

The reserve keeps the camera stand-in. COCO `yolo11n.pt` on the rendered Verifier view was made to see the vehicles, but in a live run it did not confirm the poacher. Measured on 2026-10-21 ([data/cv/savanna-vehicles-20261021/REPORT.md](../data/cv/savanna-vehicles-20261021/REPORT.md): 2,468 offline Webots stills, five 30 s sequences, two live runs):

- **The camera looked over the vehicle.** It was tilted 18 deg for a drone 10 m below; at the 30 m standoff a vehicle on the ground is 53 deg down, under the view (its bottom edge is at 37 deg). Webots' recognition found the target in none of those views. Now, when a detector reads the frames, the tilt aims at the hunted class's height at the standoff.
- **The primitive pickups are no vehicles to COCO**: "airplane", "chair", "bench" at every distance from 30 to 300 m (best vehicle confidence 0.33 in 140 views). Neither is Webots' TruckSimple, with or without its trailer (0.22). Webots' Range Rover is: YOLO's top label is "car" from 50 to 160 m in most views, but mostly at 0.15-0.5 confidence; at 0.25 or more in 10-40 % of views (a white one more often than an olive one), never beyond 160 m. From 30 m (53 deg down) it is a "cell phone" or an "airplane". The reserve's vehicles are now Range Rovers, the poachers' white.
- **No false vehicles**: no view of the elephants, buffalo, lions or the rhino, and no empty view, gave a vehicle box above 0.22 (560 animal views); YOLO names the game "bird", "airplane", "cow", and the acacias "umbrella".
- **On a moving vehicle** (the Verifier flying as the twin flies it, 120 m behind a Range Rover at 8 m/s, the live detector at conf 0.15, verify_conf 0.3, three frames in a row): the white one verified after 4.9 s and in 13 of 30 windows, the olive one after 8.9 s (7/30); never the primitive pickup or the elephant herd. Geolocated from the box's bottom edge: 135-137 m for a vehicle 135 m away.
- **What did not help**: a dark shadow plate under the vehicle, Webots shadows, inference at 1920 or 2560 pixels.
- **Live** (`earshot live --scenario scenarios/savanna.yaml --backend webots --no-webots-view`, the COCO block switched on with standoff 120 m and `arrive` 150, 600 s): four checks near poacher1 (t 254, 353, 469, 573 s), all ended "nothing". At 254, 353 and 573 s the checked place was 214-503 m behind the vehicle (a Suspicion from sound bearings lags a vehicle at 8 m/s); at 469 s the vehicle was 7-62 m from it in the 8 s before, and YOLO boxed no car before three empty windows cleared it. A preview frame of another check shows the camera 100 deg off a vehicle 150 m away. The stand-in sees a vehicle out to 300 m (in Webots in the same 57 deg wedge, in the twin all around); this camera sees a car 50-160 m ahead.

To try it, uncomment the COCO block in `scenarios/savanna.yaml` (`uv sync --group cv`; `EARSHOT_CV_DUMP=<dir>` saves what the detector saw). What it would take: a larger COCO model (yolo11s/m, not tried: a download), a gimbal that sweeps the view while holding, or a detector trained on these renders; the Suspicion's lag behind a moving vehicle is the swarm's, not the camera's.

## Results in the reserve

`uv run earshot eval --scenario scenarios/savanna.yaml --runs 4 --seed 1000 --jobs 4` (lockstep: the stand-in camera at 300 m, the sound stand-in with the game's confusion; about 20 min). Per seed, poacher1's time from cutting the fence to the first camera confirmation:

| Variant | poacher1 confirmed | its t_confirm s | poacher2 | Flights | False verifies | Drones away (max) | Load % | Clean rx % |
|---|---|---|---|---|---|---|---|---|
| without the twenty-drone measures | 2/4 | 131, 122 | 0/4 | 56.5 | 53.5 | 4 | 83 | 40 |
| **shipped**: `max_verify_distance` 3 km, `relay_radius_m` 4 km | **4/4** | 267, 103, 152, 113 | 0/4 | 12 | 11.5 | 2 | 68 | 48 |
| also holding back irrelevant healing (dropped) | 2/4 | 199, 240 | 0/4 | 12.5 | 11 | 2.5 | 59 | 52 |
| standoff 120 m, `arrive` 150 (where YOLO sees a car; not shipped) | 3/4 | -, 110, 145, 115 | 0/4 | 12 | 12 | 2 | 66 | 49 |

Reading it:
- **The poacher with a radio is found every time**, two to four minutes after it cuts the fence, confirmed by the camera as "poacher pickup", and followed. Seed 0 (`uv run earshot fast --scenario scenarios/savanna.yaml --log`, `runs/savanna-seed0.threats.jsonl`): a first check at 244 s finds nothing; at 275 s drone 1 confirms the pickup and follows it to the fence, the Confirmed Track 4-90 m from the vehicle all the way; a second drone confirming it at 464 s joins the same Threat Log entry and yields.
- **The silent poacher is never found.** poacher2 drives to the elephants between towers 13 and 14, 1-1.5 km from each; nothing else is within 2 km. Two engine bearings that wide (σ 20-27° at SNR -1 to -7 dB) give about 3 log-odds where a Suspicion needs about 4 against the threat prior. Twenty towers over 100 km² leave such gaps for a quiet vehicle without a radio: a third tower within 1.5 km, a gunshot, or a phone would close it. This is the honest limit of sound alone at this density.
- **False verifies are mostly flights toward the poacher.** The metric counts a dispatch with no intruder within 150 m at that moment; a vehicle at 8 m/s is 150-500 m ahead of its Suspicion, whose evidence fades over 30 s. In seed 1000 every one of the first 11 dispatches went toward poacher1.
- **Before the twenty-drone measures** Claims lost on the busy channel sent drones from the far side of the reserve (7 km away) to the same poacher, and each Track was relayed 5.6 times. The channel is still busy (68 %): one aggregated Observation per drone per second is the next step.

## Limits

- **A poacher on a track is Explained**, like the game drive, and never verified; a ranger off the tracks is verified. Only where a vehicle drives separates them. The rangers' own radios report GPS positions in many reserves; as City knowledge they would explain the rangers wherever they drive, and every other vehicle could be suspicious.
- **The camera is the stand-in, in Webots too** (`camera.detector: truth`). COCO YOLO on Webots' Range Rovers sees a car 50-160 m ahead, at low confidence; in a live run it cleared all four checks near the moving poacher ("The Verifier camera in Webots" above). The stand-in also knows the reserve's own vehicles (ground truth `intruder: false`), as a classifier trained on their livery would; to COCO a ranger's car is a car like the poacher's.
- **The game is a Source with a constant level.** Real calls come in bursts (a lion roars for a minute, then not for an hour); elephants' rumble is mostly infrasound. Animals use the Nuisance acoustics of a fan (3.7 dB/km absorption).
- **UHF calibration**: 50 % detection at 2.5 km in the open and 0.7 km behind a kopje, chosen, not measured; push-to-talk as a 40 % chance per second of any transmission.
- **No people on foot.** Poachers who walk in from a road make little sound; gunshots (loud, short, heard 5 km away) would be the cue, as a fifth sound class or a short Source.
- **Twenty drones on one LoRa channel** load it heavily (68 %, half the receptions collide); one aggregated Observation per drone per second (an open item since Munich) would help. Holding back healing of far-away Tracks quieted it further but lost poachers (table above).
- **A silent poacher between two towers is not found** (results above). Gunshots are not modelled.
- **Followers are not checked against battery**: `follow_for` 900 s is within a 25-minute flight, but a long chase from a far tower is not.
- **Belief size**: 50 m cells over 10 km, 20 origins per drone; a lockstep run of 900 s takes several minutes.
