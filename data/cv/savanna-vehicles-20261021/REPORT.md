# What COCO YOLO sees in the savanna world

2026-10-21. Why `yolo11n.pt` missed the poacher in a live Webots run on 2026-10-20, and what
makes it see a car. Offline renders of the savanna world (`tools/build_webots_world.py`) with
the Verifier camera as the backend builds it: 1280x768, 1 rad horizontal field of view, 40 m
above the ground (the cruise height). Ultralytics 8.4.182, `yolo11n.pt`, inference at 1280
on MPS.

## Set-up

`scripts/make_world.py` copies the built `webots/worlds/savanna.wbt` and adds candidate
vehicles parked under the ground: TruckSimple with and without its trailer, Webots' Range
Rover Sport SVR (olive and white), the Tesla Model 3, the same with a dark "contact shadow"
plate, and the reserve's primitive pickup with one. `scripts/launch.py` starts Webots on its
own port (1247, not the live 1240) and runs a capture script as the RIG's extern controller.

- `scripts/capture.py`: stills. Two open places (thornveld on poacher1's route, sand by the
  river on poacher2's), the camera south / south-west of the target at 30, 50, 80, 120, 160,
  220 and 300 m horizontally, the vehicle turned 0, 45, 90, 135 and 180 deg against the view,
  the camera tilted as before (18.4 deg: a drone 10 m below at a 30 m standoff, "fixed") or
  aimed at the target ("aimed"). Webots' recognition gives the target's box, so a YOLO box
  counts for the target when its centre is inside that box grown by 30 %.
  1,568 stills of 11 models plus empty views; 900 more for the shadow variants.
- `scripts/evaluate.py`, `scripts/summarize.py`: YOLO on each still (conf 0.05), per model,
  distance and tilt the best vehicle label on the target and the top labels: `summary.json`.
- `scripts/capture_seq.py`, `scripts/eval_seq.py`: 30 s at 10 Hz of a Verifier flying as the
  twin flies it (15 m/s towards a look-at point 2 s behind the target, holding its standoff,
  camera tilted for that standoff) behind a vehicle driving 8 m/s, then the live detector
  (`earshot.sensors.camera.YoloDetector`, ByteTrack) on the frames, 1 s windows.

## Findings

**The camera looked over the vehicle.** Tilted 18.4 deg for a drone 10 m below, the view's
bottom edge is 36.6 deg down: a vehicle on the ground is in view only from 54 m out. At the
scenario's 30 m standoff Webots' recognition finds no target in any "fixed" view: the
Verifier hovered over the poacher looking past it, and after three empty windows its camera
said "nothing". Fixed in `backend/webots.py`: with a detector reading the frames, the tilt aims at the hunted class's height at the standoff.

**Vehicle label at 0.25 or more on the target, aimed camera** (views with the target in frame;
10 per cell = 2 places x 5 headings):

| Model | 30 m | 50 m | 80 m | 120 m | 160 m | 220 m | 300 m | best | top labels |
|---|---|---|---|---|---|---|---|---|---|
| primitive pickup (poacher) | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0.33 | airplane, chair |
| primitive pickup (ranger) | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0.13 | chair, airplane |
| TruckSimple with trailer | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/8 | 0.22 | airplane, chair, clock |
| TruckSimple, cab only | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/4 | 0.09 | chair, airplane |
| Tesla Model 3 | 0/10 | 2/10 | 2/10 | 2/10 | 1/10 | 0/10 | 0/4 | 0.66 | car, cell phone |
| Range Rover, olive | 1/10 | 2/10 | 2/10 | 2/10 | 1/10 | 0/10 | 0/4 | 0.73 | car, chair |
| Range Rover, white | 1/10 | 3/10 | 3/10 | 3/10 | 4/10 | 0/10 | 0/4 | 0.69 | car, chair, airplane |
| elephant herd | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0.20 | bird, airplane, cow |
| buffalo herd | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0.22 | bird, airplane, person |
| lion pride | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0.00 | bird, airplane |
| white rhino | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | 0/10 | - | 0.06 | bird, airplane |

"best": the highest vehicle confidence on that target over all its views. The primitive
pickup and the truck never read as vehicles; the Range Rover is YOLO's top label "car" in most
views from 50 to 160 m, but mostly at 0.15-0.5. No animal view reached 0.25 for a vehicle
(560 views, highest 0.22), and no view had a vehicle box anywhere off the target (also the
empty ones). Close up and from above (30 m, 52 deg down) the Range Rover is a "cell phone" or
an "airplane". `models_50m_120m.jpg`, `rangerover_vs_pickup_yaws.jpg`.

**What did not help:** a dark contact-shadow plate under the vehicle (more "airplane" at
50-80 m, the pickup still never a vehicle); Webots shadows (`castShadows TRUE` on the light:
no visible change over the 10 km world); inference at 1920 or 2560 instead of 1280 (fewer
"car", more "chair" and "boat").

**The live detector on a moving vehicle** (conf 0.15, verify_conf 0.3, 3 frames in a row,
every 2nd frame; standoff 120 m unless named; 30 windows of 1 s):

| Sequence | windows with a verified vehicle | first after |
|---|---|---|
| Range Rover, white | 13/30 | 4.9 s |
| Range Rover, olive | 7/30 | 8.9 s |
| Range Rover, white, standoff 80 m | 3/30 | 9.9 s |
| primitive pickup | 0/30 | - |
| elephant herd (1 m/s) | 0/30 (named "bird", "sheep") | - |

With conf 0.25 / verify_conf 0.4 only the white Range Rover was verified (6/30, after 8.9 s).
Each hit's ground range from the box's bottom edge, the camera's height and its tilt was
135-137 m for a vehicle 134.7-135.9 m away.

## Live

Two live runs (`earshot live --scenario scenarios/savanna.yaml --backend webots --no-webots-view`)
with these settings switched on: Range Rovers, the camera aimed at the ground, standoff 120 m
(`params.arrive` 150), conf 0.15, verify_conf 0.3, clear_radius 150 m. Run 1 (600 s; the Mac
loaded by other evaluations, the world at about 0.4x real time): four checks near poacher1 at
254, 353, 469 and 573 s, all "nothing" (YOLO named a "bird" in view). Poacher1 was 214-276,
324-387, 7-62 and 441-503 m from the checked place in the 8 s before each: the Suspicion of
a vehicle at 8 m/s lags it. At 469 s the vehicle was near the place and YOLO still boxed no
car in three windows. A dashboard preview at 498 s shows the Verifier looking 100 deg off a
vehicle 150 m away. Run 2 (with `EARSHOT_CV_DUMP`) was stopped at t = 145 s, the world at
0.06-0.1x real time; its frames show the acacias as "umbrella" (up to 0.83, unmapped) and a
first check about 700 m from the poacher.

## Consequences (`scenarios/savanna.yaml`)

The reserve keeps the stand-in (`detector: truth`, standoff 30 m); the measured COCO settings,
standoff 120 m and `params.arrive` 150 among them, are a commented block. With the stand-in,
standoff 120 m lost poacher1 in seed 1000 of four (`earshot eval --runs 4 --seed 1000`:
poacher1 confirmed 110, 145, 115 s after the fence in seeds 1001-1003, as with 30 m; flights,
false verifies and channel load unchanged), so it stays with YOLO. Kept for every run: Range
Rovers for the reserve's vehicles (a white one for the poachers), a detector's camera aimed at
the hunted class's height at the standoff (the stand-in keeps the drone's tilt and its 300 m),
YOLO hits geolocated, `EARSHOT_CV_DUMP`. Not tried: a larger COCO model (yolo11s/m: a
download), a gimbal sweep while holding, a detector trained on these renders.
