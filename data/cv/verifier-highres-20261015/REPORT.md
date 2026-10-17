# Resolution, verification and tracking experiment

2026-10-15. Offline demo only; production Backend, Agent and Belief code unchanged.

## Matched-resolution test

30 new Webots frames, 1280×768 native capture, same 1-radian horizontal field of view,
aspect ratio, camera poses and drone orientations as verifier-demo-20261015. Red demo
markers hidden. Model: TomSmail/drone-yolo-v1 (same checkpoint as earlier evaluation).
Confidence 0.25, CPU. 20 drone views, 10 target-absent views.

| Distance | 400×240 / inference 640 | 1280×768 / inference 640 | 1280×768 / inference 1280 |
|---|---:|---:|---:|
| 5 m | 4/4 | 4/4 | 4/4 |
| 15 m | 3/4 | 3/4 | 3/4 |
| 30 m | 1/4 | 1/4 | 4/4 |
| 60 m | 0/4 | 0/4 | 2/4 |
| 90 m | 0/4 | 0/4 | 1/4 |
| Total | 8/20 | 8/20 | 14/20 |

Hits require quadcopter class and IoU >=0.5. Green reference boxes now come from paired
target-present/absent RGB differences >20 within the simulator geometry box, independently
of predictions. References for the older low-resolution images are scaled from these
high-resolution boxes, an approximation that matters for tiny objects. They are provisional
pixel labels, not human-certified annotations. The geometry-center diagnostic gives 8, 9,
and 16 hits respectively; see comparison.json. Borderline pixel overlaps should not be
overinterpreted on four examples per distance.

All three configurations produced zero quadcopter false detections on the 10 empty stills.
Those counts do not establish zero false alarms in other scenes: the video below exposes
a persistent fixed-wing false detection. The 1280 evaluation also has some extra quadcopter
boxes on positive scenes (see raw predictions). Median measured inference times were
24.5, 23.0 and 86.5 ms, excluding rendering, tracking and image I/O; these are one CPU run.

Higher-resolution capture helps only if inference retains the detail. This result does not
establish reliable 90 m detection; the object remains small and only 1/4 passes strict scoring.

## Moving sequence

moving.avi: 100 Webots-rendered frames at 10 Hz, fixed camera, a drone moving laterally
at 15 m. Frames 41–50 hide the drone below ground to simulate a one-second visibility
gap; this is a controlled disappearance, not physical occlusion. moving-labels.json stores
per-frame simulator boxes. The annotated tracking video is tracking-annotated.mp4.

ByteTrack + the drone checkpoint, confidence 0.25, inference 1280:
- Target-centered quadcopter track in 85/90 visible frames (center inside geometry box).
- ID 1 before the gap; ID 4 after reacquisition; later ID 5. Identity was not preserved.
- A background tree was classified as fixed-wing and tracked as ID 3, including all ten
  target-absent frames. 68/100 frames contain an unmatched track. This is not reliable
  general drone tracking despite successful execution and many correct localizations.
- View tracking-empty-frame.jpg for the false track. tracking.json has every prediction;
  tracking-matches.json separates target matches from unmatched tracks.

## Verification demo change

tools/cv_demos.py now uses the drone checkpoint by default for detection, tracking and
verification. Verification defaults to target quadcopter, confidence 0.75 and three
consecutive observations of the same tracker ID. Missing or weak evidence resets the
streak and returns INCONCLUSIVE. It never infers absence from a missed detection.
Unknown target classes are rejected. A single image is inconclusive by default; video
is required for the temporal rule. This is a demo policy, not calibrated probability fusion.

The 100-frame video remained INCONCLUSIVE at the conservative default: the maximum
target confidence was 0.7335. This is a tested outcome, not evidence that confirmation is
impossible. Threshold calibration requires a separate, larger validation set. Unit tests
exercise confirmation, insufficient evidence, misses and unrelated track IDs.

## Reproduce

From Earshot (cv dependencies installed):

```sh
.venv/bin/python tools/cv_demos.py tracking \
  --source data/cv/verifier-highres-20261015/moving.avi --imgsz 1280 --max-frames 0 --save
.venv/bin/python tools/cv_demos.py verify \
  --source data/cv/verifier-highres-20261015/moving.avi --imgsz 1280 --max-frames 0
.venv/bin/python tools/cv_demos.py verify \
  --source data/cv/verifier-highres-20261015/images/frame_20.png
```

Capture/evaluation helpers are included in scripts/. They use this machine's project and
/tmp paths; launch.py constructs a temporary world from demo.wbt and never overwrites the
production world. It requires Webots and uses isolated port 1247.

Next: extend negative examples (especially trees), assess a narrower field of view or
closer approach, and calibrate confidence on held-out scenes before connecting camera
decisions to the swarm. The nuisance and semantic-scene segmentation demos remain
separate unfinished work.
