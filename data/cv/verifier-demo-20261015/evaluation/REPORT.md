# Drone detector evaluation — 2026-10-15

Model: [TomSmail/drone-yolo-v1](https://huggingface.co/TomSmail/drone-yolo-v1),
YOLOv8n with quadcopter and fixed-wing classes. Credit: FRAID Labs / TomSmail;
model card license: CC-BY-4.0. The checkpoint is unmodified. See summary.json for SHA256.

30 controlled Webots views, original 400×240 RGB images, inference size 640,
CPU, confidence >= 0.25. No fine-tuning and no threshold selection on this set.
Only quadcopter counts as the correct class for the drone Source.

## Visually reviewed presence results

| Horizontal distance | Drone correctly indicated | Missed |
|---|---:|---:|
| 5 m | 4/4 | 0 |
| 15 m | 3/4 | 1 |
| 30 m | 1/4 | 3 |
| 60 m | 0/4 | 4 |
| 90 m | 0/4 | 4 |

8/20 positive images had a quadcopter prediction centered in the reference box;
contact-sheet review confirmed the predicted boxes surround the visible drone.
12/20 positive images had no predictions. The 10 empty images had zero predictions,
including zero fixed-wing predictions. This small sample cannot establish a real-world
false-alarm rate. Predictions on positive frames ranged from 0.285 to 0.762 confidence.

## Annotation caveat

Strict IoU >= 0.5 matching gives **0/20**, because simulator geometry boxes are much
taller than the visible drone body. For example frame 00 reference box is approximately
99×100 px while the predicted visible-body box is 81×35 px. Strict IoU failures must
not be described as 20 visual misses. The table uses the explicitly looser center-in-box
diagnostic plus visual inspection; it is not AP or standard localization recall.

In contact-sheet.jpg, green is the original simulator reference and red is the prediction.
Raw outputs, confidence, boxes, overlap and per-image inference times are retained in
predictions.json. No original annotations were altered to fit predictions.

## Next experiment

Correct the reference boxes independently using visible-body manual annotations or
simulator instance masks, then capture the same poses at a higher native camera
resolution (e.g. 1280×720, keeping horizontal FOV fixed) and compare with this baseline.
Increasing inference size alone cannot recover details absent from the 400×240 frame.
Evaluate on additional scenes, off-center positions, occlusion, lighting and drone meshes
before considering fine-tuning. Keep this small set for evaluation rather than training.

This is an offline detector evaluation, not deployment in the Agent or Belief pipeline.
No-detection still must not be interpreted as proof of absence.

## Reproduce

From Earshot with the optional cv dependencies installed:

```sh
.venv/bin/python data/cv/verifier-demo-20261015/evaluation/evaluate_earshot_drone.py \
  --data data/cv/verifier-demo-20261015 \
  --model data/cv/verifier-demo-20261015/evaluation/drone-best.pt \
  --out /tmp/earshot-drone-evaluation-repeat
```

Model download: https://huggingface.co/TomSmail/drone-yolo-v1/resolve/main/best.pt
The bundled checkpoint hash pins the exact weights used even if upstream main changes.
