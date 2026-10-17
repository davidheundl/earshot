"""Real-model CV demos for Earshot using Ultralytics YOLO.

Install: uv sync --group cv
Examples: uv run python tools/cv_demos.py detector --source image.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DRONE_MODEL = (Path(__file__).resolve().parents[1] /
               'data/cv/verifier-demo-20261015/evaluation/drone-best.pt')


def yolo(args, *, segmentation=False):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install CV dependencies with: uv sync --group cv") from exc
    model_name = args.model
    if segmentation and model_name == "yolo11n.pt":
        model_name = "yolo11n-seg.pt"
    return YOLO(model_name)


def labels(model):
    return {int(k): str(v).lower() for k, v in model.names.items()}


def detector(args):
    model = yolo(args)
    result = model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz,
                           save=args.save, verbose=False)[0]
    names = labels(model)
    out = []
    for box in result.boxes:
        cls = int(box.cls[0])
        out.append({"class": names[cls], "confidence": round(float(box.conf[0]), 3),
                    "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()]})
    print({"demo": "detector", "model": args.model, "detections": out})


def verify(args):
    model = yolo(args)
    names = labels(model)
    if args.target.lower() not in names.values():
        raise SystemExit(f'Target {args.target!r} absent from model classes: {names}')
    streaks = {}
    for frame_no, result in enumerate(model.track(
            source=args.source, conf=args.conf, imgsz=args.imgsz, persist=True,
            tracker=args.tracker, stream=True, verbose=False), 1):
        current = {}
        scores = []
        for box in result.boxes:
            if names[int(box.cls[0])] != args.target.lower():
                continue
            score = float(box.conf[0])
            scores.append(score)
            if box.id is not None and score >= args.verify_conf:
                key = int(box.id[0])
                current[key] = streaks.get(key, 0) + 1
        streaks = current  # a missed frame resets current verification evidence
        print(json.dumps(verification_result(streaks, args.confirm_frames, scores,
                                             frame_no, args.target)))
        if args.max_frames and frame_no >= args.max_frames:
            break


def verification_result(streaks, required, scores, frame, target):
    confirmed = [key for key, count in streaks.items() if count >= required]
    return {'demo': 'verify', 'frame': frame, 'target': target,
            'camera_evidence': 'CONFIRM' if confirmed else 'INCONCLUSIVE',
            'confirmed_track_ids': confirmed, 'consecutive_hits': max(streaks.values(), default=0),
            'required_hits': required, 'max_confidence': max(scores, default=0.0)}


def tracking(args):
    model = yolo(args)
    names = labels(model)
    for frame_no, result in enumerate(model.track(source=args.source, conf=args.conf, persist=True,
                                                  tracker=args.tracker, stream=True, verbose=False,
                                                  imgsz=args.imgsz, save=args.save), 1):
        out = []
        for box in result.boxes:
            cls = int(box.cls[0])
            out.append({"id": int(box.id[0]) if box.id is not None else None,
                        "class": names[cls], "confidence": round(float(box.conf[0]), 3),
                        "bbox_xyxy": box.xyxy[0].tolist()})
        print(json.dumps({"demo": "tracking", "frame": frame_no, "tracks": out}))
        if args.max_frames and frame_no >= args.max_frames:
            break


def nuisance(args):
    model = yolo(args)
    result = model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
    names = labels(model)
    nuisance_classes = {x.strip().lower() for x in args.nuisance.split(",")}
    found = [names[int(box.cls[0])] for box in result.boxes
             if names[int(box.cls[0])] in nuisance_classes]
    drone_seen = any(names[int(box.cls[0])] == args.target.lower() and float(box.conf[0]) >= args.verify_conf
                     for box in result.boxes)
    print({"demo": "nuisance", "nuisances": found, "verify": drone_seen})


def segmentation(args):
    model = yolo(args, segmentation=True)
    result = model.predict(source=args.source, conf=args.conf, imgsz=args.imgsz,
                           save=args.save, verbose=False)[0]
    classes = [model.names[int(c)] for c in result.boxes.cls.tolist()]
    print({"demo": "segmentation", "segments": 0 if result.masks is None else len(result.masks.data),
           "classes": classes})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("demo", choices=["detector", "verify", "tracking", "nuisance", "segmentation"])
    p.add_argument("--source", required=True, help="image, video, webcam index, or stream URL")
    p.add_argument("--model", help="checkpoint; drone weights default for detector/verify/tracking")
    p.add_argument("--target", default="quadcopter")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--verify-conf", type=float, default=0.75)
    p.add_argument("--confirm-frames", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--nuisance", default="airplane,bird,car,truck,boat")
    p.add_argument("--max-frames", type=int, default=20)
    p.add_argument("--save", action="store_true")
    args = p.parse_args()
    if args.confirm_frames < 1 or args.max_frames < 0:
        p.error('confirm-frames must be positive and max-frames nonnegative (0 = all)')
    if not 0 <= args.conf <= 1 or not 0 <= args.verify_conf <= 1:
        p.error('confidence thresholds must lie in [0, 1]')
    if args.demo == 'verify' and args.conf > args.verify_conf:
        p.error('--conf cannot exceed --verify-conf')
    if args.model is None:
        args.model = (str(DRONE_MODEL) if args.demo in ('detector', 'verify', 'tracking')
                      else 'yolo11n.pt')
    {"detector": detector, "verify": verify, "tracking": tracking,
     "nuisance": nuisance, "segmentation": segmentation}[args.demo](args)


if __name__ == "__main__":
    main()
