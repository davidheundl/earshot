"""Verifier camera detector: a trained YOLO model on the rendered Verifier view.

World side only: the World feeds it the Verifier's camera images (Webots renders them) and
turns its verdicts into CAMERA SensorReadings; Agents never see an image (ADR 0001).

    det = YoloDetector()                 # lazy: ultralytics loads on the first frame
    det.observe(frame)                   # every rendered tick: {"bgra", "w", "h", "fov", "yaw", "pitch"[, "height"]}
    det.verdict() -> CameraVerdict       # once per 1 s sensor window

Verification is multi-frame, as in Nikolas' tools/cv_demos.py: a box counts once ByteTrack has
followed the same quadcopter for `confirm_frames` consecutive processed frames at
`verify_conf` or more. "Nothing here" is only said after `clear_windows` windows without any
quadcopter box, and only for `clear_radius`, the range the model actually sees a drone at
(data/cv/verifier-highres-20261015/REPORT.md). Anything in between is inconclusive: no reading.

The model is TomSmail/drone-yolo-v1 (YOLOv8n, classes quadcopter / fixed-wing, CC BY 4.0),
fetched by Nikolas into data/cv/verifier-demo-20261015/evaluation/. Needs `uv sync --group cv`.

`classes` maps the model's labels to Target Classes (default: quadcopter -> drone). Labels
of a threat class (anything but nothing) are what a track streak verifies and what keeps a
view from counting as empty; labels mapped to nothing (an elephant, a zebra) are only named
in the verdict (`others`), for the Threat Log; unmapped labels are ignored. A Reserve uses
the COCO model (yolo11n.pt): car, truck, motorcycle, bus -> vehicle, the animals -> nothing.
A frame that says how high the camera is (`height`) also gets each hit's ground range from
the box's bottom edge: where a vehicle stands, not only its bearing.

`EARSHOT_CV_DUMP=<dir>` saves every processed frame with its boxes (JPEG) and a line per frame
in <dir>/frames.jsonl: what the detector saw, for a run that did not confirm what it should.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from earshot.messages import TargetClass

DEFAULT_WEIGHTS = (Path(__file__).resolve().parents[3]
                   / "data/cv/verifier-demo-20261015/evaluation/drone-best.pt")
TARGET = "quadcopter"
MIN_SIGMA_DEG = 2.0


@dataclass(frozen=True)
class CameraVerdict:
    # (bearing deg, sigma deg, confidence, label, ground range m; 0 = a bearing only)
    hits: list[tuple[float, float, float, str, float]]
    clear_radius: float | None  # m looked at and empty; None = inconclusive
    others: tuple[str, ...] = ()  # harmless labels seen this window (classes -> nothing)


def box_bearing(cx: float, cy: float, bw: float, frame: dict) -> tuple[float, float]:
    """Bearing (deg, 0 = north, clockwise) and sigma of a box centre in a camera pitched down.

    Camera axes: x right, y down, z forward; `fov` is horizontal (Webots fieldOfView, rad),
    `pitch` positive down (rad), `yaw` the camera heading in degrees."""
    w, h = frame["w"], frame["h"]
    f = w / (2.0 * math.tan(frame["fov"] / 2.0))
    xc, yc = (cx - w / 2.0) / f, (cy - h / 2.0) / f
    p = frame.get("pitch", 0.0)
    off = math.degrees(math.atan2(xc, math.cos(p) - yc * math.sin(p)))
    sigma = max(MIN_SIGMA_DEG, math.degrees(math.atan2(bw / 2.0, f)))
    return (frame["yaw"] + off) % 360.0, sigma


def ground_range(cx: float, by: float, frame: dict) -> float:
    """Horizontal distance (m) to where the ray through (cx, by), a box's bottom centre,
    meets flat ground `height` m below the camera: a ground target geolocated from the
    gimbal's depression angle. 0 when the frame has no height or the ray does not come down."""
    height = frame.get("height")
    if not height or height <= 0:
        return 0.0
    w, h = frame["w"], frame["h"]
    f = w / (2.0 * math.tan(frame["fov"] / 2.0))
    xc, yc = (cx - w / 2.0) / f, (by - h / 2.0) / f
    p = frame.get("pitch", 0.0)
    ahead, down = math.cos(p) - yc * math.sin(p), math.sin(p) + yc * math.cos(p)
    return height * math.hypot(ahead, xc) / down if down > 1e-3 else 0.0


@dataclass
class YoloDetector:
    weights: str | Path = DEFAULT_WEIGHTS
    conf: float = 0.25  # a box at all
    verify_conf: float = 0.5  # a box that counts towards a track's streak: on Nikolas' 15 m
    #                           video 0.75 never confirmed, 0.5 did by frame 3, none while absent
    confirm_frames: int = 3
    clear_windows: int = 3
    clear_radius: float = 40.0
    imgsz: int | None = None  # default: the image width
    every: int = 1  # process every n-th rendered frame (inference cost vs. streak speed)
    device: str | None = None  # default: EARSHOT_CV_DEVICE, else mps when available, else cpu
    classes: dict | None = None  # model label -> Target Class name; default {TARGET: "drone"}
    model: object | None = None  # injected in tests; else loaded on first use

    _streaks: dict[int, int] = field(default_factory=dict)
    _window_hits: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    _window_boxes: int = 0
    _window_others: set = field(default_factory=set)
    _empty_windows: int = 0
    _frames: int = 0
    _fresh: bool = True  # next track call re-initialises ByteTrack
    infer_ms: list[float] = field(default_factory=list)

    def reset(self) -> None:
        """A new Verifier, or the camera went off: forget tracks and streaks."""
        self._streaks.clear()
        self._window_hits.clear()
        self._window_boxes = 0
        self._window_others.clear()
        self._empty_windows = 0
        self._frames = 0
        self._fresh = True

    def class_of(self, label: str) -> TargetClass:
        """The Target Class a model label stands for (NOTHING if unmapped)."""
        name = self._classes().get(label)
        return TargetClass[name.upper()] if name else TargetClass.NOTHING

    def _classes(self) -> dict[str, str]:
        return {str(k).lower(): str(v).lower()
                for k, v in (self.classes or {TARGET: "drone"}).items()}

    def observe(self, frame: dict) -> None:
        if not frame.get("bgra"):
            return
        self._frames += 1
        if (self._frames - 1) % self.every:
            return
        img = np.frombuffer(frame["bgra"], np.uint8).reshape(frame["h"], frame["w"], 4)[..., :3]
        model = self._model()
        res = model.track(np.ascontiguousarray(img), persist=not self._fresh, conf=self.conf,
                          imgsz=self.imgsz or frame["w"], tracker="bytetrack.yaml",
                          device=self._device(), verbose=False)[0]
        self._fresh = False
        speed = getattr(res, "speed", None) or {}
        if "inference" in speed:
            self.infer_ms.append(float(speed["inference"]))
        names = {int(k): str(v).lower() for k, v in model.names.items()}
        classes = self._classes()
        if os.environ.get("EARSHOT_CV_DUMP"):
            _dump(Path(os.environ["EARSHOT_CV_DUMP"]), img, res, names, frame)
        current: dict[int, int] = {}
        for box in res.boxes:
            label = names.get(int(box.cls[0]))
            if label not in classes:
                continue
            if classes[label] == "nothing":
                self._window_others.add(label)
                continue
            self._window_boxes += 1
            score = float(box.conf[0])
            if box.id is None or score < self.verify_conf:
                continue
            key = int(box.id[0])
            current[key] = self._streaks.get(key, 0) + 1
            if current[key] >= self.confirm_frames:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                b, s = box_bearing((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, frame)
                rng = ground_range((x1 + x2) / 2.0, y2, frame)
                self._window_hits[key] = (b, s, score, label, rng)
        self._streaks = current  # a missed frame resets a track's streak

    def verdict(self, hovering: bool = True) -> CameraVerdict:
        """This window's verified tracks; "nothing" once `clear_windows` windows in a row were
        empty while the Verifier held still (`hovering`)."""
        hits = list(self._window_hits.values())
        if self._window_boxes or not hovering:
            self._empty_windows = 0
        else:
            self._empty_windows += 1
        clear = self.clear_radius if self._empty_windows >= self.clear_windows else None
        others = tuple(sorted(self._window_others))
        self._window_hits.clear()
        self._window_boxes = 0
        self._window_others.clear()
        return CameraVerdict(hits, clear, others)

    def warm(self, w: int = 1280, h: int = 768) -> None:
        """Load the model and run it once on a blank frame now (and fail now: a missing
        tracker package, a device error) rather than on the first Verifier frame."""
        self._model().track(np.zeros((h, w, 3), np.uint8), persist=False, conf=self.conf,
                            imgsz=self.imgsz or w, tracker="bytetrack.yaml",
                            device=self._device(), verbose=False)
        self.reset()

    def _device(self) -> str:
        if self.device is None:
            self.device = os.environ.get("EARSHOT_CV_DEVICE") or _best_device()
        return self.device

    def _model(self):
        if self.model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise RuntimeError("the camera detector needs `uv sync --group cv`") from exc
            if not Path(self.weights).exists():
                raise FileNotFoundError(f"camera detector weights not found: {self.weights}")
            self.model = YOLO(str(self.weights))
        return self.model


def _dump(out: Path, img: np.ndarray, res, names: dict, frame: dict) -> None:
    """One processed frame with its boxes, for debugging (EARSHOT_CV_DUMP)."""
    import json
    import time

    from PIL import Image, ImageDraw

    out.mkdir(parents=True, exist_ok=True)
    name = f"{time.time():.2f}"
    boxes = [{"label": names.get(int(b.cls[0])), "conf": round(float(b.conf[0]), 3),
              "id": None if b.id is None else int(b.id[0]),
              "xyxy": [round(float(v)) for v in b.xyxy[0].tolist()]} for b in res.boxes]
    im = Image.fromarray(img[..., ::-1].copy())
    draw = ImageDraw.Draw(im)
    for b in boxes:
        draw.rectangle(b["xyxy"], outline=(255, 40, 40), width=2)
        draw.text((b["xyxy"][0], b["xyxy"][1] - 11), f"{b['label']} {b['conf']} #{b['id']}",
                  fill=(255, 40, 40))
    im.save(out / f"{name}.jpg", quality=85)
    meta = {k: frame.get(k) for k in ("yaw", "pitch", "height")}
    with open(out / "frames.jsonl", "a") as f:
        f.write(json.dumps({"img": name, **meta, "boxes": boxes}) + "\n")


def _best_device() -> str:
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def load_detector(spec: dict | None) -> YoloDetector | None:
    """`camera:` from a Scenario or the CLI: {"detector": "yolo", ...YoloDetector fields}.
    Anything else (absent, "truth") keeps the ground-truth stand-in."""
    if not spec or spec.get("detector", "truth") != "yolo":
        return None
    kw = {k: v for k, v in spec.items() if k not in ("detector", "standoff", "resolution", "range")}
    return YoloDetector(**kw)
