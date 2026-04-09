from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision.ops import nms as torchvision_nms
from ultralytics import YOLO



TRACK_CLASSES = [0, 1, 2, 3, 36]  # COCO/raw IDs: person, bicycle, car, motorcycle, skateboard
CLASS_NAMES = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    36: "skateboard",
}

# For YOLO training export only: contiguous IDs required by training datasets.
TRAIN_ID_MAP = {
    0:  0,
    1:  1,
    2:  2,
    3:  3,
    36: 4,
}
TRAIN_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "skateboard",
}

# Per-class (min_w/h, max_w/h) aspect-ratio limits
CLASS_ASPECT_LIMITS: dict[int, tuple[float, float]] = {
    0:  (0.15, 1.50),  # person
    1:  (0.50, 3.00),  # bicycle
    2:  (0.50, 4.00),  # car
    3:  (0.40, 3.00),  # motorcycle
    36: (0.30, 4.00),  # skateboard
}

MIN_BOX_AREA_FRAC = 5e-4
MAX_BOX_AREA_FRAC = 0.90


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Detection:
    frame_idx: int
    box: np.ndarray  # [x1, y1, x2, y2] float32
    cls: int
    conf: float


@dataclass
class TrackFrame:
    frame_idx: int
    box: np.ndarray  # [x1, y1, x2, y2]
    conf: float = 0.0  # matched YOLO conf at this frame (0 if no match)


@dataclass
class CandidateTrack:
    obj_id: int
    cls: int
    init_conf: float
    key_frame: int
    frames: list[TrackFrame] = field(default_factory=list)



def track_video(
    source: str | int = 0,
    model_path: str = "yolo26x.pt",
    out_path: str = "predictions.json",
    save_video: bool = True,
):
    """
    Simple YOLO tracking with BotSORT.
    Tracks objects in a video, saves results to JSON and an annotated video.
    """
    model = YOLO(model_path)
    base_path = Path(out_path).with_suffix("")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    video_out = str(base_path.with_suffix(".mp4"))

    cap = cv2.VideoCapture(source if isinstance(source, str) else source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    results = model.track(
        source=source,
        stream=True,
        persist=True,
        tracker="botsort.yaml",
        classes=TRACK_CLASSES,
        verbose=False,
        save=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    all_predictions = []
    writer = None
    for i, r in enumerate(tqdm(results, desc="track", unit="f")):
        boxes = r.boxes
        frame_pred = {"frame_idx": i}
        if boxes is not None and len(boxes) > 0:
            frame_pred["boxes_xyxy"] = boxes.xyxy.cpu().tolist()
            frame_pred["scores"] = boxes.conf.cpu().tolist()
            frame_pred["classes"] = boxes.cls.int().cpu().tolist()
            frame_pred["ids"] = boxes.id.int().cpu().tolist() if boxes.id is not None else []
        else:
            frame_pred.update(boxes_xyxy=[], scores=[], classes=[], ids=[])

        frame_pred["n_masks"] = len(r.masks.data) if r.masks is not None else 0
        all_predictions.append(frame_pred)

        if save_video:
            frame = r.plot()
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    video_out,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
            writer.write(frame)

    if writer is not None:
        writer.release()

    pred_path = str(base_path.with_suffix(".json"))
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f)
    print(f"Saved {len(all_predictions)} frames to {pred_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",     choices=["track"], default="track")
    ap.add_argument("--source",   default="/home/brianko/Visual-Preference/test2/2_09_084511_3min.mp4")
    ap.add_argument("--model",    default="yolo26x.pt")
    ap.add_argument("--out-path", default="yolo/predictions.json")
    ap.add_argument("--save-video", action="store_true")
    args, _ = ap.parse_known_args()

    track_video(
        source=args.source,
        model_path=args.model,
        out_path=args.out_path,
        save_video=args.save_video,
    )
