"""
pseudo_label_pipeline.py

Real skeleton for:
    YOLO detect/track (BoT-SORT + ReID)
    + SAM3 visual propagation for short occlusion gaps
    + pseudo-label export

Tested design assumptions:
- Ultralytics YOLO `model.track(..., tracker="botsort.yaml"/custom_yaml, persist=True)`
- Results expose boxes.xyxy, boxes.conf, boxes.cls, boxes.id
- Ultralytics SAM3 `SAM3VideoPredictor(...)(source=video, bboxes=[...], stream=True)`

Install:
    pip install -U ultralytics opencv-python numpy

Usage:
    python pseudo_label_pipeline.py \
        --video input.mp4 \
        --yolo runs/detect/train/weights/best.pt \
        --sam sam3.pt \
        --out pseudo_labels.json \
        --classes bicycle motorcycle person scooter
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.models.sam import SAM3VideoPredictor


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Detection:
    frame_idx: int
    box: List[float]                    # [x1, y1, x2, y2]
    conf: float
    cls_id: int
    cls_name: str
    track_id: Optional[int] = None
    source: str = "yolo"               # yolo | sam3
    quality: str = "strong"            # strong | weak


@dataclass
class TrackState:
    track_id: int
    detections: Dict[int, Detection] = field(default_factory=dict)
    class_votes: Counter = field(default_factory=Counter)

    def add(self, det: Detection) -> None:
        self.detections[det.frame_idx] = det
        self.class_votes[det.cls_name] += 1

    @property
    def frame_indices(self) -> List[int]:
        return sorted(self.detections.keys())

    def majority_class(self) -> Tuple[Optional[str], float]:
        total = sum(self.class_votes.values())
        if total == 0:
            return None, 0.0
        cls_name, count = self.class_votes.most_common(1)[0]
        return cls_name, count / total


# -----------------------------
# Utility functions
# -----------------------------
def valid_box(box: Iterable[float], min_area: float = 40.0) -> bool:
    x1, y1, x2, y2 = map(float, box)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return (w * h) >= min_area and w > 1 and h > 1


def iou_xyxy(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter + 1e-9
    return inter / union


def mask_to_xyxy(mask: np.ndarray) -> Optional[List[float]]:
    """
    Convert binary mask -> [x1, y1, x2, y2].
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return [float(x1), float(y1), float(x2), float(y2)]


def ensure_botsort_reid_yaml(path: Path) -> Path:
    """
    Writes a minimal BoT-SORT config with ReID enabled.
    """
    yaml_text = """tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.10
new_track_thresh: 0.25
track_buffer: 30
match_thresh: 0.80
fuse_score: true

gmc_method: sparseOptFlow

proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: true
model: auto
"""
    path.write_text(yaml_text, encoding="utf-8")
    return path


def extract_video_clip(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    clip_path: str | Path,
    codec: str = "mp4v",
) -> Tuple[float, int, int]:
    """
    Extract inclusive [start_frame, end_frame] clip.
    Returns (fps, width, height).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(clip_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()
    return fps, width, height


# -----------------------------
# Main pipeline
# -----------------------------
class PseudoLabelPipeline:
    def __init__(
        self,
        yolo_weights: str,
        sam_weights: str = "sam3.pt",
        tracker_yaml: str = "botsort_reid.yaml",
        allowed_classes: Optional[List[str]] = None,
        det_conf: float = 0.15,
        seed_conf: float = 0.55,
        max_gap: int = 12,
        sam_conf: float = 0.25,
        half: bool = True,
    ) -> None:
        self.det_conf = det_conf
        self.seed_conf = seed_conf
        self.max_gap = max_gap

        self.yolo = YOLO(yolo_weights)
        self.names = self._normalize_names(self.yolo.names)
        self.class_ids = self._resolve_class_ids(allowed_classes)

        self.tracker_yaml = ensure_botsort_reid_yaml(Path(tracker_yaml))

        self.sam_predictor = SAM3VideoPredictor(
            overrides=dict(
                conf=sam_conf,
                task="segment",
                mode="predict",
                model=sam_weights,
                half=half,
                save=False,
                verbose=False,
            )
        )

    @staticmethod
    def _normalize_names(names) -> Dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(v) for i, v in enumerate(names)}
        raise TypeError(f"Unexpected names type: {type(names)}")

    def _resolve_class_ids(self, allowed_classes: Optional[List[str]]) -> Optional[List[int]]:
        if not allowed_classes:
            return None
        wanted = {c.strip() for c in allowed_classes}
        ids = [idx for idx, name in self.names.items() if name in wanted]
        if not ids:
            raise ValueError(
                f"No allowed classes matched model names. wanted={sorted(wanted)} "
                f"available={list(self.names.values())[:20]}"
            )
        return ids

    # -------------------------
    # 1) Pure detector function
    # -------------------------
    def yolo_detect(self, frame: np.ndarray, frame_idx: int = -1) -> List[Detection]:
        result = self.yolo.predict(
            source=frame,
            conf=self.det_conf,
            classes=self.class_ids,
            verbose=False,
        )[0]
        return self._result_to_detections(result, frame_idx, expect_track_ids=False)

    # -------------------------
    # 2) Detector + BoT-SORT
    # -------------------------
    def bot_sort_update(self, frame: np.ndarray, frame_idx: int) -> List[Detection]:
        result = self.yolo.track(
            source=frame,
            conf=self.det_conf,
            classes=self.class_ids,
            tracker=str(self.tracker_yaml),
            persist=True,
            verbose=False,
        )[0]
        return self._result_to_detections(result, frame_idx, expect_track_ids=True)

    def _result_to_detections(
        self,
        result,
        frame_idx: int,
        expect_track_ids: bool,
    ) -> List[Detection]:
        result = result.cpu()
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.numpy()
        confs = result.boxes.conf.numpy()
        clses = result.boxes.cls.numpy().astype(int)

        if expect_track_ids and result.boxes.id is not None:
            track_ids = result.boxes.id.numpy().astype(int)
        else:
            track_ids = np.array([None] * len(xyxy), dtype=object)

        dets: List[Detection] = []
        for i in range(len(xyxy)):
            box = xyxy[i].tolist()
            if not valid_box(box):
                continue
            cls_id = int(clses[i])
            dets.append(
                Detection(
                    frame_idx=frame_idx,
                    box=[float(v) for v in box],
                    conf=float(confs[i]),
                    cls_id=cls_id,
                    cls_name=self.names.get(cls_id, str(cls_id)),
                    track_id=None if track_ids[i] is None else int(track_ids[i]),
                    source="yolo",
                    quality="strong" if float(confs[i]) >= self.seed_conf else "weak",
                )
            )
        return dets

    # -------------------------
    # 3) First pass: build tracklets
    # -------------------------
    def build_tracks(self, video_path: str | Path) -> Tuple[Dict[int, TrackState], float, int]:
        tracks: Dict[int, TrackState] = {}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="YOLO tracking", unit="frame")
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tracked = self.bot_sort_update(frame, frame_idx)

            for det in tracked:
                if det.track_id is None:
                    continue
                if det.track_id not in tracks:
                    tracks[det.track_id] = TrackState(track_id=det.track_id)
                tracks[det.track_id].add(det)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        return tracks, fps, total_frames

    # -------------------------
    # 4) Find short gaps
    # -------------------------
    def find_short_gaps(self, track: TrackState) -> List[Tuple[int, int]]:
        """
        Returns list of (prev_seen_frame, next_seen_frame) where there is a short gap between them.
        """
        frames = track.frame_indices
        gaps: List[Tuple[int, int]] = []
        for a, b in zip(frames, frames[1:]):
            gap = b - a - 1
            if 1 <= gap <= self.max_gap:
                # Require a strong seed frame to launch SAM3
                if track.detections[a].conf >= self.seed_conf and valid_box(track.detections[a].box):
                    gaps.append((a, b))
        return gaps

    # -------------------------
    # 5) SAM3 propagation on clip
    # -------------------------
    def sam3_propagate(
        self,
        video_path: str | Path,
        start_frame: int,
        end_frame: int,
        seed_bbox: List[float],
    ) -> Dict[int, Dict[str, List[float]]]:
        """
        Runs SAM3 on an extracted clip [start_frame, end_frame] inclusive.
        The seed box is applied to clip frame 0.

        Returns:
            {
                global_frame_idx: {
                    "box": [x1, y1, x2, y2]
                },
                ...
            }
        """
        tmpdir = Path(tempfile.mkdtemp(prefix="sam3_gap_"))
        clip_path = tmpdir / "clip.mp4"

        try:
            extract_video_clip(video_path, start_frame, end_frame, clip_path)

            # SAM3 visual-prompt propagation on the clip
            results_iter = self.sam_predictor(
                source=str(clip_path),
                bboxes=[seed_bbox],
                stream=True,
            )

            propagated: Dict[int, Dict[str, List[float]]] = {}
            prev_box = list(seed_bbox)

            for local_idx, r in enumerate(results_iter):
                global_idx = start_frame + local_idx
                r = r.cpu()

                candidate_boxes: List[List[float]] = []
                candidate_masks: List[np.ndarray] = []

                if r.boxes is not None and len(r.boxes) > 0:
                    candidate_boxes = r.boxes.xyxy.numpy().tolist()

                if r.masks is not None and r.masks.data is not None:
                    candidate_masks = [(m > 0.5).astype(np.uint8) for m in r.masks.data.numpy()]

                    # If boxes are missing, derive from masks
                    if not candidate_boxes:
                        for m in candidate_masks:
                            box = mask_to_xyxy(m)
                            if box is not None:
                                candidate_boxes.append(box)

                if not candidate_boxes:
                    continue

                # Pick instance closest to previous box by IoU
                best_idx = int(np.argmax([iou_xyxy(prev_box, b) for b in candidate_boxes]))
                best_box = [float(v) for v in candidate_boxes[best_idx]]

                if valid_box(best_box):
                    propagated[global_idx] = {"box": best_box}
                    prev_box = best_box

            return propagated

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # -------------------------
    # 6) Fuse YOLO + SAM3
    # -------------------------
    def build_pseudo_labels(self, video_path: str | Path) -> List[dict]:
        tracks, fps, total_frames = self.build_tracks(video_path)

        name_to_id = {v: k for k, v in self.names.items()}
        pseudo_labels: List[dict] = []

        for track_id, track in tqdm(tracks.items(), desc="Building pseudo-labels", unit="track"):
            final_cls, class_ratio = track.majority_class()
            if final_cls is None:
                continue
            final_cls_id = name_to_id.get(final_cls, -1)

            # Add all YOLO-backed detections first
            for frame_idx in track.frame_indices:
                det = track.detections[frame_idx]
                pseudo_labels.append(
                    {
                        "frame_idx": frame_idx,
                        "track_id": track_id,
                        "cls_id": final_cls_id,
                        "box": [round(v, 2) for v in det.box],
                        "conf": float(det.conf),
                    }
                )

            # Fill short gaps with SAM3
            for prev_frame, next_frame in self.find_short_gaps(track):
                seed_conf = float(track.detections[prev_frame].conf)
                seed_box = track.detections[prev_frame].box
                propagated = self.sam3_propagate(
                    video_path=video_path,
                    start_frame=prev_frame,
                    end_frame=next_frame,
                    seed_bbox=seed_box,
                )

                # only use the missing frames in the middle
                for frame_idx in range(prev_frame + 1, next_frame):
                    if frame_idx in track.detections:
                        continue

                    out = propagated.get(frame_idx)
                    if out is None:
                        continue

                    box = out["box"]
                    if not valid_box(box):
                        continue

                    pseudo_labels.append(
                        {
                            "frame_idx": frame_idx,
                            "track_id": track_id,
                            "cls_id": final_cls_id,
                            "box": [round(v, 2) for v in box],
                            "conf": seed_conf,  # carry forward seed confidence
                        }
                    )

        # stable ordering
        pseudo_labels.sort(key=lambda x: (x["frame_idx"], x["track_id"]))
        return pseudo_labels

    # -------------------------
    # 7) Export
    # -------------------------
    def run(
        self,
        video_path: str | Path,
        out_json: str | Path,
        out_video: Optional[str | Path] = None,
    ) -> None:
        labels = self.build_pseudo_labels(video_path)

        # Group per-detection entries into per-frame entries
        frame_map: dict = defaultdict(list)
        for lbl in labels:
            frame_map[lbl["frame_idx"]].append(lbl)

        output = []
        for frame_idx in sorted(frame_map.keys()):
            entries = frame_map[frame_idx]
            output.append(
                {
                    "frame_idx": frame_idx,
                    "ids": [e["track_id"] for e in entries],
                    "boxes_xyxy": [e["box"] for e in entries],
                    "scores": [e["conf"] for e in entries],
                    "classes": [e["cls_id"] for e in entries],
                    "n_masks": len(entries),
                }
            )

        out_json = Path(out_json)
        out_json.write_text(json.dumps(output), encoding="utf-8")
        print(f"Saved {len(output)} frames ({len(labels)} detections) to: {out_json}")

        if out_video is not None:
            self._write_video(video_path, frame_map, Path(out_video))

    def _write_video(
        self,
        video_path: str | Path,
        frame_map: dict,
        out_video: Path,
    ) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        # Assign a consistent color per track_id
        color_cache: Dict[int, Tuple[int, int, int]] = {}

        def _color(track_id: int) -> Tuple[int, int, int]:
            if track_id not in color_cache:
                rng = np.random.default_rng(track_id)
                color_cache[track_id] = tuple(int(c) for c in rng.integers(80, 230, size=3))
            return color_cache[track_id]

        frame_idx = 0
        for _ in tqdm(range(total_frames), desc="Writing video", unit="frame"):
            ok, frame = cap.read()
            if not ok:
                break

            for entry in frame_map.get(frame_idx, []):
                x1, y1, x2, y2 = (int(v) for v in entry["box"])
                tid = entry["track_id"]
                cls_id = entry["cls_id"]
                conf = entry["conf"]
                color = _color(tid)
                label = f"{self.names.get(cls_id, cls_id)} #{tid} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(
                    frame, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

            writer.write(frame)
            frame_idx += 1

        writer.release()
        cap.release()
        print(f"Saved video to: {out_video}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--yolo", required=True, help="Path to YOLO weights")
    parser.add_argument("--sam", default="sam3.pt", help="Path to SAM3 weights")
    parser.add_argument("--out", default="pseudo_labels.json", help="Output JSON")
    parser.add_argument("--out-video", default=None, help="Output video path (optional)")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help='Allowed class names, e.g. --classes person bicycle motorcycle scooter',
    )
    parser.add_argument("--det-conf", type=float, default=0.15)
    parser.add_argument("--seed-conf", type=float, default=0.55)
    parser.add_argument("--max-gap", type=int, default=12)
    parser.add_argument("--tracker-yaml", default="botsort_reid.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = PseudoLabelPipeline(
        yolo_weights=args.yolo,
        sam_weights=args.sam,
        tracker_yaml=args.tracker_yaml,
        allowed_classes=args.classes,
        det_conf=args.det_conf,
        seed_conf=args.seed_conf,
        max_gap=args.max_gap,
    )
    pipeline.run(args.video, args.out, out_video=args.out_video)


if __name__ == "__main__":
    main()