"""
Compare two tracker JSON outputs (no ground truth).

Metrics computed per frame then averaged:
  - Detection count per tracker
  - Matched pairs via Hungarian algorithm (IoU-based)
  - Mean IoU of matched boxes
  - Unmatched boxes (FP-like / FN-like relative to each other)

Track-level metrics:
  - Number of unique track IDs
  - Track length distribution (mean, median, min, max)
  - ID switches (track ID disappears then reappears)

Usage:
    python compare_trackers.py \
        --a  sam3_2_09_084511_3min.json \
        --b  yolo/yolo_2_09_084511_3min_predictions.json \
        --iou-thr 0.5
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between every pair of boxes. Returns (N, M) matrix."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# Load JSON
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    """Load tracker JSON, return dict: frame_idx -> {"boxes": np.ndarray, "ids": list}"""
    with open(path) as f:
        data = json.load(f)

    frames = {}
    for frame in data:
        idx = frame["frame_idx"]
        boxes = np.array(frame.get("boxes_xyxy", []), dtype=float)
        ids   = frame.get("ids", [])
        frames[idx] = {"boxes": boxes, "ids": ids}
    return frames


# ---------------------------------------------------------------------------
# Track-level stats
# ---------------------------------------------------------------------------

def track_stats(frames: dict) -> dict:
    """Compute per-track length and ID-switch count."""
    track_lengths = defaultdict(int)   # id -> total frames seen
    track_last    = {}                 # id -> last frame seen
    id_switches   = 0

    for fidx in sorted(frames):
        for tid in frames[fidx]["ids"]:
            track_lengths[tid] += 1
            if tid in track_last and track_last[tid] < fidx - 1:
                id_switches += 1       # gap: disappeared then reappeared
            track_last[tid] = fidx

    lengths = list(track_lengths.values())
    return {
        "n_tracks":   len(lengths),
        "id_switches": id_switches,
        "len_mean":   float(np.mean(lengths)) if lengths else 0,
        "len_median": float(np.median(lengths)) if lengths else 0,
        "len_min":    int(np.min(lengths)) if lengths else 0,
        "len_max":    int(np.max(lengths)) if lengths else 0,
    }


# ---------------------------------------------------------------------------
# Per-frame comparison
# ---------------------------------------------------------------------------

def compare_frames(frames_a: dict, frames_b: dict, iou_thr: float) -> dict:
    all_frames = sorted(set(frames_a) | set(frames_b))

    counts_a, counts_b = [], []
    matched_ious = []
    unmatched_a  = []   # boxes in A with no match in B
    unmatched_b  = []   # boxes in B with no match in A

    for fidx in all_frames:
        fa = frames_a.get(fidx, {"boxes": np.zeros((0, 4)), "ids": []})
        fb = frames_b.get(fidx, {"boxes": np.zeros((0, 4)), "ids": []})

        ba, bb = fa["boxes"], fb["boxes"]
        counts_a.append(len(ba))
        counts_b.append(len(bb))

        if len(ba) == 0 or len(bb) == 0:
            unmatched_a.append(len(ba))
            unmatched_b.append(len(bb))
            continue

        iou = iou_matrix(ba, bb)
        row_ind, col_ind = linear_sum_assignment(-iou)  # maximise IoU

        matched = 0
        for r, c in zip(row_ind, col_ind):
            if iou[r, c] >= iou_thr:
                matched_ious.append(iou[r, c])
                matched += 1

        unmatched_a.append(len(ba) - matched)
        unmatched_b.append(len(bb) - matched)

    n = len(all_frames)
    return {
        "n_frames":        n,
        "mean_count_a":    float(np.mean(counts_a)),
        "mean_count_b":    float(np.mean(counts_b)),
        "mean_iou":        float(np.mean(matched_ious)) if matched_ious else 0.0,
        "match_rate":      len(matched_ious) / max(1, sum(max(ca, cb) for ca, cb in zip(counts_a, counts_b))),
        "mean_unmatched_a": float(np.mean(unmatched_a)),
        "mean_unmatched_b": float(np.mean(unmatched_b)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a",       required=True, help="First tracker JSON (e.g. sam3)")
    ap.add_argument("--b",       required=True, help="Second tracker JSON (e.g. yolo)")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for matching")
    ap.add_argument("--name-a",  default="TrackerA")
    ap.add_argument("--name-b",  default="TrackerB")
    args = ap.parse_args()

    print(f"Loading {args.a} ...")
    frames_a = load_json(args.a)
    print(f"Loading {args.b} ...")
    frames_b = load_json(args.b)

    stats_a = track_stats(frames_a)
    stats_b = track_stats(frames_b)
    frame_cmp = compare_frames(frames_a, frames_b, args.iou_thr)

    col = 28
    print(f"\n{'='*60}")
    print(f"  Tracker comparison  (IoU threshold = {args.iou_thr})")
    print(f"{'='*60}")
    print(f"{'':30s}  {args.name_a:>12s}  {args.name_b:>12s}")
    print(f"{'-'*60}")
    print(f"{'Unique tracks':{col}s}  {stats_a['n_tracks']:>12d}  {stats_b['n_tracks']:>12d}")
    print(f"{'ID switches':{col}s}  {stats_a['id_switches']:>12d}  {stats_b['id_switches']:>12d}")
    print(f"{'Track len mean (frames)':{col}s}  {stats_a['len_mean']:>12.1f}  {stats_b['len_mean']:>12.1f}")
    print(f"{'Track len median':{col}s}  {stats_a['len_median']:>12.1f}  {stats_b['len_median']:>12.1f}")
    print(f"{'Track len min':{col}s}  {stats_a['len_min']:>12d}  {stats_b['len_min']:>12d}")
    print(f"{'Track len max':{col}s}  {stats_a['len_max']:>12d}  {stats_b['len_max']:>12d}")
    print(f"{'Avg detections/frame':{col}s}  {frame_cmp['mean_count_a']:>12.2f}  {frame_cmp['mean_count_b']:>12.2f}")
    print(f"{'Avg unmatched/frame':{col}s}  {frame_cmp['mean_unmatched_a']:>12.2f}  {frame_cmp['mean_unmatched_b']:>12.2f}")
    print(f"{'-'*60}")
    print(f"{'Mean IoU (matched pairs)':{col}s}  {frame_cmp['mean_iou']:>27.4f}")
    print(f"{'Match rate':{col}s}  {frame_cmp['match_rate']:>27.4f}")
    print(f"{'='*60}")
    print()
    print("Interpretation:")
    print(f"  Higher track len mean  → tracker maintains tracks longer")
    print(f"  Lower ID switches      → more stable track IDs")
    print(f"  Higher mean IoU        → boxes agree well between trackers")
    print(f"  Higher avg unmatched   → tracker finds objects the other misses")


if __name__ == "__main__":
    main()
