"""eval_model.py

Evaluate a YOLO model against CVAT 1.1 XML ground truth on a video.

Metrics:
  - Per-class: Precision, Recall, F1, AP50
  - Overall:   mAP50, mAP50-95, MOTA, ID Switches

Usage:
    python eval_model.py \
        --model  runs/hptune/best_overall.pt \
        --video  2_09_084511_3min.mp4 \
        --gt     sam3_2_09_084511_3min_cvat.xml

    # Only evaluate on first 500 frames
    python eval_model.py --model best.pt --max-frames 500
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_TO_CLS: Dict[str, int] = {
    "person":   0,
    "bicycle":  1,
    "car":      2,
    "truck":    7,
}
CLS_NAMES: Dict[int, str] = {v: k for k, v in LABEL_TO_CLS.items()}
TRACK_CLASSES = list(LABEL_TO_CLS.values())


# ---------------------------------------------------------------------------
# Parse CVAT XML
# ---------------------------------------------------------------------------

def parse_cvat_xml(xml_path: str) -> Dict[int, List[dict]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt: Dict[int, List[dict]] = {}

    for track_el in root.findall("track"):
        label  = track_el.get("label", "").lower()
        cls_id = LABEL_TO_CLS.get(label)
        if cls_id is None:
            continue
        track_id = int(track_el.get("id", -1))

        for box_el in track_el.findall("box"):
            if box_el.get("outside", "0") == "1":
                continue
            fidx = int(box_el.get("frame"))
            gt.setdefault(fidx, []).append({
                "cls":      cls_id,
                "track_id": track_id,
                "box":      [
                    float(box_el.get("xtl")),
                    float(box_el.get("ytl")),
                    float(box_el.get("xbr")),
                    float(box_el.get("ybr")),
                ],
            })

    return gt


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, 0:1], b[:, 0])
    y1 = np.maximum(a[:, 1:2], b[:, 1])
    x2 = np.minimum(a[:, 2:3], b[:, 2])
    y2 = np.minimum(a[:, 3:4], b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# AP calculation (11-point interpolation)
# ---------------------------------------------------------------------------

def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        prec_at_rec = [p for r, p in zip(recalls, precisions) if r >= thr]
        ap += max(prec_at_rec) if prec_at_rec else 0.0
    return ap / 11.0


# ---------------------------------------------------------------------------
# Run inference on video
# ---------------------------------------------------------------------------

def run_inference(
    model: YOLO,
    video_path: str,
    conf: float,
    iou: float,
    max_frames: Optional[int],
) -> Dict[int, List[dict]]:
    preds: Dict[int, List[dict]] = {}

    results = model.predict(
        source=video_path,
        stream=True,
        classes=TRACK_CLASSES,
        conf=conf,
        iou=iou,
        verbose=False,
        save=False,
    )

    for frame_idx, r in enumerate(results):
        if max_frames is not None and frame_idx >= max_frames:
            break
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy  = boxes.xyxy.cpu().numpy()
        clses = boxes.cls.int().cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for i in range(len(xyxy)):
            preds.setdefault(frame_idx, []).append({
                "cls":  int(clses[i]),
                "conf": float(confs[i]),
                "box":  xyxy[i].tolist(),
            })

        if frame_idx % 300 == 0:
            print(f"  inference frame {frame_idx} ...")

    return preds


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    gt: Dict[int, List[dict]],
    preds: Dict[int, List[dict]],
    iou_thrs: List[float],
    max_frames: Optional[int],
) -> dict:
    """
    Compute per-class and overall metrics across multiple IoU thresholds.
    Returns a results dict.
    """
    if max_frames is not None:
        gt    = {k: v for k, v in gt.items()    if k < max_frames}
        preds = {k: v for k, v in preds.items() if k < max_frames}

    all_frames   = sorted(set(gt.keys()) | set(preds.keys()))
    all_cls_ids  = sorted(set(
        d["cls"] for dets in gt.values()    for d in dets
    ) | set(
        d["cls"] for dets in preds.values() for d in dets
    ))

    # For AP: collect (conf, is_tp) per class per iou_thr
    # Shape: cls_id -> iou_thr -> list of (conf, matched)
    det_records: Dict[int, Dict[float, List[Tuple[float, bool]]]] = {
        c: {t: [] for t in iou_thrs} for c in all_cls_ids
    }
    gt_counts: Dict[int, int] = defaultdict(int)

    # For MOTA at iou_thrs[0]
    mota_thr = iou_thrs[0]
    fp_total = fn_total = id_sw_total = gt_total = 0
    last_gt_match: Dict[int, int] = {}   # pred_box_idx_global -> gt_track_id

    for fidx in all_frames:
        gt_dets  = gt.get(fidx,    [])
        pr_dets  = preds.get(fidx, [])

        for d in gt_dets:
            gt_counts[d["cls"]] += 1
        gt_total += len(gt_dets)

        if len(gt_dets) == 0:
            for d in pr_dets:
                for thr in iou_thrs:
                    det_records[d["cls"]][thr].append((d["conf"], False))
            fp_total += len(pr_dets)
            continue

        if len(pr_dets) == 0:
            fn_total += len(gt_dets)
            continue

        gt_boxes = np.array([d["box"] for d in gt_dets])
        pr_boxes = np.array([d["box"] for d in pr_dets])
        iou      = iou_matrix(pr_boxes, gt_boxes)

        for thr in iou_thrs:
            row_ind, col_ind = linear_sum_assignment(-iou)
            matched_pr: set = set()
            matched_gt: set = set()
            for r, c in zip(row_ind, col_ind):
                if iou[r, c] >= thr:
                    matched_pr.add(r)
                    matched_gt.add(c)
                    det_records[pr_dets[r]["cls"]][thr].append(
                        (pr_dets[r]["conf"], True)
                    )
            for r, d in enumerate(pr_dets):
                if r not in matched_pr:
                    det_records[d["cls"]][thr].append((d["conf"], False))

            # MOTA at first threshold only
            if thr == mota_thr:
                fp_total += len(pr_dets) - len(matched_pr)
                fn_total += len(gt_dets)  - len(matched_gt)
                for r, c in zip(row_ind, col_ind):
                    if iou[r, c] >= thr:
                        gt_tid = gt_dets[c]["track_id"]
                        if r in last_gt_match and last_gt_match[r] != gt_tid:
                            id_sw_total += 1
                        last_gt_match[r] = gt_tid

    # Compute per-class AP at each threshold
    per_class: Dict[int, dict] = {}
    ap50_list:    List[float] = []
    ap50_95_list: List[float] = []

    for cls_id in all_cls_ids:
        cls_n_gt = gt_counts[cls_id]
        aps: Dict[float, float] = {}

        for thr in iou_thrs:
            records = sorted(det_records[cls_id][thr], key=lambda x: -x[0])
            tp_cum = fp_cum = 0
            recalls, precisions = [], []
            for _, is_tp in records:
                if is_tp:
                    tp_cum += 1
                else:
                    fp_cum += 1
                recalls.append(tp_cum / max(cls_n_gt, 1))
                precisions.append(tp_cum / (tp_cum + fp_cum))
            aps[thr] = compute_ap(recalls, precisions) if records else 0.0

        ap50     = aps.get(0.50, 0.0)
        ap50_95  = float(np.mean([aps[t] for t in iou_thrs]))

        # Precision / Recall at IoU=0.5 (last point on PR curve)
        recs50  = sorted(det_records[cls_id][0.50], key=lambda x: -x[0])
        tp = sum(1 for _, m in recs50 if m)
        fp = sum(1 for _, m in recs50 if not m)
        fn = max(0, cls_n_gt - tp)
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)

        per_class[cls_id] = {
            "name":      CLS_NAMES.get(cls_id, str(cls_id)),
            "n_gt":      cls_n_gt,
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1":        round(2 * precision * recall / (precision + recall + 1e-9), 4),
            "ap50":      round(ap50,    4),
            "ap50_95":   round(ap50_95, 4),
        }
        ap50_list.append(ap50)
        ap50_95_list.append(ap50_95)

    mota = 1.0 - (fp_total + fn_total + id_sw_total) / max(gt_total, 1)

    return {
        "map50":       round(float(np.mean(ap50_list)),    4) if ap50_list    else 0.0,
        "map50_95":    round(float(np.mean(ap50_95_list)), 4) if ap50_95_list else 0.0,
        "mota":        round(float(mota), 4),
        "fp":          fp_total,
        "fn":          fn_total,
        "id_switches": id_sw_total,
        "gt_total":    gt_total,
        "per_class":   per_class,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: dict, model_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")
    print(f"  {'mAP50':<14} {results['map50']:.4f}")
    print(f"  {'mAP50-95':<14} {results['map50_95']:.4f}")
    print(f"  {'MOTA':<14} {results['mota']:.4f}")
    print(f"  {'FP':<14} {results['fp']}")
    print(f"  {'FN':<14} {results['fn']}")
    print(f"  {'ID Switches':<14} {results['id_switches']}")
    print(f"  {'GT boxes':<14} {results['gt_total']}")
    print(f"\n  Per-class (IoU=0.50):")
    print(f"  {'Class':<12} {'GT':>6} {'P':>7} {'R':>7} {'F1':>7} {'AP50':>7} {'AP50-95':>9}")
    print(f"  {'-'*58}")
    for cls_id, m in sorted(results["per_class"].items()):
        print(
            f"  {m['name']:<12} {m['n_gt']:>6} "
            f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
            f"{m['f1']:>7.4f} {m['ap50']:>7.4f} {m['ap50_95']:>9.4f}"
        )
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",      default="runs/hptune/best_overall.pt")
    ap.add_argument("--video",      default="/home/brianko/Visual-Preference/test2/2_09_084511_3min.mp4")
    ap.add_argument("--gt",         default="/home/brianko/Visual-Preference/test2/sam3_2_09_084511_3min_cvat.xml")
    ap.add_argument("--conf",       type=float, default=0.25)
    ap.add_argument("--iou",        type=float, default=0.45)
    ap.add_argument("--max-frames", type=int,   default=None)
    ap.add_argument("--out",        default=None,
                    help="Optional JSON file to save results (default: <model_stem>_eval.json)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"Parsing GT: {args.gt}")
    gt = parse_cvat_xml(args.gt)
    print(f"  {len(gt)} annotated frames")

    print(f"\nRunning inference on: {args.video}")
    preds = run_inference(
        model=model,
        video_path=args.video,
        conf=args.conf,
        iou=args.iou,
        max_frames=args.max_frames,
    )
    print(f"  {len(preds)} frames with detections")

    print("\nEvaluating ...")
    iou_thrs = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05).tolist()]
    results  = evaluate(gt, preds, iou_thrs=iou_thrs, max_frames=args.max_frames)

    print_report(results, args.model)

    # Save JSON
    out_path = args.out or str(Path(args.model).stem) + "_eval.json"
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "conf": args.conf, **results}, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
