"""
Compute tracking & detection metrics from SAM3 predictions vs CVAT XML ground truth.

Metrics:
  - HOTA  (Higher Order Tracking Accuracy)
  - mAP50 (Detection mean Average Precision @ IoU=0.5)
  - Gap events (GT track lost and re-ID'd with a different pred ID)
  - Avg Track Length (mean frames per predicted track)

Usage:
    python eval_metrics.py \
        --pred  result/easy/all/124441_10-13min_sam3.json \
        --gt    data/easy/annotations.xml
"""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

# ── config ────────────────────────────────────────────────────────────────────

CLASS_NAMES = {"person": 0, "bicycle": 1, "car": 2, "truck": 7}
HOTA_ALPHAS = np.arange(0.05, 0.96, 0.05)   # 19 IoU thresholds


# ── IoU ───────────────────────────────────────────────────────────────────────

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(preds, gts):
    """(N,M) IoU matrix."""
    M = np.zeros((len(preds), len(gts)))
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            M[i, j] = iou(p, g)
    return M


# ── load ground truth XML (CVAT interpolation format) ────────────────────────

def load_gt(xml_path):
    """
    Returns:
      gt_frames : dict  frame_idx -> list of (gt_tid, box_xyxy, cls_id)
      gt_tracks : dict  gt_tid -> sorted list of frame_idx
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_frames = defaultdict(list)
    gt_tracks = defaultdict(list)

    for track_el in root.findall("track"):
        tid   = int(track_el.get("id"))
        label = track_el.get("label")
        cls   = CLASS_NAMES.get(label, -1)
        if cls == -1:
            continue

        for box_el in track_el.findall("box"):
            if box_el.get("outside", "0") == "1":
                continue
            fidx = int(box_el.get("frame"))
            box  = [float(box_el.get("xtl")), float(box_el.get("ytl")),
                    float(box_el.get("xbr")), float(box_el.get("ybr"))]
            gt_frames[fidx].append((tid, box, cls))
            gt_tracks[tid].append(fidx)

    # sort frame lists
    for tid in gt_tracks:
        gt_tracks[tid].sort()

    return gt_frames, gt_tracks


# ── load predictions JSON ─────────────────────────────────────────────────────

def load_pred(json_path):
    """
    Returns:
      pred_frames : dict  frame_idx -> list of (pred_tid, box_xyxy, cls_id, score)
      pred_tracks : dict  pred_tid -> sorted list of frame_idx
    """
    with open(json_path) as f:
        data = json.load(f)

    pred_frames = defaultdict(list)
    pred_tracks = defaultdict(list)

    for entry in data:
        fidx = entry["frame_idx"]
        for tid, box, cls in zip(entry["ids"], entry["boxes_xyxy"], entry["classes"]):
            score = entry["scores"][entry["ids"].index(tid)] if "scores" in entry else 1.0
            pred_frames[fidx].append((tid, box, cls, score))
            pred_tracks[tid].append(fidx)

    for tid in pred_tracks:
        pred_tracks[tid].sort()

    return pred_frames, pred_tracks


# ── Hungarian matching for one frame ─────────────────────────────────────────

def match_frame(pred_list, gt_list, iou_thresh):
    """
    pred_list : list of (tid, box, cls, score)
    gt_list   : list of (tid, box, cls)
    Returns   : list of (pred_tid, gt_tid) matched pairs
    """
    if not pred_list or not gt_list:
        return []

    pred_boxes = [p[1] for p in pred_list]
    gt_boxes   = [g[1] for g in gt_list]
    M = iou_matrix(pred_boxes, gt_boxes)

    # only same-class matches allowed
    for i, p in enumerate(pred_list):
        for j, g in enumerate(gt_list):
            if p[2] != g[2]:
                M[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(-M)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if M[r, c] >= iou_thresh:
            matches.append((pred_list[r][0], gt_list[c][0]))
    return matches


# ── HOTA ─────────────────────────────────────────────────────────────────────

def compute_hota(pred_frames, gt_frames, all_frames):
    hota_per_alpha = []

    for alpha in HOTA_ALPHAS:
        tp_total = fp_total = fn_total = 0

        # track all frames each gt/pred track is present
        gt_frames_of_tid   = defaultdict(set)  # gt_tid  -> set of frame_idx where it exists
        pred_frames_of_tid = defaultdict(set)  # pred_tid -> set of frame_idx where it exists
        pair_matched_frames = defaultdict(set)  # (gt_tid, pred_tid) -> set of matched frame_idx

        for fidx in all_frames:
            preds = pred_frames.get(fidx, [])
            gts   = gt_frames.get(fidx, [])

            for g in gts:
                gt_frames_of_tid[g[0]].add(fidx)
            for p in preds:
                pred_frames_of_tid[p[0]].add(fidx)

            matches = match_frame(preds, gts, alpha)

            matched_pred = {m[0] for m in matches}
            matched_gt   = {m[1] for m in matches}

            tp_total += len(matches)
            fp_total += len(preds) - len(matched_pred)
            fn_total += len(gts)  - len(matched_gt)

            for pred_tid, gt_tid in matches:
                pair_matched_frames[(gt_tid, pred_tid)].add(fidx)

        det_a = tp_total / (tp_total + fp_total + fn_total + 1e-9)

        # AssA: TPA-weighted mean of per-pair association score
        ass_num = 0.0
        for (gt_tid, pred_tid), matched in pair_matched_frames.items():
            tpa = len(matched)
            fna = len(gt_frames_of_tid[gt_tid]   - matched)  # gt present, not matched to this pred
            fpa = len(pred_frames_of_tid[pred_tid] - matched) # pred present, not matched to this gt
            ass_num += tpa * (tpa / (tpa + fna + fpa + 1e-9))

        ass_a = ass_num / (tp_total + 1e-9) if tp_total > 0 else 0.0
        hota_per_alpha.append((det_a * ass_a) ** 0.5)

    return float(np.mean(hota_per_alpha))


# ── mAP@0.5:0.95 ──────────────────────────────────────────────────────────────

MAP_THRESHOLDS = np.arange(0.5, 1.0, 0.05)   # 0.50, 0.55, ..., 0.95


def _collect_dets_at_thresh(pred_frames, gt_frames, all_frames, iou_thresh):
    """Return (cls_dets, cls_n_gt) at a single IoU threshold."""
    cls_dets = defaultdict(list)   # cls -> list of (confidence, is_true_positive)
    cls_n_gt = defaultdict(int)    # cls -> total number of GT boxes

    for fidx in all_frames:
        for _, _, cls in gt_frames.get(fidx, []):
            cls_n_gt[cls] += 1

    for fidx in all_frames:
        preds = pred_frames.get(fidx, [])
        gts   = gt_frames.get(fidx, [])

        cls_preds = defaultdict(list)
        cls_gts   = defaultdict(list)
        for p in preds:
            cls_preds[p[2]].append(p)
        for g in gts:
            cls_gts[g[2]].append(g)

        for cls in set(cls_preds.keys()) | set(cls_gts.keys()):
            cp = cls_preds[cls]
            cg = cls_gts[cls]

            if not cp:
                continue
            if not cg:
                for p in cp:
                    cls_dets[cls].append((p[3], False))
                continue

            M = iou_matrix([p[1] for p in cp], [g[1] for g in cg])
            matched_gt = set()
            order = sorted(range(len(cp)), key=lambda i: cp[i][3], reverse=True)
            for i in order:
                best_j, best_iou = -1, -1.0
                for j in range(len(cg)):
                    if j in matched_gt:
                        continue
                    if M[i, j] >= iou_thresh and M[i, j] > best_iou:
                        best_iou = M[i, j]
                        best_j = j
                if best_j >= 0:
                    matched_gt.add(best_j)
                    cls_dets[cls].append((cp[i][3], True))
                else:
                    cls_dets[cls].append((cp[i][3], False))

    return cls_dets, cls_n_gt


def _ap_from_dets(cls_dets, cls_n_gt):
    """101-point interpolated AP averaged over classes with GT."""
    ap_list = []
    for cls, n_gt in cls_n_gt.items():
        if n_gt == 0:
            continue
        dets = sorted(cls_dets[cls], key=lambda x: -x[0])
        if len(dets) == 0:
            ap_list.append(0.0)
            continue
        tp = np.array([1.0 if is_tp else 0.0 for _, is_tp in dets], dtype=float)
        fp = 1.0 - tp
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec  = tp_cum / (n_gt + 1e-9)
        prec = tp_cum / (tp_cum + fp_cum + 1e-9)

        ap = 0.0
        for t in np.linspace(0, 1, 101):
            p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
            ap += p / 101.0
        ap_list.append(ap)

    return float(np.mean(ap_list)) if ap_list else 0.0


def compute_map(pred_frames, gt_frames, all_frames):
    """mAP averaged over IoU thresholds 0.50:0.05:0.95."""
    ap_at_thresh = []
    for iou_thresh in MAP_THRESHOLDS:
        cls_dets, cls_n_gt = _collect_dets_at_thresh(
            pred_frames, gt_frames, all_frames, iou_thresh)
        ap_at_thresh.append(_ap_from_dets(cls_dets, cls_n_gt))
    return float(np.mean(ap_at_thresh))


# ── Gap events ────────────────────────────────────────────────────────────────

def compute_gap_events(pred_frames, gt_frames, all_frames, iou_thresh=0.5):
    """
    A gap event occurs when a GT track disappears from the matched pred_tid
    for ≥1 frame and then reappears with a *different* pred_tid.
    """
    # Per frame: gt_tid -> matched pred_tid
    gt_to_pred_per_frame = {}
    for fidx in sorted(all_frames):
        preds = pred_frames.get(fidx, [])
        gts   = gt_frames.get(fidx, [])
        matches = match_frame(preds, gts, iou_thresh)
        gt_to_pred_per_frame[fidx] = {gt: pred for pred, gt in matches}

    # Per GT track: walk frames in order, detect ID switches after gaps
    gt_tracks_frames = defaultdict(list)
    for fidx in sorted(all_frames):
        for gt_tid, _, _ in gt_frames.get(fidx, []):
            gt_tracks_frames[gt_tid].append(fidx)

    gap_events = 0
    for gt_tid, frames in gt_tracks_frames.items():
        last_pred_id = None
        last_frame   = None
        for fidx in frames:
            cur_pred = gt_to_pred_per_frame.get(fidx, {}).get(gt_tid, None)
            if cur_pred is not None:
                if last_pred_id is not None and cur_pred != last_pred_id:
                    # ID switch — check if there was a gap before
                    if last_frame is not None and fidx > last_frame + 1:
                        gap_events += 1
                last_pred_id = cur_pred
                last_frame   = fidx

    return gap_events


# ── Avg Track Length ──────────────────────────────────────────────────────────

def compute_avg_track_length(pred_tracks):
    lengths = [len(frames) for frames in pred_tracks.values()]
    return float(np.mean(lengths)) if lengths else 0.0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred",    default="result/easy/all/124441_10-13min_sam3.json")
    ap.add_argument("--gt",      default="data/easy/annotations.xml")
    args = ap.parse_args()

    print(f"Loading GT  : {args.gt}")
    gt_frames, gt_tracks = load_gt(args.gt)

    print(f"Loading pred: {args.pred}")
    pred_frames, pred_tracks = load_pred(args.pred)

    all_frames = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))
    print(f"Total frames evaluated: {len(all_frames)}")
    print(f"GT tracks: {len(gt_tracks)}  |  Pred tracks: {len(pred_tracks)}")
    print()

    print("Computing HOTA ...")
    hota = compute_hota(pred_frames, gt_frames, all_frames)

    print("Computing mAP@0.5:0.95 ...")
    map_val = compute_map(pred_frames, gt_frames, all_frames)

    print("Computing gap events ...")
    gaps = compute_gap_events(pred_frames, gt_frames, all_frames)

    avg_len = compute_avg_track_length(pred_tracks)

    print()
    print("=" * 40)
    print(f"  HOTA              : {hota:.4f}")
    print(f"  mAP@0.5:0.95      : {map_val:.4f}")
    print(f"  Gap events        : {gaps}")
    print(f"  Avg Track Length  : {avg_len:.1f} frames")
    print("=" * 40)


if __name__ == "__main__":
    main()
