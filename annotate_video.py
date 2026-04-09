"""
Draw bounding boxes + trajectory-prediction polylines onto a video,
and write an annotated JSON with predicted future centers per track.

Usage:
    python annotate_video.py \
        --input   124441_10-13min.mp4 \
        --annot   sam3_124441_10_13.json \
        --output  124441_annotated.mp4 \
        --json-out 124441_annotated.json \
        --horizon 30 \
        --lookback 5
"""

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np


# ── visual config ─────────────────────────────────────────────────────────────

CLASS_COLORS = {
    0: (50,  205,  50),   # person → green  (BGR)
    1: (255, 140,   0),   # car    → orange (BGR)
}
CLASS_NAMES = {0: "person", 1: "car"}


# ── helpers ───────────────────────────────────────────────────────────────────

def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def estimate_velocity(history, lookback):
    """
    history : list of (frame_idx, box) ordered oldest→newest (last `lookback` kept)
    Returns (vx, vy) pixels/frame.
    """
    if len(history) < 2:
        return 0.0, 0.0
    f0, box0 = history[0]
    f1, box1 = history[-1]
    dt = f1 - f0
    if dt == 0:
        return 0.0, 0.0
    cx0, cy0 = box_center(box0)
    cx1, cy1 = box_center(box1)
    return (cx1 - cx0) / dt, (cy1 - cy0) / dt


def predict_points(cx, cy, vx, vy, horizon):
    """Return list of (x, y) for frames 1..horizon ahead."""
    return [(cx + k * vx, cy + k * vy) for k in range(1, horizon + 1)]


def draw_dashed_polyline(img, points, color, thickness=1, dash=4, gap=3):
    """Draw a dashed polyline through `points` (list of (x,y) floats)."""
    pts = [(int(round(x)), int(round(y))) for x, y in points]
    segment_px = 0
    drawing   = True
    for i in range(len(pts) - 1):
        p1, p2 = pts[i], pts[i + 1]
        dx, dy  = p2[0] - p1[0], p2[1] - p1[1]
        dist    = max(1, int((dx**2 + dy**2) ** 0.5))
        for d in range(dist):
            if drawing and segment_px < dash:
                frac = d / dist
                x = int(p1[0] + frac * dx)
                y = int(p1[1] + frac * dy)
                cv2.circle(img, (x, y), thickness, color, -1)
                segment_px += 1
            elif not drawing and segment_px < gap:
                segment_px += 1
            else:
                drawing = not drawing
                segment_px = 0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",    default="124441_10-13min.mp4")
    ap.add_argument("--annot",    default="sam3_124441_10_13.json")
    ap.add_argument("--output",   default="124441_annotated.mp4")
    ap.add_argument("--json-out", default="124441_annotated.json")
    ap.add_argument("--horizon",  type=int, default=30)
    ap.add_argument("--lookback", type=int, default=5)
    ap.add_argument("--near-miss", default=None,
                    help="Path to near_misses.json for near-miss overlay")
    args = ap.parse_args()

    # ── load annotations ──────────────────────────────────────────────────────
    print(f"Loading {args.annot} ...")
    with open(args.annot) as f:
        data = json.load(f)

    # per-frame lookup
    frame_annots = {}                       # frame_idx -> [(tid, box, cls)]
    for entry in data:
        fidx = entry["frame_idx"]
        frame_annots[fidx] = list(zip(entry["ids"],
                                      entry["boxes_xyxy"],
                                      entry["classes"]))

    # ── load near-miss events (optional) ─────────────────────────────────────
    nm_by_frame = defaultdict(list)
    if args.near_miss:
        with open(args.near_miss) as f:
            nm_events = json.load(f)
        for e in nm_events:
            # show warning from detection frame through closest approach
            for fi in range(e["frame_idx"], e["frame_idx"] + e["ttca_frames"] + 1):
                nm_by_frame[fi].append(e)
        print(f"Loaded {len(nm_events)} near-miss events from {args.near_miss}")

    # ── open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")

    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps  = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    print(f"Video: {W}x{H}  {fps:.1f}fps  {total} frames")
    print(f"Prediction horizon: {args.horizon} frames  |  "
          f"Velocity lookback: {args.lookback} frames")

    # per-track rolling history: tid -> deque of (frame_idx, box)
    track_history = defaultdict(list)

    # accumulates the annotated JSON output
    annotated_records = []

    # ── process frame by frame ────────────────────────────────────────────────
    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annots = frame_annots.get(fidx, [])

        # update history for every active track this frame
        for tid, box, cls in annots:
            hist = track_history[tid]
            hist.append((fidx, box))
            if len(hist) > args.lookback:
                hist.pop(0)

        # ── build per-frame prediction data & JSON record ─────────────────
        trajectories = []
        pred_cache   = {}   # tid -> (cx, cy, vx, vy, future)

        for tid, box, cls in annots:
            hist       = track_history[tid]
            cx, cy     = box_center(box)
            vx, vy     = estimate_velocity(hist, args.lookback)
            future     = predict_points(cx, cy, vx, vy, args.horizon)
            pred_cache[tid] = (cx, cy, vx, vy, future)

            trajectories.append({
                "id":                tid,
                "velocity_px_frame": [round(vx, 4), round(vy, 4)],
                "predicted_centers": [[round(x, 2), round(y, 2)]
                                       for x, y in future],
            })

        orig_entry = frame_annots.get(fidx, [])
        annotated_records.append({
            "frame_idx":    fidx,
            "boxes_xyxy":   [list(box) for _, box, _ in orig_entry],
            "classes":      [cls       for _, _,   cls in orig_entry],
            "ids":          [tid       for tid, _,  _  in orig_entry],
            "trajectories": trajectories,   # predicted_centers for horizon frames
        })

        # draw: trajectory first (so boxes appear on top)
        overlay = frame.copy()

        for tid, box, cls in annots:
            color = CLASS_COLORS.get(cls, (200, 200, 200))
            cx, cy, vx, vy, future = pred_cache[tid]

            # ── trajectory prediction line (dashed, fading) ───────────────
            # draw from current center through future points
            all_pts = [(cx, cy)] + future

            # fade: bright near current, dim at horizon
            for i in range(len(all_pts) - 1):
                alpha  = 1.0 - i / len(all_pts)         # 1.0 → 0.0
                t_color = tuple(int(c * alpha) for c in color)
                p1 = (int(all_pts[i][0]),     int(all_pts[i][1]))
                p2 = (int(all_pts[i + 1][0]), int(all_pts[i + 1][1]))
                cv2.line(overlay, p1, p2, t_color, 1)

            # endpoint dot
            ep = (int(round(future[-1][0])), int(round(future[-1][1])))
            cv2.circle(overlay, ep, 3, color, -1)

        # blend trajectory overlay (semi-transparent)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # ── draw bounding boxes and labels ────────────────────────────────
        for tid, box, cls in annots:
            color = CLASS_COLORS.get(cls, (200, 200, 200))
            x1, y1, x2, y2 = (int(round(v)) for v in box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            label = f"{CLASS_NAMES.get(cls, cls)} #{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.35, 1)
            # filled label background
            cv2.rectangle(frame,
                          (x1, y1 - th - 4), (x1 + tw + 2, y1),
                          color, -1)
            cv2.putText(frame, label,
                        (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (0, 0, 0), 1, cv2.LINE_AA)

        # ── near-miss overlay ─────────────────────────────────────────────
        active_nms = nm_by_frame.get(fidx, [])
        if active_nms:
            # build tid -> center lookup for this frame
            tid_center = {}
            for tid, box, cls in annots:
                tid_center[tid] = box_center(box)

            for nm in active_nms:
                ca = tid_center.get(nm["track_a_id"])
                cb = tid_center.get(nm["track_b_id"])
                if ca is None or cb is None:
                    continue

                # red dashed line between the two objects
                draw_dashed_polyline(frame, [ca, cb],
                                    (0, 0, 255), thickness=2, dash=8, gap=4)

                # warning label at midpoint
                mx = int((ca[0] + cb[0]) / 2)
                my = int((ca[1] + cb[1]) / 2)
                label = f"NEAR MISS  {nm['min_pred_dist']:.0f}px  {nm['ttca_sec']:.1f}s"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame,
                              (mx - 2, my - th - 6),
                              (mx + tw + 4, my + 4),
                              (0, 0, 200), -1)
                cv2.putText(frame, label, (mx + 1, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # ── frame counter ─────────────────────────────────────────────────
        cv2.putText(frame, f"frame {fidx}/{total-1}",
                    (6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (220, 220, 220), 1, cv2.LINE_AA)

        writer.write(frame)

        if fidx % 500 == 0:
            print(f"  {fidx}/{total} frames done ...")

        fidx += 1

    cap.release()
    writer.release()
    print(f"\nSaved → {args.output}")

    # ── write annotated JSON ──────────────────────────────────────────────────
    with open(args.json_out, "w") as f:
        json.dump(annotated_records, f)
    print(f"Saved → {args.json_out}")


if __name__ == "__main__":
    main()
