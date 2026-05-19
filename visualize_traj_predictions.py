"""
Render trajectory predictions onto a video using a CVAT predictions XML
produced by traj_predict_from_cvat.py or transformer_trajectory.py.

For every frame the script draws:
  - Ground-truth bounding boxes      (white/class-colored outline, labelled)
  - Linear-predicted future path     (red     fading line + endpoint dot)
  - Kalman-predicted future path     (blue    fading line + endpoint dot)
  - Transformer-predicted future path (magenta fading line + endpoint dot)
                                     (drawn only when transformer_pred_*
                                      polylines are present in the XML)
  - Ground-truth future path         (green   fading line + endpoint dot)

Usage
-----
    python visualize_traj_predictions.py \
        --video  data/easy/124441_10-13min.mp4 \
        --xml    predictions.xml \
        --output 124441_traj_annotated.mp4
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

import cv2
import numpy as np


# ── CVAT XML parsing + interpolation ─────────────────────────────────────────

def parse_polyline_points(pts_str):
    """'x1,y1;x2,y2;...' -> list of (float, float)"""
    result = []
    pts_str = pts_str.strip()
    if not pts_str:
        return result
    for token in pts_str.split(";"):
        x, y = token.split(",")
        result.append((float(x), float(y)))
    return result


def lerp_boxes(b0, b1, t):
    return tuple(b0[i] + (b1[i] - b0[i]) * t for i in range(4))


def lerp_polylines(p0, p1, t):
    """Linearly interpolate two point lists of equal length."""
    return [
        (
            p0[i][0] + (p1[i][0] - p0[i][0]) * t,
            p0[i][1] + (p1[i][1] - p0[i][1]) * t,
        )
        for i in range(len(p0))
    ]


def build_frame_index(keyframes):
    """
    keyframes : sorted list of (frame_idx, data, outside)
    Returns a callable:
        get(frame_idx) -> data | None
    using CVAT-style linear interpolation between keyframes.
    """
    kf = keyframes

    def get(fidx):
        if not kf:
            return None

        if fidx < kf[0][0]:
            return None
        if fidx > kf[-1][0]:
            return None

        lo = hi = None
        for i, (f, d, out) in enumerate(kf):
            if f <= fidx:
                lo = i
            if f >= fidx and hi is None:
                hi = i

        if lo is None:
            return None

        f_lo, d_lo, out_lo = kf[lo]

        if out_lo:
            return None

        if f_lo == fidx:
            return d_lo

        if hi is None or hi == lo:
            return d_lo

        f_hi, d_hi, out_hi = kf[hi]

        if out_hi:
            return d_lo

        if f_hi == f_lo:
            return d_lo

        t = (fidx - f_lo) / (f_hi - f_lo)

        if isinstance(d_lo, tuple) and len(d_lo) == 4:
            return lerp_boxes(d_lo, d_hi, t)
        elif isinstance(d_lo, list):
            if len(d_lo) == len(d_hi):
                return lerp_polylines(d_lo, d_hi, t)
            return d_lo

        return d_lo

    return get


def parse_predictions_xml(path):
    """
    Parse predictions.xml produced by traj_predict_from_cvat.py.

    Returns
    -------
    objects : list of dicts, one per object, each with:
        {
            "obj_idx": int,
            "label": str,
            "bbox_get": callable,
            "lin_get": callable | None,
            "kalman_get": callable | None,
            "transformer_get": callable | None,
            "gt_get": callable | None,
        }
    """
    tree = ET.parse(path)
    root = tree.getroot()

    all_tracks = []
    for track_el in root.findall("track"):
        tid = int(track_el.get("id"))
        orig_id_str = track_el.get("orig_id")
        orig_id = int(orig_id_str) if orig_id_str is not None else tid
        label = track_el.get("label")

        if track_el.find("box") is not None:
            kf = []
            for box in track_el.findall("box"):
                f = int(box.get("frame"))
                out = box.get("outside", "0") == "1"
                data = (
                    float(box.get("xtl")),
                    float(box.get("ytl")),
                    float(box.get("xbr")),
                    float(box.get("ybr")),
                )
                kf.append((f, data, out))
            kf.sort(key=lambda x: x[0])
            all_tracks.append({
                "id": tid,
                "orig_id": orig_id,
                "label": label,
                "type": "bbox",
                "orig_label": label,
                "get": build_frame_index(kf),
            })

        elif track_el.find("polyline") is not None:
            kf = []
            for pl in track_el.findall("polyline"):
                f = int(pl.get("frame"))
                out = pl.get("outside", "0") == "1"
                pts = parse_polyline_points(pl.get("points", ""))
                kf.append((f, pts, out))
            kf.sort(key=lambda x: x[0])

            if label.startswith("lin_pred_"):
                orig_label = label[len("lin_pred_"):]
                track_type = "lin"
            elif label.startswith("kalman_pred_"):
                orig_label = label[len("kalman_pred_"):]
                track_type = "kalman"
            elif label.startswith("transformer_pred_"):
                orig_label = label[len("transformer_pred_"):]
                track_type = "transformer"
            elif label.startswith("gt_"):
                orig_label = label[len("gt_"):]
                track_type = "gt"
            else:
                continue

            all_tracks.append({
                "id": tid,
                "orig_id": orig_id,
                "label": label,
                "type": track_type,
                "orig_label": orig_label,
                "get": build_frame_index(kf),
            })

    # Group polylines to bboxes by orig_id. The XML writer stamps each
    # cyclist-agent polyline with the agent id (= the bicycle's track id),
    # so the merged trajectory attaches to the bicycle's bbox and the
    # rider's person-bbox simply renders without a predicted path.
    polys_by_orig = {}
    for t in all_tracks:
        if t["type"] == "bbox":
            continue
        polys_by_orig.setdefault(t["orig_id"], []).append(t)

    objects = []
    obj_idx = 0
    for tr in sorted((t for t in all_tracks if t["type"] == "bbox"),
                     key=lambda x: x["id"]):
        obj = {
            "obj_idx": obj_idx,
            "track_id": tr["orig_id"],
            "label": tr["orig_label"],
            "bbox_get": tr["get"],
            "lin_get": None,
            "kalman_get": None,
            "transformer_get": None,
            "gt_get": None,
        }
        for nxt in polys_by_orig.get(tr["orig_id"], []):
            if nxt["type"] == "lin":
                obj["lin_get"] = nxt["get"]
            elif nxt["type"] == "kalman":
                obj["kalman_get"] = nxt["get"]
            elif nxt["type"] == "transformer":
                obj["transformer_get"] = nxt["get"]
            elif nxt["type"] == "gt":
                obj["gt_get"] = nxt["get"]
        objects.append(obj)
        obj_idx += 1

    return objects


# ── drawing helpers ───────────────────────────────────────────────────────────

LABEL_BGR = {
    "car":        (0,   140, 255),
    "person":     (50,  205,  50),
    "bicycle":    (255, 255,   0),
    "truck":      (180,   0, 180),
    "cyclist":    (0,   255, 255),
    "scooter":    (255,   0, 128),
    "bike":       (0,   128, 255),
    "skateboard": (200, 150,  50),
}
DEFAULT_BGR = (200, 200, 200)

LIN_COLOR   = (60,  60, 255)   # red     (BGR)
KAL_COLOR   = (255,  80,  80)  # blue
TRANS_COLOR = (221,  68, 255)  # magenta — matches #ff44dd in XML
GT_COLOR    = (60, 220,  60)   # green


def label_color(lbl):
    return LABEL_BGR.get(lbl, DEFAULT_BGR)


def draw_dashed_line(img, p1, p2, color, thickness=2, dash=8, gap=4):
    """Draw a dashed line between two (x,y) points."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = max(1, int((dx**2 + dy**2) ** 0.5))
    drawing = True
    seg = 0
    for d in range(dist):
        limit = dash if drawing else gap
        if seg < limit:
            if drawing:
                frac = d / dist
                x = int(p1[0] + frac * dx)
                y = int(p1[1] + frac * dy)
                cv2.circle(img, (x, y), thickness, color, -1)
            seg += 1
        else:
            drawing = not drawing
            seg = 0


def draw_fading_path(frame, pts, color, thickness=1, min_alpha=0.0):
    """
    Draw a polyline that fades from full `color` at pts[0] to dim at pts[-1].
    pts        : list of (x, y)
    min_alpha  : floor for the per-segment alpha so the tail stays visible.
    """
    n = len(pts)
    if n < 2:
        return

    ipts = [(int(round(x)), int(round(y))) for x, y in pts]
    for i in range(n - 1):
        alpha = max(min_alpha, 1.0 - i / n)
        c = tuple(int(v * alpha) for v in color)
        cv2.line(frame, ipts[i], ipts[i + 1], c, thickness, cv2.LINE_AA)

    cv2.circle(frame, ipts[-1], 3, color, -1, cv2.LINE_AA)


def draw_bbox(frame, xtl, ytl, xbr, ybr, label, tid, color):
    x1, y1 = int(round(xtl)), int(round(ytl))
    x2, y2 = int(round(xbr)), int(round(ybr))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    text = f"{label} #{tid}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

    y_top = max(0, y1 - th - 4)
    cv2.rectangle(frame, (x1, y_top), (x1 + tw + 2, y1), color, -1)
    cv2.putText(
        frame, text, (x1 + 1, max(10, y1 - 3)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualize trajectory predictions on video."
    )
    ap.add_argument("--video", default="data/easy/124441_10-13min.mp4",
                    help="Input video file")
    ap.add_argument("--xml", default="predictions.xml",
                    help="Predictions XML from traj_predict_from_cvat.py")
    ap.add_argument("--output", default="124441_traj_annotated.mp4",
                    help="Output annotated video file")

    ap.add_argument("--no-gt", action="store_true",
                    help="Do not draw ground-truth future path")
    ap.add_argument("--no-linear", action="store_true",
                    help="Do not draw linear predicted future path")
    ap.add_argument("--no-kalman", action="store_true",
                    help="Do not draw Kalman predicted future path")
    ap.add_argument("--no-transformer", action="store_true",
                    help="Do not draw Transformer predicted future path")

    ap.add_argument("--alpha", type=float, default=0.6,
                    help="Trajectory overlay opacity (default: 0.6)")
    ap.add_argument("--ttc", default=None,
                    help="Path to time_to_collision.json for TTC overlay")
    ap.add_argument("--save-ttc-frames", default=None,
                    help="Directory to save TTC event frames (with full annotation)")
    ap.add_argument("--pet", default=None,
                    help="Path to pet_events.json for PET overlay")
    ap.add_argument("--pet-cell-size", type=float, default=30.0,
                    help="PET marker reference size in px (default: 30)")
    ap.add_argument("--pet-max-sec", type=float, default=5.0,
                    help="PET seconds at which the overlay fades to fully dim "
                         "(default: 5.0)")
    ap.add_argument("--save-pet-frames", default=None,
                    help="Directory to save PET trigger frames")
    ap.add_argument("--pet-history-frames", type=int, default=60,
                    help="Visual length of past bbox-centre trail prepended "
                         "to the snapshot, in frames (default: 60)")
    ap.add_argument("--pet-future-frames", type=int, default=0,
                    help="Cap the visualised future polyline at this many "
                         "points (0 = no cap, use full prediction). Lower "
                         "this to shorten the predicted line in the render.")
    ap.add_argument("--pet-tail-extrap-frames", type=int, default=25,
                    help="Linear extrapolation appended to the tail of the "
                         "predicted polyline, in frames (default: 25, set 0 "
                         "to disable)")
    args = ap.parse_args()

    # ── load TTC events ──────────────────────────────────────────────────────
    ttc_by_frame = defaultdict(list)
    ttc_trigger_frames = set()
    if args.ttc:
        with open(args.ttc) as f:
            ttc_events = json.load(f)
        for e in ttc_events:
            ttc_trigger_frames.add(e["frame_idx"])
            for fi in range(e["frame_idx"], e["frame_idx"] + e["ttc_frames"] + 1):
                ttc_by_frame[fi].append(e)
        print(f"Loaded {len(ttc_events)} TTC events from {args.ttc}")

    # ── load PET events ──────────────────────────────────────────────────────
    # Trajectory snapshots are built later, after the predictions XML has
    # been parsed (so we can sample each agent's transformer/gt polyline
    # ONCE at the frame they enter the conflict area).
    pet_events = []
    pet_trigger_frames = set()
    if args.pet:
        with open(args.pet) as f:
            pet_events = json.load(f)
        for e in pet_events:
            pet_trigger_frames.add(int(e["frame_event"]))
        print(f"Loaded {len(pet_events)} PET events from {args.pet}")
    pet_mode = args.pet is not None
    pet_renders_by_frame = defaultdict(list)  # filled below

    # ── prepare frame-save dirs ──────────────────────────────────────────────
    if args.save_ttc_frames:
        os.makedirs(args.save_ttc_frames, exist_ok=True)
    if args.save_pet_frames:
        os.makedirs(args.save_pet_frames, exist_ok=True)

    print(f"Parsing {args.xml} ...")
    objects = parse_predictions_xml(args.xml)

    # ── build PET trajectory snapshots ───────────────────────────────────────
    # For each PET event, capture each involved agent's transformer + gt
    # polyline ONCE at the frame they enter the conflict area, then replay
    # that exact polyline across every frame in the PET window. This keeps
    # the trajectory static (it does not move with the object).
    #
    # To make the line easier to read at this camera distance we extend it:
    #   1. Prepend the agent's PAST bbox centres (HISTORY_LEN frames before
    #      the anchor), so the line shows where they came from.
    #   2. Append a short LINEAR EXTRAPOLATION at the tail to keep the line
    #      readable when the prediction horizon is short.
    HISTORY_LEN = max(0, int(args.pet_history_frames))
    FUTURE_CAP = max(0, int(args.pet_future_frames))    # 0 = no cap
    EXTRAP_LEN = max(0, int(args.pet_tail_extrap_frames))
    if pet_mode:
        tid_to_obj = {o["track_id"]: o for o in objects}

        def _bbox_centre(obj, fr):
            if obj is None:
                return None
            box = obj["bbox_get"](fr)
            if box is None:
                return None
            return (
                (box[0] + box[2]) * 0.5,
                (box[1] + box[3]) * 0.5,
            )

        def _build_traj(obj, anchor, key):
            if obj is None or obj[key] is None:
                return None
            future = obj[key](anchor)
            if not future or len(future) < 2:
                return None

            future = list(future)
            if FUTURE_CAP > 0:
                future = future[:max(2, FUTURE_CAP)]

            past = []
            for f in range(anchor - HISTORY_LEN, anchor):
                c = _bbox_centre(obj, f)
                if c is not None:
                    past.append(c)

            pts = past + future

            if EXTRAP_LEN > 0 and len(future) >= 2:
                p_n = future[-1]
                p_m = future[-2]
                vx = p_n[0] - p_m[0]
                vy = p_n[1] - p_m[1]
                for i in range(1, EXTRAP_LEN + 1):
                    pts.append((p_n[0] + vx * i, p_n[1] + vy * i))

            return pts

        for e in pet_events:
            tid_a = int(e["track_a_id"])
            tid_b = int(e["track_b_id"])
            a_int = e.get("a_interval", [e["frame_event"], e["frame_event"]])
            b_int = e.get("b_interval", [e["frame_event"], e["frame_event"]])
            anchor_a = int(a_int[0])
            anchor_b = int(b_int[0])

            obj_a = tid_to_obj.get(tid_a)
            obj_b = tid_to_obj.get(tid_b)

            snapshot = {
                "event": e,
                "trans_a": _build_traj(obj_a, anchor_a, "transformer_get"),
                "gt_a":    _build_traj(obj_a, anchor_a, "gt_get"),
                "trans_b": _build_traj(obj_b, anchor_b, "transformer_get"),
                "gt_b":    _build_traj(obj_b, anchor_b, "gt_get"),
            }

            start = min(anchor_a, anchor_b)
            end = max(int(a_int[1]), int(b_int[1]))
            for fi in range(start, end + 1):
                pet_renders_by_frame[fi].append(snapshot)

    n_lin = sum(1 for o in objects if o["lin_get"] is not None)
    n_kal = sum(1 for o in objects if o["kalman_get"] is not None)
    n_trf = sum(1 for o in objects if o["transformer_get"] is not None)
    n_gt  = sum(1 for o in objects if o["gt_get"] is not None)

    print(
        f"  {len(objects)} objects | "
        f"{n_lin} linear tracks | "
        f"{n_kal} kalman tracks | "
        f"{n_trf} transformer tracks | "
        f"{n_gt} gt tracks"
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )

    print(f"Video: {W}x{H}  {fps:.1f} fps  {total} frames")

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()

        # ── trajectories ────────────────────────────────────────────────────
        # PET mode: draw frozen trajectory snapshots captured when each agent
        # entered the conflict cell. Same polyline every frame in the window;
        # it does not follow the moving object. Non-PET mode keeps old behavior.
        if pet_mode:
            for snap in pet_renders_by_frame.get(fidx, []):
                if not args.no_transformer:
                    if snap["trans_a"]:
                        draw_fading_path(overlay, snap["trans_a"], TRANS_COLOR,
                                         thickness=2, min_alpha=0.7)
                    if snap["trans_b"]:
                        draw_fading_path(overlay, snap["trans_b"], TRANS_COLOR,
                                         thickness=2, min_alpha=0.7)
                if not args.no_gt:
                    if snap["gt_a"]:
                        draw_fading_path(overlay, snap["gt_a"], GT_COLOR,
                                         thickness=2, min_alpha=0.7)
                    if snap["gt_b"]:
                        draw_fading_path(overlay, snap["gt_b"], GT_COLOR,
                                         thickness=2, min_alpha=0.7)
        else:
            for obj in objects:
                if not args.no_linear and obj["lin_get"] is not None:
                    pts = obj["lin_get"](fidx)
                    if pts and len(pts) >= 2:
                        draw_fading_path(overlay, pts, LIN_COLOR, thickness=1)

                if not args.no_kalman and obj["kalman_get"] is not None:
                    pts = obj["kalman_get"](fidx)
                    if pts and len(pts) >= 2:
                        draw_fading_path(overlay, pts, KAL_COLOR, thickness=1)

                if not args.no_transformer and obj["transformer_get"] is not None:
                    pts = obj["transformer_get"](fidx)
                    if pts and len(pts) >= 2:
                        draw_fading_path(overlay, pts, TRANS_COLOR, thickness=1)

                if not args.no_gt and obj["gt_get"] is not None:
                    pts = obj["gt_get"](fidx)
                    if pts and len(pts) >= 2:
                        draw_fading_path(overlay, pts, GT_COLOR, thickness=1)

        cv2.addWeighted(overlay, args.alpha, frame, 1.0 - args.alpha, 0, frame)

        # ── bounding boxes ──────────────────────────────────────────────────
        for obj in objects:
            box = obj["bbox_get"](fidx)
            if box is None:
                continue
            color = label_color(obj["label"])
            draw_bbox(
                frame,
                box[0], box[1], box[2], box[3],
                obj["label"], obj["obj_idx"], color
            )

        # ── legend ──────────────────────────────────────────────────────────
        legend_x = W - 200
        if not args.no_linear:
            cv2.putText(
                frame, "lin", (legend_x, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, LIN_COLOR, 1, cv2.LINE_AA
            )
            legend_x += 35

        if not args.no_kalman:
            cv2.putText(
                frame, "kal", (legend_x, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, KAL_COLOR, 1, cv2.LINE_AA
            )
            legend_x += 35

        if not args.no_transformer and n_trf > 0:
            cv2.putText(
                frame, "trf", (legend_x, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, TRANS_COLOR, 1, cv2.LINE_AA
            )
            legend_x += 35

        if not args.no_gt:
            cv2.putText(
                frame, "gt", (legend_x, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GT_COLOR, 1, cv2.LINE_AA
            )

        # ── TTC overlay ──────────────────────────────────────────────────────
        active_ttcs = ttc_by_frame.get(fidx, [])
        if active_ttcs:
            tid_to_obj = {o["track_id"]: o for o in objects}
            for tc in active_ttcs:
                obj_a = tid_to_obj.get(tc["track_a_id"])
                obj_b = tid_to_obj.get(tc["track_b_id"])
                if obj_a is None or obj_b is None:
                    continue
                box_a = obj_a["bbox_get"](fidx)
                box_b = obj_b["bbox_get"](fidx)
                if box_a is None or box_b is None:
                    continue

                ca = (int((box_a[0] + box_a[2]) / 2), int((box_a[1] + box_a[3]) / 2))
                cb = (int((box_b[0] + box_b[2]) / 2), int((box_b[1] + box_b[3]) / 2))

                draw_dashed_line(frame, ca, cb, (0, 200, 255), thickness=2)

                mx, my = (ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2 + 18
                label = f"TTC {tc['ttc_sec']:.2f}s  {tc['collision_dist']:.0f}px"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(frame, (mx - 2, my - th - 6), (mx + tw + 4, my + 4),
                              (0, 140, 200), -1)
                cv2.putText(frame, label, (mx + 1, my),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # ── PET overlay ──────────────────────────────────────────────────────
        # Small "X" at the PET conflict point. No severity-based size scaling;
        # colour alone encodes severity.
        active_snaps = pet_renders_by_frame.get(fidx, [])
        if active_snaps:
            half_cell = max(4, int(round(args.pet_cell_size / 4.0)))  # ~7 px
            for snap in sorted(active_snaps,
                               key=lambda s: -s["event"]["pet_sec"]):
                pe = snap["event"]
                pet_sec = float(pe["pet_sec"])
                sev = max(0.0, min(1.0, 1.0 - pet_sec / max(args.pet_max_sec, 1e-3)))

                # BGR: red (severe) -> yellow (mild)
                col = (0, int(round(255 * (1.0 - sev))), 255)

                cx_zone = int(round(pe["conflict_x"]))
                cy_zone = int(round(pe["conflict_y"]))

                size = half_cell
                thick = 2
                cv2.line(frame,
                         (cx_zone - size, cy_zone - size),
                         (cx_zone + size, cy_zone + size),
                         col, thick, cv2.LINE_AA)
                cv2.line(frame,
                         (cx_zone - size, cy_zone + size),
                         (cx_zone + size, cy_zone - size),
                         col, thick, cv2.LINE_AA)

                label = f"PET {pet_sec:.2f}s"
                if pe.get("is_critical"):
                    label += " CRIT"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
                )
                lx = max(0, cx_zone - tw // 2)
                ly = max(th + 4, cy_zone - size - 4)
                cv2.rectangle(frame,
                              (lx - 2, ly - th - 3), (lx + tw + 2, ly + 2),
                              col, -1)
                cv2.putText(frame, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 0, 0), 1, cv2.LINE_AA)

        # ── frame counter ───────────────────────────────────────────────────
        cv2.putText(
            frame, f"frame {fidx}/{total - 1}",
            (6, 14), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (220, 220, 220), 1, cv2.LINE_AA
        )

        writer.write(frame)

        # ── save event frames ───────────────────────────────────────────────
        if args.save_ttc_frames and fidx in ttc_trigger_frames:
            cv2.imwrite(os.path.join(args.save_ttc_frames, f"ttc_frame_{fidx:05d}.jpg"), frame)

        if args.save_pet_frames and fidx in pet_trigger_frames:
            cv2.imwrite(os.path.join(args.save_pet_frames, f"pet_frame_{fidx:05d}.jpg"), frame)

        if fidx % 500 == 0:
            print(f"  {fidx}/{total} frames done ...")

        fidx += 1

    cap.release()
    writer.release()
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
