"""
make_pet_clips.py — render one short MP4 per PET event.

Each clip shows the source video around a single PET event with ONLY:
  - the frozen ground-truth trajectory snapshot for the two involved agents
    (sampled once at their cell-entry frame; the same polyline is repeated
    every frame so the line stays fixed and does not move with the object)
  - the PET conflict X marker at the conflict cell
  - a small top-left label with the validated PET time + pet_frames

No bounding boxes are drawn. No other agents' trajectories are drawn. No
transformer / linear / Kalman prediction is drawn — ground truth only.

PET and TTC shown on screen are recomputed at clip-render time using
``pet/pet_ttc.py``'s closest-point method, from bbox-center tracks of the
two involved agents around the event. This replaces the bbox-overlap PET
that pet_events.json was generated with — the JSON is still used only to
discover the event pair, anchor frames, and conflict location.

The source video is read sequentially. Each event has its own VideoWriter
that opens lazily when its window begins and closes when its window ends,
which avoids the seek glitches that the mp4v decoder sometimes shows on
non-keyframe seeks.

Usage
-----
    python make_pet_clips.py \\
        --video    Esay/124441_10-13min.mp4 \\
        --xml      output/easy/predictions_with_transformer_v6.xml \\
        --pet      output/easy/safety/pet_events.json \\
        --out-dir  output/easy/pet_clips \\
        --pre-frames 60 --post-frames 30
"""

import argparse
import json
import os
import sys

import cv2

from visualize_traj_predictions import (
    parse_predictions_xml,
    draw_fading_path,
)

# Per-agent trajectory colors (BGR). A = green, B = orange — chosen for
# strong contrast against asphalt and each other.
COLOR_A = (60, 220, 60)
COLOR_B = (0, 165, 255)

# pet/ is a sibling folder of this file; add it so we can import pet_ttc
# without needing an __init__.py there.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "pet"))
from pet_ttc import compute_pet as _compute_pet
from pet_ttc import compute_ttc as _compute_ttc


def bbox_centre_track(obj, f_lo, f_hi):
    """Return ``[(frame, cx, cy), ...]`` for ``obj`` over [f_lo, f_hi]."""
    if obj is None:
        return []
    out = []
    bg = obj["bbox_get"]
    for f in range(f_lo, f_hi + 1):
        bx = bg(f)
        if bx is None:
            continue
        out.append((f, (bx[0] + bx[2]) * 0.5, (bx[1] + bx[3]) * 0.5))
    return out


def build_gt_snapshot(obj, anchor, history_len, extrap_len):
    """Return the frozen GT polyline for ``obj`` anchored at ``anchor``.

    Shape: [past bbox centres] + [gt polyline @ anchor] + [linear tail].
    Returns ``None`` if the agent has no GT polyline at the anchor frame.
    """
    if obj is None or obj["gt_get"] is None:
        return None
    future = obj["gt_get"](anchor)
    if not future or len(future) < 2:
        return None

    future = list(future)
    past = []
    bbox_get = obj["bbox_get"]
    for f in range(anchor - history_len, anchor):
        box = bbox_get(f)
        if box is None:
            continue
        past.append(((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5))

    pts = past + future
    if extrap_len > 0 and len(future) >= 2:
        p_n = future[-1]
        p_m = future[-2]
        vx = p_n[0] - p_m[0]
        vy = p_n[1] - p_m[1]
        for i in range(1, extrap_len + 1):
            pts.append((p_n[0] + vx * i, p_n[1] + vy * i))
    return pts


def render_overlay(frame, snap, pet_cell_size, pet_max_sec):
    """Apply frozen GT + PET marker + small top-left label to ``frame`` in-place."""
    gt_a = snap["gt_a"]
    gt_b = snap["gt_b"]
    ev = snap["event"]

    overlay = frame.copy()
    if gt_a:
        draw_fading_path(overlay, gt_a, COLOR_A, thickness=2, min_alpha=0.7)
    if gt_b:
        draw_fading_path(overlay, gt_b, COLOR_B, thickness=2, min_alpha=0.7)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    pet_sec_display = float(snap["pet_sec_display"])
    sev = max(0.0, min(1.0, 1.0 - pet_sec_display / max(pet_max_sec, 1e-3)))
    col_marker = (0, int(round(255 * (1.0 - sev))), 255)

    # Marker is sized to the *actual* conflict bbox from pet_events.json:
    # rectangle outline = conflict region, X inside = pinpoints the centre.
    cb = ev.get("conflict_box")
    if cb and len(cb) == 4:
        x1, y1, x2, y2 = (int(round(v)) for v in cb)
    else:
        cx0 = int(round(ev["conflict_x"]))
        cy0 = int(round(ev["conflict_y"]))
        half = max(3, int(round(pet_cell_size / 4.0)))
        x1, y1, x2, y2 = cx0 - half, cy0 - half, cx0 + half, cy0 + half

    # ensure a minimum visible size so single-pixel overlaps don't vanish
    if x2 - x1 < 6:
        cx = (x1 + x2) // 2
        x1, x2 = cx - 3, cx + 3
    if y2 - y1 < 6:
        cy = (y1 + y2) // 2
        y1, y2 = cy - 3, cy + 3

    cv2.rectangle(frame, (x1, y1), (x2, y2), col_marker, 1, cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x2, y2), col_marker, 2, cv2.LINE_AA)
    cv2.line(frame, (x1, y2), (x2, y1), col_marker, 2, cv2.LINE_AA)

    pet_pt = snap.get("pet_sec_pt")
    ttc_pt = snap.get("ttc_sec_pt")
    pet_part = (f"PET {pet_pt:.2f}s" if pet_pt is not None
                else f"PET {pet_sec_display:.2f}s")
    ttc_part = f"TTC {ttc_pt:.2f}s" if ttc_pt is not None else "TTC n/a"
    label = f"{pet_part}  {ttc_part}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thick = 1
    (tw, th), bl = cv2.getTextSize(label, font, scale, thick)
    pad = 4
    x0, y0 = 6, 6
    cv2.rectangle(frame, (x0 - 2, y0 - 2),
                  (x0 + tw + pad, y0 + th + bl + pad), (0, 0, 0), -1)
    cv2.putText(frame, label, (x0 + 1, y0 + th + 1),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--pet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lead-in", type=int, default=30,
                    help="frames included before the FIRST agent enters the "
                         "conflict cell (default 30)")
    ap.add_argument("--lead-out", type=int, default=30,
                    help="frames included after the SECOND agent leaves the "
                         "conflict cell (default 30)")
    ap.add_argument("--pet-cell-size", type=float, default=30.0,
                    help="PET marker reference size in px (default 30)")
    ap.add_argument("--pet-max-sec", type=float, default=5.0,
                    help="PET sec at which marker fades to yellow (default 5)")
    ap.add_argument("--history-frames", type=int, default=60,
                    help="past bbox-centre frames prepended to GT polyline "
                         "(default 60)")
    ap.add_argument("--extrap-frames", type=int, default=0,
                    help="constant-velocity linear extrapolation appended to "
                         "the GT polyline tail. NOT ground truth — disabled "
                         "by default (set >0 to extend the visible line)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only render the first N events (0 = all)")
    ap.add_argument("--metric-window-pad", type=int, default=90,
                    help="frames padded before/after the agents' cell-entry "
                         "anchors when measuring PET/TTC (default 90)")
    ap.add_argument("--meters-per-pixel", type=float, default=0.03,
                    help="pixel->meter scale for TTC's min-distance "
                         "(default 0.03, matching pet_ttc.py)")
    ap.add_argument("--max-pet-sec", type=float, default=6.0,
                    help="only render events whose displayed PET is <= this "
                         "many seconds (default 6.0)")
    ap.add_argument("--allowed-pairs", type=str,
                    default="car-person,car-cyclist,car-car",
                    help="comma-separated label pairs to keep (order-"
                         "insensitive, e.g. 'car-person'). Default keeps "
                         "car-person, car-cyclist, car-car. Pass an empty "
                         "string to disable filtering.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.pet) as f:
        events = json.load(f)
    if args.limit > 0:
        events = events[:args.limit]

    allowed_pairs = None
    if args.allowed_pairs.strip():
        allowed_pairs = set()
        for tok in args.allowed_pairs.split(","):
            parts = tok.strip().split("-")
            if len(parts) != 2:
                continue
            allowed_pairs.add(tuple(sorted(p.strip() for p in parts)))
        print(f"Pair filter active: {sorted(allowed_pairs)}")

    print(f"Rendering {len(events)} PET events")

    print(f"Parsing {args.xml} ...")
    objects = parse_predictions_xml(args.xml)
    tid_to_obj = {o["track_id"]: o for o in objects}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}x{H}  {fps:.1f} fps  {total} frames")

    snaps = []
    for i, ev in enumerate(events):
        pair_key = tuple(sorted([ev["label_a"], ev["label_b"]]))
        if allowed_pairs is not None and pair_key not in allowed_pairs:
            print(f"  [skip] event {i} f={int(ev['frame_event'])} "
                  f"pair={pair_key[0]}-{pair_key[1]} not in allowed set")
            continue

        tid_a = int(ev["track_a_id"])
        tid_b = int(ev["track_b_id"])
        obj_a = tid_to_obj.get(tid_a)
        obj_b = tid_to_obj.get(tid_b)
        anchor_a = int(ev.get("a_interval", [ev["frame_event"]])[0])
        anchor_b = int(ev.get("b_interval", [ev["frame_event"]])[0])
        gt_a = build_gt_snapshot(obj_a, anchor_a,
                                 args.history_frames, args.extrap_frames)
        gt_b = build_gt_snapshot(obj_b, anchor_b,
                                 args.history_frames, args.extrap_frames)

        event_frame = int(ev["frame_event"])
        a_lo, a_hi = int(ev["a_interval"][0]), int(ev["a_interval"][1])
        b_lo, b_hi = int(ev["b_interval"][0]), int(ev["b_interval"][1])
        first_entry = min(a_lo, b_lo)
        last_exit = max(a_hi, b_hi)
        f_start = max(0, first_entry - args.lead_in)
        f_end = min(total - 1, last_exit + args.lead_out)

        win_start = max(0, min(anchor_a, anchor_b) - args.metric_window_pad)
        win_end = min(total - 1,
                      max(anchor_a, anchor_b) + args.metric_window_pad)
        track_a = bbox_centre_track(obj_a, win_start, win_end)
        track_b = bbox_centre_track(obj_b, win_start, win_end)
        crossing = (float(ev["conflict_x"]), float(ev["conflict_y"]))
        steps = win_end - win_start

        pet_sec_pt = _compute_pet(track_a, track_b, crossing,
                                  win_start, steps, fps)
        ttc_sec_pt, min_dist_m = _compute_ttc(
            track_a, track_b, win_start, steps, fps,
            meters_per_pixel=args.meters_per_pixel,
        )

        fallback_pet_sec = int(ev.get("pet_frames", 0)) / fps if fps > 0 \
            else float(ev["pet_sec"])
        pet_sec_display = pet_sec_pt if pet_sec_pt is not None \
            else fallback_pet_sec

        if pet_sec_display > args.max_pet_sec:
            print(f"  [skip] event {i} f={event_frame} PET={pet_sec_display:.2f}s "
                  f"> max {args.max_pet_sec:.2f}s")
            continue

        clip_name = (
            f"pet_{i:02d}_f{event_frame:05d}"
            f"_{ev['label_a']}{tid_a}_{ev['label_b']}{tid_b}"
            f"_{pet_sec_display:.2f}s.mp4"
        )
        out_path = os.path.join(args.out_dir, clip_name)
        snaps.append({
            "event": ev, "obj_a": obj_a, "obj_b": obj_b,
            "gt_a": gt_a, "gt_b": gt_b,
            "f_start": f_start, "f_end": f_end,
            "out_path": out_path, "writer": None,
            "name": clip_name, "idx": i,
            "pet_sec_display": pet_sec_display,
            "pet_sec_pt": pet_sec_pt,
            "ttc_sec_pt": ttc_sec_pt,
            "min_dist_m": min_dist_m,
        })
        if gt_a is None and gt_b is None:
            print(f"  [warn] event {i} f={event_frame} — no GT polyline for "
                  f"either agent; clip shows only X marker + bboxes")

    print(f"\nReading source video and writing {len(snaps)} clips ...")
    fidx = 0
    while True:
        if all(s["f_end"] < fidx for s in snaps):
            break
        ret, frame = cap.read()
        if not ret:
            break

        for s in snaps:
            if s["f_start"] <= fidx <= s["f_end"]:
                if s["writer"] is None:
                    s["writer"] = cv2.VideoWriter(
                        s["out_path"],
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (W, H),
                    )
                fcopy = frame.copy()
                render_overlay(fcopy, s,
                               args.pet_cell_size, args.pet_max_sec)
                s["writer"].write(fcopy)
                if fidx == s["f_end"]:
                    s["writer"].release()
                    s["writer"] = None
                    print(f"  [{s['idx']+1}/{len(snaps)}] {s['name']}")

        if fidx % 500 == 0:
            print(f"    src frame {fidx}/{total}")

        fidx += 1

    for s in snaps:
        if s["writer"] is not None:
            s["writer"].release()

    cap.release()
    print(f"\nDone. Clips written to {args.out_dir}")


if __name__ == "__main__":
    main()
