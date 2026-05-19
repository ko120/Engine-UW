"""
safety_analytics.py — Milestone E: PET + heatmap analytics.

Reads a CVAT annotations XML (the ground-truth tracks) and the
`time_to_collision.json` produced by traj_predict_from_cvat.py, then
emits:

  • pet_events.json         — pair-wise post-encroachment-time events
  • heatmap_occupancy.png   — cumulative bbox occupancy across the clip
  • heatmap_conflict.png    — conflict density (TTC + PET) over a frame
  • ttc_histogram.png       — distribution of TTC times
  • pet_histogram.png       — distribution of PET times
  • safety_report.txt       — text summary (counts, percentiles, top-k)

PET (Post-Encroachment Time) is the time gap between when one road user
*leaves* a conflict area and another *enters* that same conflict area.
Lower PET ⇒ more severe near-miss.

By default this script approximates the conflict area from annotation
footprints: if two agents' bboxes cover the same image-space region at
different times, the frame gap is a PET candidate. Same-frame footprint
overlap is skipped because that is co-occupancy, not post-encroachment.
The older bbox-centre grid method remains available with
`--pet-method center-grid`.

Usage
-----
    python safety_analytics.py \\
        --input    label/easy/annotations_easy.xml \\
        --ttc      label/easy/time_to_collision.json \\
        --video    /path/to/easy.mp4 \\
        --out-dir  output/easy/safety \\
        --fps      30 \\
        --pet-method footprint \\
        --pet-interaction-threshold 5.0 \\
        --pet-critical-threshold 1.5
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2

from traj_predict_from_cvat import parse_cvat_xml, group_cyclist_agents


# ── PET ──────────────────────────────────────────────────────────────────────

def _clip_box(box, width, height):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(width), float(x1)))
    y1 = max(0.0, min(float(height), float(y1)))
    x2 = max(0.0, min(float(width), float(x2)))
    y2 = max(0.0, min(float(height), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_overlap_info(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, 0.0

    inter = (x1, y1, x2, y2)
    inter_area = _box_area(inter)
    smaller = max(1e-6, min(_box_area(a), _box_area(b)))
    return inter, inter_area / smaller, inter_area


def _pet_severity(pet_sec, critical_threshold_sec):
    return "critical" if pet_sec <= critical_threshold_sec else "interaction"


def track_frame_boxes(track, width, height):
    """
    Return sorted [(frame, bbox), ...] with bboxes clipped to image bounds.
    """
    rows = []
    for f, x1, y1, x2, y2 in track["frames"]:
        box = _clip_box((x1, y1, x2, y2), width, height)
        if box is not None:
            rows.append((int(f), box))
    rows.sort(key=lambda row: row[0])
    return rows


def _best_footprint_candidate(first_id, second_id,
                              first_rows, second_rows, first_by_frame,
                              threshold_frames, min_overlap_ratio):
    """
    Find the smallest positive frame gap where `second_id` enters image-space
    area previously occupied by `first_id`.
    """
    best = None
    second_start = 0

    for first_frame, first_box in first_rows:
        while (second_start < len(second_rows) and
               second_rows[second_start][0] <= first_frame):
            second_start += 1

        j = second_start
        while j < len(second_rows):
            second_frame, second_box = second_rows[j]
            gap = second_frame - first_frame
            if gap > threshold_frames:
                break

            inter, overlap_ratio, overlap_area = _box_overlap_info(
                first_box, second_box
            )
            if inter is None or overlap_ratio < min_overlap_ratio:
                j += 1
                continue

            # If the first agent is still in the same footprint when the
            # second arrives, this is co-occupancy rather than PET.
            first_at_second = first_by_frame.get(second_frame)
            if first_at_second is not None:
                same_inter, same_ratio, _ = _box_overlap_info(
                    first_at_second, second_box
                )
                if same_inter is not None and same_ratio >= min_overlap_ratio:
                    j += 1
                    continue

            candidate = {
                "gap": gap,
                "first_id": first_id,
                "second_id": second_id,
                "first_frame": first_frame,
                "second_frame": second_frame,
                "conflict_box": inter,
                "overlap_ratio": overlap_ratio,
                "overlap_area": overlap_area,
            }
            if (best is None or
                    candidate["gap"] < best["gap"] or
                    (candidate["gap"] == best["gap"] and
                     candidate["second_frame"] < best["second_frame"])):
                best = candidate
                if gap <= 1:
                    return best

            j += 1

    return best


def compute_pet_footprint(tracks, meta, fps=30.0,
                          pet_threshold_sec=5.0,
                          critical_threshold_sec=1.5,
                          min_overlap_ratio=0.15):
    """
    Compute PET from bbox footprint overlap.

    A candidate occurs when one agent's bbox overlaps the image-space
    footprint previously occupied by another agent, after the first agent has
    left that footprint. The smallest candidate per pair is kept.
    """
    width = meta["width"]
    height = meta["height"]
    threshold_frames = int(round(pet_threshold_sec * fps))

    track_boxes = {
        t["id"]: track_frame_boxes(t, width, height)
        for t in tracks
    }
    track_boxes = {tid: rows for tid, rows in track_boxes.items() if rows}
    track_by_frame = {
        tid: {f: box for f, box in rows}
        for tid, rows in track_boxes.items()
    }
    track_label = {t["id"]: t["label"] for t in tracks}
    track_members = {
        t["id"]: set(t.get("member_ids", [t["id"]]))
        for t in tracks
    }

    events = []
    tids = sorted(track_boxes.keys())
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a_id, b_id = tids[i], tids[j]
            shared_members = (
                track_members.get(a_id, {a_id}) &
                track_members.get(b_id, {b_id})
            )
            if shared_members:
                continue

            best_ab = _best_footprint_candidate(
                a_id, b_id,
                track_boxes[a_id], track_boxes[b_id], track_by_frame[a_id],
                threshold_frames, min_overlap_ratio,
            )
            best_ba = _best_footprint_candidate(
                b_id, a_id,
                track_boxes[b_id], track_boxes[a_id], track_by_frame[b_id],
                threshold_frames, min_overlap_ratio,
            )
            candidates = [c for c in (best_ab, best_ba) if c is not None]
            if not candidates:
                continue

            best = min(candidates, key=lambda c: (c["gap"], c["second_frame"]))
            if best["gap"] > threshold_frames:
                continue

            pet_sec = best["gap"] / fps
            x1, y1, x2, y2 = best["conflict_box"]
            cx_px = (x1 + x2) * 0.5
            cy_px = (y1 + y2) * 0.5
            t_event_sec = best["second_frame"] / fps

            if best["first_id"] == a_id:
                a_interval = [best["first_frame"], best["first_frame"]]
                b_interval = [best["second_frame"], best["second_frame"]]
            else:
                a_interval = [best["second_frame"], best["second_frame"]]
                b_interval = [best["first_frame"], best["first_frame"]]

            severity = _pet_severity(pet_sec, critical_threshold_sec)
            events.append({
                "track_a_id": int(a_id),
                "track_b_id": int(b_id),
                "label_a": track_label[a_id],
                "label_b": track_label[b_id],
                "pet_method": "bbox_footprint",
                "pet_frames": int(best["gap"]),
                "pet_sec": round(pet_sec, 3),
                "severity": severity,
                "is_critical": severity == "critical",
                "first_to_arrive": int(best["first_id"]),
                "second_to_arrive": int(best["second_id"]),
                "conflict_x": round(float(cx_px), 1),
                "conflict_y": round(float(cy_px), 1),
                "conflict_box": [
                    round(float(x1), 1),
                    round(float(y1), 1),
                    round(float(x2), 1),
                    round(float(y2), 1),
                ],
                "overlap_ratio": round(float(best["overlap_ratio"]), 3),
                "overlap_area": round(float(best["overlap_area"]), 1),
                "frame_event": int(best["second_frame"]),
                "time_sec": round(t_event_sec, 2),
                "time_str": f"{int(t_event_sec // 60)}:{t_event_sec % 60:05.2f}",
                "a_interval": [int(a_interval[0]), int(a_interval[1])],
                "b_interval": [int(b_interval[0]), int(b_interval[1])],
            })

    events.sort(key=lambda e: e["pet_sec"])
    return events


def track_cell_intervals(track, cell_size, width, height):
    """
    Return {(cx, cy): [(f_start, f_end), ...]} for one track.

    A cell is "visited" on frame f when the track's bbox **centre** sits
    in that cell — a single ground-truth point per frame, not the full
    bbox footprint. Consecutive frames in the same cell are merged into
    a single interval (with a 1-frame gap tolerance to bridge
    interpolation skips).
    """
    max_cx = max(0, int(width  - 1) // int(cell_size))
    max_cy = max(0, int(height - 1) // int(cell_size))

    visits = defaultdict(list)
    for f, x1, y1, x2, y2 in track["frames"]:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        ix = int(max(0.0, cx) // cell_size)
        iy = int(max(0.0, cy) // cell_size)
        ix = min(ix, max_cx)
        iy = min(iy, max_cy)
        visits[(ix, iy)].append(f)

    intervals = {}
    for cell, frames in visits.items():
        frames.sort()
        merged = [(frames[0], frames[0])]
        for f in frames[1:]:
            if f <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], f)
            else:
                merged.append((f, f))
        intervals[cell] = merged
    return intervals


def compute_pet_center_grid(tracks, meta, cell_size=30, fps=30.0,
                            pet_threshold_sec=5.0,
                            critical_threshold_sec=1.5):
    """
    Compute PET events over all pairs of tracks using bbox-centre grid cells.

    Returns
    -------
    events : list of dicts (one per pair, only if best PET ≤ threshold).
        Keys: track_a_id, track_b_id, label_a, label_b,
              pet_frames, pet_sec, first_to_arrive,
              conflict_x, conflict_y, frame_event,
              time_sec, time_str,
              a_interval, b_interval.
    """
    width = meta["width"]
    height = meta["height"]

    track_cells = {t["id"]: track_cell_intervals(t, cell_size, width, height)
                   for t in tracks}
    track_label = {t["id"]: t["label"] for t in tracks}
    track_members = {
        t["id"]: set(t.get("member_ids", [t["id"]]))
        for t in tracks
    }

    events = []
    pet_threshold_frames = pet_threshold_sec * fps

    tids = sorted(track_cells.keys())
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            a_id, b_id = tids[i], tids[j]
            shared_members = (
                track_members.get(a_id, {a_id}) &
                track_members.get(b_id, {b_id})
            )
            if shared_members:
                continue
            a_cells = track_cells[a_id]
            b_cells = track_cells[b_id]

            shared = set(a_cells.keys()) & set(b_cells.keys())
            if not shared:
                continue

            best = None  # (gap, cell, a_int, b_int, who_first, t_event)
            for cell in shared:
                for a_s, a_e in a_cells[cell]:
                    for b_s, b_e in b_cells[cell]:
                        if a_e < b_s:
                            gap = b_s - a_e
                            who_first = a_id
                            t_event = b_s
                        elif b_e < a_s:
                            gap = a_s - b_e
                            who_first = b_id
                            t_event = a_s
                        else:
                            # Co-occupancy in this cell — not a PET sample.
                            continue
                        if best is None or gap < best[0]:
                            best = (gap, cell, (a_s, a_e), (b_s, b_e),
                                    who_first, t_event)

            if best is None:
                continue
            gap, cell, a_int, b_int, who_first, t_event = best
            if gap > pet_threshold_frames:
                continue

            pet_sec = gap / fps
            cx_px = cell[0] * cell_size + cell_size / 2.0
            cy_px = cell[1] * cell_size + cell_size / 2.0
            t_event_sec = t_event / fps
            second_id = b_id if who_first == a_id else a_id
            severity = _pet_severity(pet_sec, critical_threshold_sec)

            events.append({
                "track_a_id": int(a_id),
                "track_b_id": int(b_id),
                "label_a": track_label[a_id],
                "label_b": track_label[b_id],
                "pet_method": "center_grid",
                "pet_frames": int(gap),
                "pet_sec": round(pet_sec, 3),
                "severity": severity,
                "is_critical": severity == "critical",
                "first_to_arrive": int(who_first),
                "second_to_arrive": int(second_id),
                "conflict_x": round(float(cx_px), 1),
                "conflict_y": round(float(cy_px), 1),
                "frame_event": int(t_event),
                "time_sec": round(t_event_sec, 2),
                "time_str": f"{int(t_event_sec // 60)}:{t_event_sec % 60:05.2f}",
                "a_interval": [int(a_int[0]), int(a_int[1])],
                "b_interval": [int(b_int[0]), int(b_int[1])],
            })

    events.sort(key=lambda e: e["pet_sec"])
    return events


def compute_pet(tracks, meta, cell_size=30, fps=30.0,
                pet_threshold_sec=5.0,
                critical_threshold_sec=1.5,
                method="footprint",
                min_overlap_ratio=0.15):
    """
    Compute PET events over all pairs of tracks.

    Returns
    -------
    events : list of dicts (one per pair, only if best PET ≤ threshold).
        Keys include: track_a_id, track_b_id, label_a, label_b,
        pet_frames, pet_sec, severity, first_to_arrive, conflict_x,
        conflict_y, frame_event, time_sec, time_str, a_interval, b_interval.
    """
    if method == "footprint":
        return compute_pet_footprint(
            tracks, meta,
            fps=fps,
            pet_threshold_sec=pet_threshold_sec,
            critical_threshold_sec=critical_threshold_sec,
            min_overlap_ratio=min_overlap_ratio,
        )
    if method == "center-grid":
        return compute_pet_center_grid(
            tracks, meta,
            cell_size=cell_size,
            fps=fps,
            pet_threshold_sec=pet_threshold_sec,
            critical_threshold_sec=critical_threshold_sec,
        )
    raise ValueError(f"unknown PET method: {method}")


# ── Heatmaps ─────────────────────────────────────────────────────────────────

def build_occupancy_heatmap(tracks, meta, blur_sigma=12.0):
    """Pixel-level cumulative bbox occupancy across the whole clip."""
    H, W = meta["height"], meta["width"]
    grid = np.zeros((H, W), dtype=np.float32)
    for t in tracks:
        for f, x1, y1, x2, y2 in t["frames"]:
            xa = int(max(0, x1))
            ya = int(max(0, y1))
            xb = int(min(W, x2))
            yb = int(min(H, y2))
            if xb > xa and yb > ya:
                grid[ya:yb, xa:xb] += 1.0
    if blur_sigma > 0:
        grid = gaussian_filter(grid, sigma=blur_sigma)
    return grid


def build_conflict_heatmap(meta, ttc_events, pet_events,
                           tracks=None,
                           ttc_floor=0.05,
                           pet_floor=0.05,
                           sigma=18.0):
    """
    Conflict density: a Gaussian blob at each TTC and PET event location,
    weighted by 1/ttc_sec and 1/pet_sec respectively (so closer-to-collision
    events shine brighter). TTC location is the midpoint of the two bboxes
    at the trigger frame; PET location is the stored conflict point.
    """
    H, W = meta["height"], meta["width"]
    grid = np.zeros((H, W), dtype=np.float32)

    track_box_at = {}
    if tracks is not None:
        for t in tracks:
            track_box_at[t["id"]] = {
                f: (x1, y1, x2, y2) for f, x1, y1, x2, y2 in t["frames"]
            }

    for e in ttc_events:
        cx = cy = None
        if tracks is not None:
            ba = track_box_at.get(e["track_a_id"], {}).get(e["frame_idx"])
            bb = track_box_at.get(e["track_b_id"], {}).get(e["frame_idx"])
            if ba and bb:
                cx = (ba[0] + ba[2] + bb[0] + bb[2]) / 4.0
                cy = (ba[1] + ba[3] + bb[1] + bb[3]) / 4.0
        if cx is None or cy is None:
            continue
        ttc = max(float(e.get("ttc_sec", 1.0)), ttc_floor)
        ix, iy = int(cx), int(cy)
        if 0 <= ix < W and 0 <= iy < H:
            grid[iy, ix] += 1.0 / ttc

    for e in pet_events:
        ix = int(e["conflict_x"])
        iy = int(e["conflict_y"])
        pet = max(float(e.get("pet_sec", 1.0)), pet_floor)
        if 0 <= ix < W and 0 <= iy < H:
            grid[iy, ix] += 1.0 / pet

    if sigma > 0:
        grid = gaussian_filter(grid, sigma=sigma)
    return grid


def save_heatmap_overlay(heat, base_img_bgr, out_path,
                         alpha=0.55, cmap="inferno", title=None):
    H, W = heat.shape
    heat_norm = heat / heat.max() if heat.max() > 0 else heat

    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=120)
    ax.imshow(cv2.cvtColor(base_img_bgr, cv2.COLOR_BGR2RGB))
    masked = np.ma.masked_where(heat_norm < 1e-3, heat_norm)
    im = ax.imshow(masked, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("density (norm)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    plt.tight_layout(pad=0.2)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_histogram(values, out_path, xlabel, title, bins=20, color="#cc3344"):
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=120)
    if values:
        ax.hist(values, bins=bins, color=color, edgecolor="black",
                linewidth=0.5, alpha=0.85)
    else:
        ax.text(0.5, 0.5, "no events", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def first_video_frame(video_path, fallback_size):
    H, W = fallback_size
    if video_path is None or not os.path.exists(video_path):
        return np.full((H, W, 3), 32, dtype=np.uint8)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.full((H, W, 3), 32, dtype=np.uint8)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return np.full((H, W, 3), 32, dtype=np.uint8)
    if frame.shape[0] != H or frame.shape[1] != W:
        frame = cv2.resize(frame, (W, H))
    return frame


# ── reporting ────────────────────────────────────────────────────────────────

def _percentiles(values):
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def write_report(out_path, ttc_events, pet_events, meta, params):
    ttc_secs = [float(e["ttc_sec"]) for e in ttc_events if "ttc_sec" in e]
    pet_secs = [float(e["pet_sec"]) for e in pet_events if "pet_sec" in e]
    critical_pet = [e for e in pet_events if e.get("is_critical")]

    ttc_stats = _percentiles(ttc_secs)
    pet_stats = _percentiles(pet_secs)

    by_pair = defaultdict(int)
    for e in ttc_events:
        by_pair[f"{e['label_a']} <-> {e['label_b']}"] += 1
    for e in pet_events:
        by_pair[f"{e['label_a']} <-> {e['label_b']}"] += 1

    lines = []
    lines.append("Safety / Conflict Analytics Report")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Scene size       : {meta['width']} x {meta['height']}")
    lines.append(f"Total frames     : {meta['total_frames']}")
    lines.append(f"FPS              : {params['fps']}")
    lines.append(f"PET method       : {params['pet_method']}")
    lines.append(f"PET interaction  : <= {params['pet_interaction_threshold']} s")
    lines.append(f"PET critical     : <= {params['pet_critical_threshold']} s")
    if params["pet_method"] == "center-grid":
        lines.append(f"PET cell size    : {params['cell_size']} px")
    else:
        lines.append(f"PET min overlap  : {params['pet_min_overlap_ratio']:.3f} "
                     "of smaller bbox")
    lines.append("")
    lines.append(f"TTC events       : {len(ttc_events)}")
    if ttc_stats["n"]:
        lines.append(f"  TTC sec  min/median/max : "
                     f"{ttc_stats['min']:.2f} / {ttc_stats['median']:.2f} / "
                     f"{ttc_stats['max']:.2f}")
        lines.append(f"  TTC sec  mean/p25/p75   : "
                     f"{ttc_stats['mean']:.2f} / {ttc_stats['p25']:.2f} / "
                     f"{ttc_stats['p75']:.2f}")
    lines.append("")
    lines.append(f"PET events       : {len(pet_events)} "
                 f"({len(critical_pet)} critical)")
    if pet_stats["n"]:
        lines.append(f"  PET sec  min/median/max : "
                     f"{pet_stats['min']:.2f} / {pet_stats['median']:.2f} / "
                     f"{pet_stats['max']:.2f}")
        lines.append(f"  PET sec  mean/p25/p75   : "
                     f"{pet_stats['mean']:.2f} / {pet_stats['p25']:.2f} / "
                     f"{pet_stats['p75']:.2f}")
    lines.append("")

    if by_pair:
        lines.append("Conflict counts by class pair:")
        for k, v in sorted(by_pair.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:<28}  {v:>4}")
        lines.append("")

    if ttc_events:
        lines.append(f"Top {min(10, len(ttc_events))} most severe TTC events:")
        lines.append(f"  {'frame':>6}  {'time':>8}  {'pair':<24}  {'ttc(s)':>7}")
        for e in sorted(ttc_events, key=lambda x: x["ttc_sec"])[:10]:
            pair = f"{e['label_a']} <-> {e['label_b']}"
            lines.append(f"  {e['frame_idx']:>6}  {e.get('time_str', ''):>8}  "
                         f"{pair:<24}  {e['ttc_sec']:>7.2f}")
        lines.append("")

    if pet_events:
        lines.append(f"Top {min(10, len(pet_events))} most severe PET events:")
        lines.append(f"  {'frame':>6}  {'time':>8}  {'pair':<24}  "
                     f"{'pet(s)':>7}  {'severity':>11}")
        for e in pet_events[:10]:
            pair = f"{e['label_a']} <-> {e['label_b']}"
            lines.append(f"  {e['frame_event']:>6}  {e['time_str']:>8}  "
                         f"{pair:<24}  {e['pet_sec']:>7.2f}  "
                         f"{e.get('severity', ''):>11}")
        lines.append("")

    text = "\n".join(lines)
    with open(out_path, "w") as f:
        f.write(text)
    return text


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="CVAT annotations XML (ground-truth tracks)")
    ap.add_argument("--ttc", default=None,
                    help="time_to_collision.json from traj_predict_from_cvat.py "
                         "(optional but recommended for the conflict heatmap)")
    ap.add_argument("--video", default=None,
                    help="Optional video file used as background for heatmap "
                         "overlays (first frame is used)")
    ap.add_argument("--out-dir", required=True,
                    help="Directory to write outputs into")

    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--pet-method", choices=["footprint", "center-grid"],
                    default="footprint",
                    help="PET detector. 'footprint' uses bbox overlap in "
                         "image space; 'center-grid' uses the older bbox "
                         "centre grid approximation (default: footprint).")
    ap.add_argument("--cell-size", type=float, default=30.0,
                    help="PET grid cell size in pixels for --pet-method "
                         "center-grid, and marker sizing in the renderer "
                         "(default: 30).")
    ap.add_argument("--pet-threshold", type=float, default=None,
                    help="Deprecated alias for --pet-interaction-threshold.")
    ap.add_argument("--pet-interaction-threshold", type=float, default=5.0,
                    help="Keep PET events <= this many seconds "
                         "(default: 5.0).")
    ap.add_argument("--pet-critical-threshold", type=float, default=1.5,
                    help="Mark PET events <= this many seconds as critical "
                         "(default: 1.5).")
    ap.add_argument("--pet-min-overlap-ratio", type=float, default=0.15,
                    help="For --pet-method footprint, require bbox-footprint "
                         "overlap area to be at least this fraction of the "
                         "smaller bbox area (default: 0.15).")
    ap.add_argument("--occupancy-sigma", type=float, default=12.0,
                    help="Gaussian blur sigma for occupancy heatmap (px)")
    ap.add_argument("--conflict-sigma", type=float, default=18.0,
                    help="Gaussian blur sigma for conflict heatmap (px)")
    ap.add_argument("--use-agents", action="store_true",
                    help="Group rider+bike into a single cyclist agent before "
                         "computing PET (matches TTC behaviour)")
    args = ap.parse_args()
    pet_interaction_threshold = (
        args.pet_threshold
        if args.pet_threshold is not None
        else args.pet_interaction_threshold
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load tracks ──────────────────────────────────────────────────────
    print(f"Parsing {args.input} ...")
    meta, tracks = parse_cvat_xml(args.input)
    print(f"  {len(tracks)} tracks  |  {meta['total_frames']} frames  |  "
          f"{meta['width']}x{meta['height']} px")

    if args.use_agents:
        tracks_for_pet = group_cyclist_agents(tracks)
        n_cyc = sum(1 for t in tracks_for_pet if t.get("label") == "cyclist")
        print(f"  grouped into {len(tracks_for_pet)} agents "
              f"({n_cyc} cyclists) for PET")
    else:
        tracks_for_pet = tracks

    # ── load TTC ─────────────────────────────────────────────────────────
    ttc_events = []
    if args.ttc and os.path.exists(args.ttc):
        with open(args.ttc) as f:
            ttc_events = json.load(f)
        print(f"  loaded {len(ttc_events)} TTC events from {args.ttc}")
    else:
        print("  (no TTC json supplied — conflict heatmap will use PET only)")

    # ── PET ──────────────────────────────────────────────────────────────
    if args.pet_method == "footprint":
        pet_desc = (f"method=footprint, interaction<={pet_interaction_threshold}s, "
                    f"critical<={args.pet_critical_threshold}s, "
                    f"min_overlap={args.pet_min_overlap_ratio}")
    else:
        pet_desc = (f"method=center-grid, cell={args.cell_size}px, "
                    f"interaction<={pet_interaction_threshold}s, "
                    f"critical<={args.pet_critical_threshold}s, "
                    "point=bbox-centre")
    print(f"\nComputing PET ({pet_desc}) ...")
    pet_events = compute_pet(
        tracks_for_pet, meta,
        cell_size=args.cell_size,
        fps=args.fps,
        pet_threshold_sec=pet_interaction_threshold,
        critical_threshold_sec=args.pet_critical_threshold,
        method=args.pet_method,
        min_overlap_ratio=args.pet_min_overlap_ratio,
    )
    n_critical_pet = sum(1 for e in pet_events if e.get("is_critical"))
    print(f"  found {len(pet_events)} PET events under interaction threshold "
          f"({n_critical_pet} critical)")
    pet_path = out_dir / "pet_events.json"
    with open(pet_path, "w") as f:
        json.dump(pet_events, f, indent=2)
    print(f"  -> {pet_path}")

    if pet_events:
        print(f"\n{'Frame':>6}  {'Time':>8}  {'A':>5}  {'B':>5}  "
              f"{'Pair':<24}  {'PET(s)':>7}  {'Severity':>11}")
        print("-" * 83)
        for e in pet_events[:20]:
            pair = f"{e['label_a']} <-> {e['label_b']}"
            print(f"  {e['frame_event']:>4}  {e['time_str']:>8}  "
                  f"{e['track_a_id']:>5}  {e['track_b_id']:>5}  "
                  f"{pair:<24}  {e['pet_sec']:>7.2f}  "
                  f"{e.get('severity', ''):>11}")
        if len(pet_events) > 20:
            print(f"  ... and {len(pet_events) - 20} more")

    # ── Heatmaps ─────────────────────────────────────────────────────────
    print("\nBuilding heatmaps ...")
    base = first_video_frame(args.video, (meta["height"], meta["width"]))

    occ = build_occupancy_heatmap(tracks, meta, blur_sigma=args.occupancy_sigma)
    occ_path = out_dir / "heatmap_occupancy.png"
    save_heatmap_overlay(occ, base, occ_path,
                         cmap="viridis", alpha=0.55,
                         title="Trajectory occupancy (cumulative bbox, blurred)")
    print(f"  -> {occ_path}")

    conflict = build_conflict_heatmap(
        meta, ttc_events, pet_events, tracks=tracks,
        sigma=args.conflict_sigma,
    )
    conflict_path = out_dir / "heatmap_conflict.png"
    save_heatmap_overlay(conflict, base, conflict_path,
                         cmap="inferno", alpha=0.65,
                         title="Conflict density (1/TTC + 1/PET, blurred)")
    print(f"  -> {conflict_path}")

    # ── Histograms ───────────────────────────────────────────────────────
    ttc_secs = [float(e["ttc_sec"]) for e in ttc_events if "ttc_sec" in e]
    pet_secs = [float(e["pet_sec"]) for e in pet_events]
    save_histogram(ttc_secs, out_dir / "ttc_histogram.png",
                   xlabel="TTC (s)", title="Time-to-Collision distribution",
                   color="#d65a31")
    save_histogram(pet_secs, out_dir / "pet_histogram.png",
                   xlabel="PET (s)", title="Post-Encroachment-Time distribution",
                   color="#3a86ff")
    print(f"  -> {out_dir / 'ttc_histogram.png'}")
    print(f"  -> {out_dir / 'pet_histogram.png'}")

    # ── Report ───────────────────────────────────────────────────────────
    report_path = out_dir / "safety_report.txt"
    text = write_report(report_path, ttc_events, pet_events, meta, {
        "fps": args.fps,
        "pet_method": args.pet_method,
        "cell_size": args.cell_size,
        "pet_interaction_threshold": pet_interaction_threshold,
        "pet_critical_threshold": args.pet_critical_threshold,
        "pet_min_overlap_ratio": args.pet_min_overlap_ratio,
    })
    print(f"\n{text}")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
