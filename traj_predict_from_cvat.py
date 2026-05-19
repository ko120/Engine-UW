"""
Trajectory prediction evaluated against CVAT ground-truth annotations.

Reads a CVAT annotations.xml, then for every track and every eligible frame:
  1. Linear predictor   – constant-velocity model over the last `--lookback`
                          frames, forecasting the next `--horizon` frames.
  2. Kalman predictor   – constant-velocity Kalman filter over the same history,
                          forecasting the next `--horizon` frames.
  3. Ground-truth       – the actual future positions recorded in the XML.

Outputs
-------
  • A new CVAT XML with four track layers per object:
      bbox track               – original ground-truth bounding boxes
      polyline "lin_pred_*"    – linear-predicted future path
      polyline "kalman_pred_*" – Kalman-predicted future path
      polyline "gt_*"          – ground-truth future path

  • Console metrics:
      per-track ADE / FDE for linear and Kalman
      dataset-wide averages for linear and Kalman

Usage
-----
    python traj_predict_from_cvat.py \
        --input  data/easy/annotations.xml \
        --output predictions.xml \
        --horizon 30 --lookback 10 --poly-stride 5
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom
from datetime import datetime, timezone
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
from filterpy.kalman import KalmanFilter


# ── XML parsing ────────────────────────────────────────────────────────────────

def parse_cvat_xml(path):
    """
    Parse a CVAT interpolation-mode XML.

    Returns
    -------
    meta : dict   – width, height, total_frames, task_name, label_colors
    tracks : list of dicts, each with keys:
        id, label, frames (sorted list of (frame_idx, xtl, ytl, xbr, ybr))
    """
    tree = ET.parse(path)
    root = tree.getroot()

    orig = root.find(".//original_size")
    width = int(orig.find("width").text)
    height = int(orig.find("height").text)

    size_el = root.find(".//size")
    task_name_el = root.find(".//name")
    total_frames = int(size_el.text) if size_el is not None else None
    task_name = task_name_el.text if task_name_el is not None else "annotations"

    label_colors = {}
    for lbl in root.findall(".//label"):
        name = lbl.find("name").text
        color = lbl.find("color")
        label_colors[name] = color.text if color is not None else "#ffffff"

    tracks = []
    for track_el in root.findall("track"):
        tid = int(track_el.get("id"))
        label = track_el.get("label")

        frames = []
        for box in track_el.findall("box"):
            if box.get("outside", "0") == "1":
                continue
            frames.append((
                int(box.get("frame")),
                float(box.get("xtl")),
                float(box.get("ytl")),
                float(box.get("xbr")),
                float(box.get("ybr")),
            ))

        frames.sort(key=lambda x: x[0])
        if frames:
            tracks.append({"id": tid, "label": label, "frames": frames})

    if total_frames is None:
        total_frames = max(f[0] for t in tracks for f in t["frames"]) + 1

    meta = {
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "task_name": task_name,
        "label_colors": label_colors,
    }
    return meta, tracks


# ── geometry helpers ──────────────────────────────────────────────────────────

def box_center(xtl, ytl, xbr, ybr):
    return ((xtl + xbr) / 2.0, (ytl + ybr) / 2.0)


def box_dims(xtl, ytl, xbr, ybr):
    """Return (width, height) of a bounding box."""
    return (xbr - xtl, ybr - ytl)


def frame_to_center(track):
    """Returns {frame_idx: (cx, cy)} for quick look-up."""
    return {f[0]: box_center(f[1], f[2], f[3], f[4]) for f in track["frames"]}


def frame_to_dims(track):
    """Returns {frame_idx: (w, h)} for quick look-up."""
    return {f[0]: box_dims(f[1], f[2], f[3], f[4]) for f in track["frames"]}


# ── predictors ────────────────────────────────────────────────────────────────

def linear_predict(history, horizon):
    """
    Constant-velocity prediction using average velocity over history.

    Parameters
    ----------
    history : list of (x, y) – observed positions (oldest first)
    horizon : int – number of future steps to predict

    Returns
    -------
    list of (x, y) of length `horizon`
    """
    if len(history) >= 2:
        x0, y0 = history[0]
        x1, y1 = history[-1]
        dt = len(history) - 1
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt
    else:
        x1, y1 = history[-1]
        vx = vy = 0.0

    return [(x1 + k * vx, y1 + k * vy) for k in range(1, horizon + 1)]


def init_kalman_filter(history, process_var=1.0, meas_var=4.0):
    """
    Create a 2D constant-velocity Kalman filter.

    State:
        [x, y, vx, vy]^T

    Measurement:
        [x, y]^T
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State transition (dt = 1 frame)
    kf.F = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    # Measurement function
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    # Initialize state from history
    x_last, y_last = history[-1]

    if len(history) >= 2:
        x_prev, y_prev = history[-2]
        vx0 = x_last - x_prev
        vy0 = y_last - y_prev
    else:
        vx0 = 0.0
        vy0 = 0.0

    kf.x = np.array([x_last, y_last, vx0, vy0], dtype=float)

    # Initial covariance: more uncertain in velocity than position
    kf.P = np.array([
        [10, 0,   0,   0],
        [0,  10,  0,   0],
        [0,  0, 100,   0],
        [0,  0,   0, 100],
    ], dtype=float)

    # Measurement noise
    kf.R = np.array([
        [meas_var, 0],
        [0, meas_var],
    ], dtype=float)

    # Process noise
    q = process_var
    kf.Q = np.array([
        [0.25*q, 0,      0.5*q, 0],
        [0,      0.25*q, 0,     0.5*q],
        [0.5*q,  0,      1.0*q, 0],
        [0,      0.5*q,  0,     1.0*q],
    ], dtype=float)

    return kf


def kalman_predict(history, horizon, process_var=1.0, meas_var=4.0):
    """
    Kalman-filter prediction using the same history window.

    Parameters
    ----------
    history : list of (x, y) – observed positions (oldest first)
    horizon : int – number of future steps to predict

    Returns
    -------
    list of (x, y) of length `horizon`
    """
    if len(history) == 0:
        return []
    if len(history) == 1:
        x, y = history[-1]
        return [(x, y) for _ in range(horizon)]

    kf = init_kalman_filter(history, process_var=process_var, meas_var=meas_var)

    # Re-run filter through history in temporal order
    # We initialized at last point for a reasonable state guess, but for a clean
    # history fit we reset state using first observation.
    x0, y0 = history[0]
    if len(history) >= 2:
        x1, y1 = history[1]
        vx0 = x1 - x0
        vy0 = y1 - y0
    else:
        vx0 = vy0 = 0.0

    kf.x = np.array([x0, y0, vx0, vy0], dtype=float)

    for i, (x, y) in enumerate(history):
        if i > 0:
            kf.predict()
        kf.update(np.array([x, y], dtype=float))

    preds = []
    for _ in range(horizon):
        kf.predict()
        preds.append((float(kf.x[0]), float(kf.x[1])))

    return preds


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(tracks, horizon, lookback, kalman_process_var=1.0, kalman_meas_var=4.0):
    """
    For every frame t in every track that has `horizon` future GT frames:
      - build a linear prediction from the last `lookback` observed centres
      - build a Kalman prediction from the same observed centres
      - collect the ground-truth future centres
      - compute per-instance ADE and FDE

    Returns
    -------
    per_track : dict
    overall   : dict
    """
    per_track = {}

    all_lin_ade, all_lin_fde = [], []
    all_kal_ade, all_kal_fde = [], []

    for track in tracks:
        tid = track["id"]
        fc = frame_to_center(track)
        fidxs = sorted(fc.keys())

        lin_ades, lin_fdes = [], []
        kal_ades, kal_fdes = [], []

        for i, t in enumerate(fidxs):
            future_fidxs = [f for f in fidxs if f > t][:horizon]
            if len(future_fidxs) < horizon:
                continue

            past_fidxs = fidxs[max(0, i - lookback + 1): i + 1]
            history = [fc[f] for f in past_fidxs]

            lin_pred = linear_predict(history, horizon)
            kal_pred = kalman_predict(
                history,
                horizon,
                process_var=kalman_process_var,
                meas_var=kalman_meas_var,
            )
            gt = [fc[f] for f in future_fidxs]

            lin_dists = [np.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(lin_pred, gt)]
            kal_dists = [np.hypot(p[0] - g[0], p[1] - g[1]) for p, g in zip(kal_pred, gt)]

            lin_ades.append(float(np.mean(lin_dists)))
            lin_fdes.append(float(lin_dists[-1]))

            kal_ades.append(float(np.mean(kal_dists)))
            kal_fdes.append(float(kal_dists[-1]))

        if lin_ades:
            per_track[tid] = {
                "label": track["label"],
                "n": len(lin_ades),
                "lin_ade": float(np.mean(lin_ades)),
                "lin_fde": float(np.mean(lin_fdes)),
                "kal_ade": float(np.mean(kal_ades)),
                "kal_fde": float(np.mean(kal_fdes)),
            }

            all_lin_ade.extend(lin_ades)
            all_lin_fde.extend(lin_fdes)
            all_kal_ade.extend(kal_ades)
            all_kal_fde.extend(kal_fdes)

    overall = {
        "lin_ade": float(np.mean(all_lin_ade)) if all_lin_ade else float("nan"),
        "lin_fde": float(np.mean(all_lin_fde)) if all_lin_fde else float("nan"),
        "kal_ade": float(np.mean(all_kal_ade)) if all_kal_ade else float("nan"),
        "kal_fde": float(np.mean(all_kal_fde)) if all_kal_fde else float("nan"),
    }
    return per_track, overall


# ── cyclist-agent grouping ───────────────────────────────────────────────────

def _rider_score(person_box, bike_box):
    """
    Score in [0, 1] for how likely `person_box` represents the rider of
    `bike_box` in a given frame.  Takes the max of two forgiving signals
    so tall/narrow rider bboxes don't fail an IoU-only test:

      • overlap_frac = intersection / min(area_p, area_b)
      • proximity    = 1 - d(centers) / (bike_diag + person_diag),
                       measured against an upward-extended "rider zone"
                       (bike y-range extended up by one bike height,
                       since the rider sits above the frame)
    """
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    iw = max(0.0, min(px2, bx2) - max(px1, bx1))
    ih = max(0.0, min(py2, by2) - max(py1, by1))
    inter = iw * ih
    area_p = max(1e-6, (px2 - px1) * (py2 - py1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    overlap = inter / min(area_p, area_b) if inter > 0.0 else 0.0

    bike_w = max(1.0, bx2 - bx1)
    bike_h = max(1.0, by2 - by1)
    person_w = max(1.0, px2 - px1)
    person_h = max(1.0, py2 - py1)

    # Anchor against the rider zone: bike x-centre, y-centre shifted up
    # by bike_h/2 (so the rider's torso sits near the zone's centre).
    bcx = (bx1 + bx2) / 2.0
    bcy_zone = (by1 + by2) / 2.0 - bike_h / 2.0
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0

    dx = (pcx - bcx) / bike_w
    dy = (pcy - bcy_zone) / (bike_h + person_h)
    dist = (dx * dx + dy * dy) ** 0.5
    proximity = max(0.0, 1.0 - dist)

    return max(overlap, proximity)


def _box_overlap_frac(a, b):
    """Legacy helper — kept for callers that only need raw IoU-like overlap."""
    return _rider_score(a, b)


def group_cyclist_agents(tracks,
                         score_thresh=0.25,
                         frac_thresh=0.4,
                         min_common_frames=3,
                         verbose=True):
    """
    Fuse each rider (person) with their bicycle into a single 'cyclist' agent
    when their boxes score as co-located across shared frames.

    Each cyclist reuses the bicycle's track id as the agent id and carries a
    per-frame union bbox, so downstream TTC sees one agent per rider-bike
    pair instead of two tracks that produce duplicate near-miss events.
    Unbound tracks pass through unchanged.
    """
    per_track = {t["id"]: t for t in tracks}
    frames_by_tid = {
        tid: {f[0]: (f[1], f[2], f[3], f[4]) for f in t["frames"]}
        for tid, t in per_track.items()
    }

    persons = [tid for tid, t in per_track.items() if t["label"] == "person"]
    bicycles = [tid for tid, t in per_track.items() if t["label"] == "bicycle"]

    per_bike_best = {bid: None for bid in bicycles}

    candidates = []
    for pid in persons:
        pf = frames_by_tid[pid]
        for bid in bicycles:
            bf = frames_by_tid[bid]
            common = pf.keys() & bf.keys()
            if len(common) < min_common_frames:
                continue
            scores = [_rider_score(pf[f], bf[f]) for f in common]
            hits = sum(1 for s in scores if s >= score_thresh)
            frac = hits / len(common)
            mean_score = sum(scores) / len(scores)
            best = per_bike_best.get(bid)
            if best is None or frac > best[0]:
                per_bike_best[bid] = (frac, mean_score, len(common), pid)
            if frac >= frac_thresh:
                candidates.append((frac, len(common), pid, bid))

    # Greedy best-frac matching — each BIKE used once, a person may bind to
    # multiple bikes (covers CVAT track fragmentation where the same rider
    # spans several bike tracks).
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    used_p, used_b, bindings = set(), set(), []
    for frac, _, pid, bid in candidates:
        if bid in used_b:
            continue
        used_p.add(pid)
        used_b.add(bid)
        bindings.append((pid, bid))

    # For each bound person, collect the frames covered by its bound bikes
    # so the standalone person track only keeps the uncovered frames.
    person_bike_frames = {pid: set() for pid in used_p}
    for pid, bid in bindings:
        person_bike_frames[pid].update(frames_by_tid[bid].keys())

    agent_tracks = []
    for pid, bid in bindings:
        pf = frames_by_tid[pid]
        bf = frames_by_tid[bid]
        frames = []
        for f in sorted(bf.keys()):
            boxes = [bf[f]]
            if f in pf:
                boxes.append(pf[f])
            x1 = min(b[0] for b in boxes)
            y1 = min(b[1] for b in boxes)
            x2 = max(b[2] for b in boxes)
            y2 = max(b[3] for b in boxes)
            frames.append((f, x1, y1, x2, y2))
        agent_tracks.append({
            "id": bid,
            "label": "cyclist",
            "frames": frames,
            "member_ids": [pid, bid],
        })

    for tid, t in per_track.items():
        if tid in used_b:
            continue
        if tid in used_p:
            # Keep only the person's frames not already owned by a cyclist agent.
            kept = [row for row in t["frames"] if row[0] not in person_bike_frames[tid]]
            if not kept:
                continue
            agent_tracks.append({**t, "frames": kept, "member_ids": [tid]})
            continue
        agent_tracks.append({**t, "member_ids": [tid]})

    if verbose:
        print(f"  score_thresh={score_thresh}  frac_thresh={frac_thresh}  "
              f"min_common_frames={min_common_frames}")
        print(f"  per-bicycle best rider match (pass ? = frac >= {frac_thresh}):")
        for bid in sorted(bicycles):
            best = per_bike_best.get(bid)
            if best is None:
                print(f"    bike {bid:>3}: no person track shares "
                      f">= {min_common_frames} frames")
                continue
            frac, mean_score, n, pid = best
            passed = "PASS" if bid in used_b else "fail"
            print(f"    bike {bid:>3}: best person {pid:>3}  "
                  f"frac={frac:.2f}  mean_score={mean_score:.2f}  "
                  f"shared_frames={n}  -> {passed}")

    return agent_tracks


# ── time-to-collision ────────────────────────────────────────────────────────

def compute_time_to_collision(tracks, horizon, lookback,
                              collision_margin=0.0,
                              predictor="kalman",
                              kalman_process_var=1.0,
                              kalman_meas_var=4.0,
                              min_speed=2.0,
                              stride=5,
                              merge_window=30):
    """
    Compute Time-to-Collision (TTC) for all pairs of tracks using predicted
    trajectories.

    At every `stride`-th shared frame, predicts future positions of both
    objects using the chosen predictor, then walks the predicted paths to
    find the first future step where the two predicted bounding boxes
    overlap.  Bounding-box dimensions are taken from the last observed
    frame for each object and held constant over the prediction horizon.

    A collision at predicted step k means TTC = k frames from the
    evaluation frame.

    Parameters
    ----------
    tracks             : list of track dicts from parse_cvat_xml
    horizon            : int   – prediction horizon (frames)
    lookback           : int   – observation window (frames)
    collision_margin   : float – extra clearance in px added to each side of
                                 both boxes before testing overlap (default 0;
                                 use a positive value for a safety buffer)
    predictor          : str   – "linear" or "kalman"
    kalman_process_var : float
    kalman_meas_var    : float
    min_speed          : float – at least one object must move this fast
                                 (px/frame) to be considered
    stride             : int   – evaluate every N shared frames
    merge_window       : int   – merge same-pair events within N frames

    Returns
    -------
    list of dicts, each with keys:
        frame_idx, track_a_id, track_b_id, label_a, label_b,
        ttc_frames, collision_dist, predictor
    """
    # Pre-compute per-track lookups (centres + box dimensions)
    track_info = {}
    for track in tracks:
        tid = track["id"]
        fc = frame_to_center(track)
        fd = frame_to_dims(track)
        fidxs = sorted(fc.keys())
        track_info[tid] = {
            "label": track["label"],
            "fc": fc,
            "fd": fd,
            "fidxs": fidxs,
            "frame_set": set(fidxs),
            "frame_to_idx": {f: i for i, f in enumerate(fidxs)},
        }

    predict_fn = kalman_predict if predictor == "kalman" else linear_predict
    predict_kw = ({"process_var": kalman_process_var, "meas_var": kalman_meas_var}
                  if predictor == "kalman" else {})

    raw_events = []
    tids = list(track_info.keys())

    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            aid, bid = tids[i], tids[j]
            a, b = track_info[aid], track_info[bid]

            common = sorted(a["frame_set"] & b["frame_set"])
            if not common:
                continue

            for t in common[::stride]:
                # -- build histories --
                ia = a["frame_to_idx"][t]
                hist_a = [a["fc"][f]
                          for f in a["fidxs"][max(0, ia - lookback + 1): ia + 1]]
                ib = b["frame_to_idx"][t]
                hist_b = [b["fc"][f]
                          for f in b["fidxs"][max(0, ib - lookback + 1): ib + 1]]

                # -- speed filter --
                speed_a = np.hypot(
                    hist_a[-1][0] - hist_a[0][0],
                    hist_a[-1][1] - hist_a[0][1],
                ) / max(len(hist_a) - 1, 1)
                speed_b = np.hypot(
                    hist_b[-1][0] - hist_b[0][0],
                    hist_b[-1][1] - hist_b[0][1],
                ) / max(len(hist_b) - 1, 1)
                if max(speed_a, speed_b) < min_speed:
                    continue

                # -- predict future centres --
                pred_a = predict_fn(hist_a, horizon, **predict_kw)
                pred_b = predict_fn(hist_b, horizon, **predict_kw)
                if not pred_a or not pred_b:
                    continue

                # -- box dimensions from last observed frame --
                w_a, h_a = a["fd"][t]
                w_b, h_b = b["fd"][t]

                # Half-extents with optional safety margin
                hw_a = w_a / 2.0 + collision_margin
                hh_a = h_a / 2.0 + collision_margin
                hw_b = w_b / 2.0 + collision_margin
                hh_b = h_b / 2.0 + collision_margin

                # -- skip if boxes already overlap at current frame (e.g. rider + bicycle) --
                cx_a, cy_a = a["fc"][t]
                cx_b, cy_b = b["fc"][t]
                if abs(cx_a - cx_b) < (hw_a + hw_b) and abs(cy_a - cy_b) < (hh_a + hh_b):
                    continue

                # -- walk predicted trajectories to find first overlap --
                ttc_frame = None
                collision_dist = None
                for k, (pa, pb) in enumerate(zip(pred_a, pred_b)):
                    dx = abs(pa[0] - pb[0])
                    dy = abs(pa[1] - pb[1])
                    if dx < (hw_a + hw_b) and dy < (hh_a + hh_b):
                        ttc_frame = k + 1          # 1-indexed
                        collision_dist = float(np.hypot(dx, dy))
                        break

                if ttc_frame is None:
                    continue

                raw_events.append({
                    "frame_idx": t,
                    "track_a_id": aid,
                    "track_b_id": bid,
                    "label_a": a["label"],
                    "label_b": b["label"],
                    "ttc_frames": ttc_frame,
                    "collision_dist": round(collision_dist, 2),
                    "predictor": predictor,
                })

    # -- merge consecutive flags for the same pair --
    if merge_window > 0 and raw_events:
        raw_events.sort(key=lambda e: (e["track_a_id"], e["track_b_id"],
                                       e["frame_idx"]))
        merged = [raw_events[0]]
        for evt in raw_events[1:]:
            prev = merged[-1]
            same_pair = (evt["track_a_id"] == prev["track_a_id"] and
                         evt["track_b_id"] == prev["track_b_id"])
            if same_pair and evt["frame_idx"] - prev["frame_idx"] <= merge_window:
                # Keep the event with the smallest TTC
                if evt["ttc_frames"] < prev["ttc_frames"]:
                    merged[-1] = evt
            else:
                merged.append(evt)
        return merged

    return raw_events


# ── XML output ────────────────────────────────────────────────────────────────

def build_output_xml(meta, tracks, horizon, lookback, poly_stride,
                     kalman_process_var=1.0, kalman_meas_var=4.0,
                     box_threshold=0.5, agent_tracks=None,
                     transformer_predict_fn=None):
    """
    Produces a CVAT-compatible XML with up to five track layers per object:
      • bbox track (from raw tracks)
      • lin_pred_<label> / kalman_pred_<label> / gt_<label>
        (from agent_tracks when provided — so a rider+bike cyclist agent
        emits a single merged trajectory instead of two overlapping ones)
      • transformer_pred_<label> (only when `transformer_predict_fn` is given;
        takes a (history, horizon) -> list[(x, y)] callable so this file
        stays free of the deep-learning dependency)
    """
    now = datetime.now(timezone.utc).isoformat()

    if agent_tracks is None:
        agent_tracks = tracks

    orig_labels = sorted({t["label"] for t in tracks})
    pred_labels = sorted({t["label"] for t in agent_tracks})
    lc = meta["label_colors"]

    root = Element("annotations")
    SubElement(root, "version").text = "1.1"

    meta_el = SubElement(root, "meta")
    job_el = SubElement(meta_el, "job")
    SubElement(job_el, "id").text = "1"
    SubElement(job_el, "name").text = meta["task_name"] + "_traj"
    SubElement(job_el, "size").text = str(meta["total_frames"])
    SubElement(job_el, "mode").text = "interpolation"
    SubElement(job_el, "overlap").text = "0"
    SubElement(job_el, "bugtracker")
    SubElement(job_el, "created").text = now
    SubElement(job_el, "updated").text = now
    SubElement(job_el, "start_frame").text = "0"
    SubElement(job_el, "stop_frame").text = str(meta["total_frames"] - 1)
    SubElement(job_el, "frame_filter")

    labels_el = SubElement(job_el, "labels")

    for lname in orig_labels:
        lbl = SubElement(labels_el, "label")
        SubElement(lbl, "name").text = lname
        SubElement(lbl, "color").text = lc.get(lname, "#ffffff")
        SubElement(lbl, "type").text = "any"
        SubElement(lbl, "attributes")

    for lname in pred_labels:
        lbl = SubElement(labels_el, "label")
        SubElement(lbl, "name").text = f"lin_pred_{lname}"
        SubElement(lbl, "color").text = "#ff4444"   # red
        SubElement(lbl, "type").text = "any"
        SubElement(lbl, "attributes")

    for lname in pred_labels:
        lbl = SubElement(labels_el, "label")
        SubElement(lbl, "name").text = f"kalman_pred_{lname}"
        SubElement(lbl, "color").text = "#4444ff"   # blue
        SubElement(lbl, "type").text = "any"
        SubElement(lbl, "attributes")

    for lname in pred_labels:
        lbl = SubElement(labels_el, "label")
        SubElement(lbl, "name").text = f"gt_{lname}"
        SubElement(lbl, "color").text = "#44ff44"   # green
        SubElement(lbl, "type").text = "any"
        SubElement(lbl, "attributes")

    if transformer_predict_fn is not None:
        for lname in pred_labels:
            lbl = SubElement(labels_el, "label")
            SubElement(lbl, "name").text = f"transformer_pred_{lname}"
            SubElement(lbl, "color").text = "#ff44dd"   # magenta
            SubElement(lbl, "type").text = "any"
            SubElement(lbl, "attributes")

    segs = SubElement(job_el, "segments")
    seg = SubElement(segs, "segment")
    SubElement(seg, "id").text = "1"
    SubElement(seg, "start").text = "0"
    SubElement(seg, "stop").text = str(meta["total_frames"] - 1)
    SubElement(seg, "url")

    orig_el = SubElement(job_el, "original_size")
    SubElement(orig_el, "width").text = str(meta["width"])
    SubElement(orig_el, "height").text = str(meta["height"])
    SubElement(meta_el, "dumped").text = now

    cvat_id = 0

    # ── (1) Bbox layer — one track per original detection ─────────────────
    for track in tracks:
        label = track["label"]
        fidxs = sorted(f[0] for f in track["frames"])
        last_f = fidxs[-1]

        bbox_el = SubElement(root, "track",
                             id=str(cvat_id), orig_id=str(track["id"]),
                             label=label, source="file", z_order="0")
        cvat_id += 1

        last_box = None
        for (f, xtl, ytl, xbr, ybr) in track["frames"]:
            box = (xtl, ytl, xbr, ybr)
            changed = (last_box is None or
                       any(abs(box[i] - last_box[i]) > box_threshold for i in range(4)))
            if changed:
                SubElement(
                    bbox_el, "box",
                    frame=str(f), outside="0", occluded="0",
                    keyframe="1",
                    xtl=f"{xtl:.2f}", ytl=f"{ytl:.2f}",
                    xbr=f"{xbr:.2f}", ybr=f"{ybr:.2f}",
                    z_order="0"
                )
                last_box = box

        last_row = track["frames"][-1]
        if last_f + 1 < meta["total_frames"]:
            SubElement(
                bbox_el, "box",
                frame=str(last_f + 1), outside="1", occluded="0",
                keyframe="1",
                xtl=f"{last_row[1]:.2f}", ytl=f"{last_row[2]:.2f}",
                xbr=f"{last_row[3]:.2f}", ybr=f"{last_row[4]:.2f}",
                z_order="0"
            )

    # ── (2-4) Prediction + GT polylines — one set per agent ──────────────
    for agent in agent_tracks:
        label = agent["label"]
        fc = frame_to_center(agent)
        fidxs = sorted(fc.keys())
        last_f = fidxs[-1]
        agent_orig_id = str(agent["id"])

        def write_poly_track(elem_label, pts_fn, z_order):
            nonlocal cvat_id
            el = SubElement(
                root, "track",
                id=str(cvat_id), orig_id=agent_orig_id, label=elem_label,
                source="auto", z_order=str(z_order)
            )
            cvat_id += 1
            last_pts_str = ""

            for i, t in enumerate(fidxs):
                on_stride = (i % poly_stride == 0)
                is_last = (i == len(fidxs) - 1)
                if not (on_stride or is_last):
                    continue

                pts = pts_fn(i, t)
                if pts:
                    cx, cy = fc[t]
                    all_pts = [(cx, cy)] + pts
                    last_pts_str = ";".join(f"{x:.2f},{y:.2f}" for x, y in all_pts)
                    SubElement(
                        el, "polyline",
                        frame=str(t), outside="0", occluded="0",
                        keyframe="1", points=last_pts_str,
                        z_order=str(z_order)
                    )

            if last_f + 1 < meta["total_frames"] and last_pts_str:
                SubElement(
                    el, "polyline",
                    frame=str(last_f + 1), outside="1", occluded="0",
                    keyframe="1", points=last_pts_str,
                    z_order=str(z_order)
                )

        # (2) linear prediction
        def lin_pred_pts(i, t):
            past = fidxs[max(0, i - lookback + 1): i + 1]
            history = [fc[f] for f in past]
            return linear_predict(history, horizon)

        write_poly_track(f"lin_pred_{label}", lin_pred_pts, z_order=1)

        # (3) kalman prediction
        def kalman_pred_pts(i, t):
            past = fidxs[max(0, i - lookback + 1): i + 1]
            history = [fc[f] for f in past]
            return kalman_predict(
                history,
                horizon,
                process_var=kalman_process_var,
                meas_var=kalman_meas_var,
            )

        write_poly_track(f"kalman_pred_{label}", kalman_pred_pts, z_order=2)

        # (4) ground-truth future
        def gt_pts(i, t):
            future = [f for f in fidxs if f > t][:horizon]
            return [fc[f] for f in future]

        write_poly_track(f"gt_{label}", gt_pts, z_order=3)

        # (5) transformer prediction (optional)
        if transformer_predict_fn is not None:
            def transformer_pred_pts(i, t):
                past = fidxs[max(0, i - lookback + 1): i + 1]
                history = [fc[f] for f in past]
                return transformer_predict_fn(history, horizon)

            write_poly_track(f"transformer_pred_{label}", transformer_pred_pts, z_order=4)

    return root


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Trajectory prediction + GT comparison from CVAT XML."
    )
    ap.add_argument("--input", default="/Users/brianko/Desktop/Engine final/Fall /engineVideo-main/label/annotations_easy.xml",
                    help="Input CVAT annotations XML")
    ap.add_argument("--output", default=None,
                    help="Output CVAT XML (default: <input_dir>/predictions.xml)")
    ap.add_argument("--horizon", type=int, default=30,
                    help="Frames ahead to predict (default: 30)")
    ap.add_argument("--lookback", type=int, default=10,
                    help="Observation frames for velocity estimate/filtering (default: 10)")
    ap.add_argument("--poly-stride", type=int, default=5,
                    help="Write polyline keyframe every N frames (default: 5)")
    ap.add_argument("--box-threshold", type=float, default=0.5,
                    help="Min pixel change for a new bbox keyframe (default: 0.5)")
    ap.add_argument("--kalman-process-var", type=float, default=1.0,
                    help="Kalman process noise variance (default: 1.0)")
    ap.add_argument("--kalman-meas-var", type=float, default=4.0,
                    help="Kalman measurement noise variance (default: 4.0)")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Video FPS for timestamp conversion (default: 30)")
    ap.add_argument("--no-xml", action="store_true",
                    help="Skip XML output, only print metrics")
    ap.add_argument("--ttc-predictor", choices=["linear", "kalman"], default="kalman",
                    help="TTC: predictor type (default: kalman)")
    ap.add_argument("--ttc-margin", type=float, default=0.0,
                    help="TTC: extra clearance in px added to each box side (default: 0)")
    ap.add_argument("--ttc-min-speed", type=float, default=2.0,
                    help="TTC: min speed in px/frame for at least one object (default: 2.0)")
    ap.add_argument("--ttc-stride", type=int, default=5,
                    help="TTC: evaluate every N frames (default: 5)")
    ap.add_argument("--ttc-merge-window", type=int, default=30,
                    help="TTC: merge same-pair events within N frames (default: 30)")
    ap.add_argument("--cyclist-score-thresh", type=float, default=0.25,
                    help="Cyclist grouping: per-frame rider score threshold (default: 0.25)")
    ap.add_argument("--cyclist-frac-thresh", type=float, default=0.4,
                    help="Cyclist grouping: min fraction of shared frames above score (default: 0.4)")
    ap.add_argument("--cyclist-min-common", type=int, default=3,
                    help="Cyclist grouping: min shared frames between person and bike (default: 3)")
    args = ap.parse_args()

    if args.output is None:
        input_dir = os.path.dirname(os.path.abspath(args.input))
        args.output = os.path.join(input_dir, "predictions.xml")

    print(f"Parsing {args.input} ...")
    meta, tracks = parse_cvat_xml(args.input)
    print(f"  {len(tracks)} tracks  |  {meta['total_frames']} frames  |  {meta['width']}x{meta['height']} px")

    print(
        f"\nComputing metrics "
        f"(lookback={args.lookback}, horizon={args.horizon}, "
        f"kalman_process_var={args.kalman_process_var}, "
        f"kalman_meas_var={args.kalman_meas_var}) ..."
    )
    per_track, overall = compute_metrics(
        tracks,
        args.horizon,
        args.lookback,
        kalman_process_var=args.kalman_process_var,
        kalman_meas_var=args.kalman_meas_var,
    )

    print(
        f"\n{'Track':>6}  {'Label':<12}  {'Instances':>9}  "
        f"{'Lin ADE':>10}  {'Lin FDE':>10}  {'Kal ADE':>10}  {'Kal FDE':>10}"
    )
    print("-" * 85)
    for tid in sorted(per_track):
        m = per_track[tid]
        print(
            f"  {tid:>4}  {m['label']:<12}  {m['n']:>9}  "
            f"{m['lin_ade']:>10.3f}  {m['lin_fde']:>10.3f}  "
            f"{m['kal_ade']:>10.3f}  {m['kal_fde']:>10.3f}"
        )
    print("-" * 85)
    print(
        f"{'Overall':<22}  {'':>9}  "
        f"{overall['lin_ade']:>10.3f}  {overall['lin_fde']:>10.3f}  "
        f"{overall['kal_ade']:>10.3f}  {overall['kal_fde']:>10.3f}"
    )

    fps = args.fps

    # ── Time-to-Collision ───────────────────────────────────────────────
    print(
        f"\nComputing Time-to-Collision "
        f"(predictor={args.ttc_predictor}, margin={args.ttc_margin}px, "
        f"stride={args.ttc_stride}) ..."
    )
    agent_tracks = group_cyclist_agents(
        tracks,
        score_thresh=args.cyclist_score_thresh,
        frac_thresh=args.cyclist_frac_thresh,
        min_common_frames=args.cyclist_min_common,
    )
    n_cyclists = sum(1 for t in agent_tracks if t.get("label") == "cyclist")
    print(f"  grouped into {len(agent_tracks)} agents ({n_cyclists} cyclists)")
    agent_members = {t["id"]: t.get("member_ids", [t["id"]]) for t in agent_tracks}

    ttc_events = compute_time_to_collision(
        agent_tracks, args.horizon, args.lookback,
        collision_margin=args.ttc_margin,
        predictor=args.ttc_predictor,
        kalman_process_var=args.kalman_process_var,
        kalman_meas_var=args.kalman_meas_var,
        min_speed=args.ttc_min_speed,
        stride=args.ttc_stride,
        merge_window=args.ttc_merge_window,
    )

    for e in ttc_events:
        t_sec = e["frame_idx"] / fps
        e["time_sec"] = round(t_sec, 2)
        e["time_str"] = f"{int(t_sec // 60)}:{t_sec % 60:05.2f}"
        e["ttc_sec"] = round(e["ttc_frames"] / fps, 2)
        e["members_a"] = agent_members.get(e["track_a_id"], [e["track_a_id"]])
        e["members_b"] = agent_members.get(e["track_b_id"], [e["track_b_id"]])

    if ttc_events:
        print(
            f"\n{'Frame':>6}  {'Time':>8}  {'Trk A':>6}  {'Trk B':>6}  "
            f"{'Interaction':<25}  {'TTC(f)':>7}  {'TTC(s)':>7}  {'ColDist':>8}"
        )
        print("-" * 88)
        for e in ttc_events:
            pair = f"{e['label_a']} <-> {e['label_b']}"
            print(
                f"  {e['frame_idx']:>4}  {e['time_str']:>8}  "
                f"{e['track_a_id']:>6}  {e['track_b_id']:>6}  "
                f"{pair:<25}  {e['ttc_frames']:>7}  "
                f"{e['ttc_sec']:>7.2f}  {e['collision_dist']:>8.1f}"
            )
        print(f"\nTotal collision events: {len(ttc_events)}")
    else:
        print("  No predicted collisions detected within the horizon.")

    ttc_path = str(Path(args.output).with_name("time_to_collision.json"))
    with open(ttc_path, "w") as f:
        json.dump(ttc_events, f, indent=2)
    print(f"  TTC events saved to {ttc_path}")

    if not args.no_xml:
        print(f"\nBuilding output XML ...")
        xml_root = build_output_xml(
            meta, tracks,
            args.horizon, args.lookback, args.poly_stride,
            kalman_process_var=args.kalman_process_var,
            kalman_meas_var=args.kalman_meas_var,
            box_threshold=args.box_threshold,
            agent_tracks=agent_tracks,
        )

        raw = tostring(xml_root, encoding="unicode")
        dom = xml.dom.minidom.parseString(raw)
        pretty = ('<?xml version="1.0" encoding="utf-8"?>\n'
                  + dom.toprettyxml(indent="  ").split("\n", 1)[1])

        with open(args.output, "w", encoding="utf-8") as f:
            f.write(pretty)

        out_tree = ET.parse(args.output)
        n_bbox = sum(1 for e in out_tree.getroot().iter("box") if e.get("outside") == "0")
        n_poly = sum(1 for e in out_tree.getroot().iter("polyline") if e.get("outside") == "0")
        print(f"  Written -> {args.output}")
        print(f"  Bbox keyframes:     {n_bbox}")
        print(f"  Polyline keyframes: {n_poly}  (lin_pred + kalman_pred + gt combined)")


if __name__ == "__main__":
    main()