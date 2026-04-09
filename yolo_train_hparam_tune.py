"""yolo_train_hparam_tune.py

Hyperparameter tuning loop for YOLO fine-tuning.

Pipeline:
  1. Parse CVAT 1.1 XML ground truth
  2. Sample frames (with per-class cap + total frame cap)
  3. Extract frames + write YOLO .txt labels
  4. Train/test split -> train with model.train(), evaluate on test with model.val()
  5. Random-search over training HPs
  6. Save results to CSV sorted by mAP50-95

Usage:
    python yolo_train_hparam_tune.py \
        --video   2_09_084511_3min.mp4 \
        --gt      sam3_2_09_084511_3min_cvat.xml \
        --base-model yolo26l.pt \
        --project runs/hptune \
        --epochs   100 \
        --max-frames 600 \
        --max-frames-per-class 150

    # Quick smoke-test (300 frames, 5 epochs):
    python yolo_train_hparam_tune.py --max-frames 300 --epochs 5
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import cv2
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# CVAT label -> YOLO class id
# ---------------------------------------------------------------------------

LABEL_TO_CLS: Dict[str, int] = {
    "person":   0,
    "bicycle":  1,
    "car":      2,
    "truck":    7,
}

CLS_NAMES: Dict[int, str] = {v: k for k, v in LABEL_TO_CLS.items()}


# ---------------------------------------------------------------------------
# Step 1 – Parse CVAT XML
# ---------------------------------------------------------------------------

def parse_cvat_xml(xml_path: str) -> Tuple[Dict[int, List[dict]], int, int]:
    """
    Returns:
        annotations: {frame_idx: [{"cls": int, "box_xyxy": [x1,y1,x2,y2]}, ...]}
        width, height
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    orig = root.find(".//original_size")
    width  = int(orig.find("width").text)  if orig is not None else 1920
    height = int(orig.find("height").text) if orig is not None else 1080

    annotations: Dict[int, List[dict]] = {}

    for track_el in root.findall("track"):
        label  = track_el.get("label", "").lower()
        cls_id = LABEL_TO_CLS.get(label)
        if cls_id is None:
            continue

        for box_el in track_el.findall("box"):
            if box_el.get("outside", "0") == "1":
                continue
            fidx = int(box_el.get("frame"))
            x1 = float(box_el.get("xtl"))
            y1 = float(box_el.get("ytl"))
            x2 = float(box_el.get("xbr"))
            y2 = float(box_el.get("ybr"))
            annotations.setdefault(fidx, []).append(
                {"cls": cls_id, "box_xyxy": [x1, y1, x2, y2]}
            )

    return annotations, width, height


# ---------------------------------------------------------------------------
# Step 2 – Frame selection with per-class cap + total cap
# ---------------------------------------------------------------------------

def uniform_stride(frames: List[int], n: int) -> List[int]:
    """Pick up to n frames from a sorted list using uniform stride to cover the full range."""
    if len(frames) <= n:
        return list(frames)
    step = len(frames) / n          # floating-point step for even spacing
    return [frames[int(i * step)] for i in range(n)]


def select_frames(
    annotations: Dict[int, List[dict]],
    max_frames: Optional[int],
    max_frames_per_class: Optional[int],
) -> List[int]:
    """
    Select a temporally balanced subset of annotated frames.

    Algorithm:
      - Build class -> sorted list of frames mapping
      - For each class, pick every Nth frame (uniform stride) up to max_frames_per_class
        so that the full video timeline is covered for that class
      - Union all selected frames
      - If total > max_frames, apply uniform stride again on the union
    """
    # class_id -> sorted list of frames containing that class
    class_frames: Dict[int, List[int]] = defaultdict(list)
    for fidx in sorted(annotations.keys()):
        seen = set(d["cls"] for d in annotations[fidx])
        for cls_id in seen:
            class_frames[cls_id].append(fidx)

    print("\nFrame counts per class (before sampling):")
    for cls_id in sorted(class_frames):
        print(f"  {CLS_NAMES.get(cls_id, cls_id)}: {len(class_frames[cls_id])} frames")

    selected: set = set()

    if max_frames_per_class is not None:
        # Interleave frames from all classes so no single class dominates.
        # Build per-class strided iterators, then round-robin pick frames
        # until every class hits its cap.
        cls_iters = {
            cls_id: iter(uniform_stride(frames, max_frames_per_class))
            for cls_id, frames in class_frames.items()
        }
        cls_counts: Dict[int, int] = defaultdict(int)
        active = set(cls_iters.keys())

        while active:
            for cls_id in sorted(active):   # sorted for determinism
                try:
                    fidx = next(cls_iters[cls_id])
                except StopIteration:
                    active.discard(cls_id)
                    continue

                # Only add this frame if this class still needs more frames.
                # Count how many already-selected frames contain this class.
                if cls_counts[cls_id] >= max_frames_per_class:
                    active.discard(cls_id)
                    continue

                selected.add(fidx)
                # Credit every class present in this frame
                for det in annotations[fidx]:
                    cls_counts[det["cls"]] += 1

                # Discard classes that just hit their cap
                for cid in list(active):
                    if cls_counts[cid] >= max_frames_per_class:
                        active.discard(cid)
    else:
        selected = set(annotations.keys())

    selected = sorted(selected)

    if max_frames is not None and len(selected) > max_frames:
        selected = uniform_stride(selected, max_frames)

    print(f"\nSelected {len(selected)} frames total after caps.")

    # Print per-class counts after selection
    print("Frame counts per class (after sampling):")
    for cls_id in sorted(class_frames):
        count = sum(
            1 for f in selected
            if any(d["cls"] == cls_id for d in annotations[f])
        )
        print(f"  {CLS_NAMES.get(cls_id, cls_id)}: {count} frames")

    return selected


# ---------------------------------------------------------------------------
# Step 3 – Extract frames + write YOLO label files
# ---------------------------------------------------------------------------

def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float, W: int, H: int
) -> Tuple[float, float, float, float]:
    cx = (x1 + x2) / 2.0 / W
    cy = (y1 + y2) / 2.0 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return cx, cy, w, h


def build_yolo_dataset(
    video_path: str,
    annotations: Dict[int, List[dict]],
    selected_frames: List[int],
    width: int,
    height: int,
    dataset_dir: Path,
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[int], List[int]]:
    """
    Extract selected_frames from video, write images + YOLO label .txt files.
    Returns (train_frames, test_frames).

    Layout:
        dataset_dir/images/train/
        dataset_dir/images/test/
        dataset_dir/labels/train/
        dataset_dir/labels/test/
    """
    for split in ("train", "test"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Train/test split
    shuffled = list(selected_frames)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_ratio))
    test_set  = set(shuffled[:n_test])
    train_set = set(shuffled[n_test:])

    target_set = set(selected_frames)
    max_target = max(selected_frames)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    print(f"\nExtracting frames: train={len(train_set)}, test={len(test_set)} ...")

    written = 0
    current = 0

    while current <= max_target:
        ok, frame = cap.read()
        if not ok:
            break

        if current in target_set:
            split = "test" if current in test_set else "train"
            stem  = f"frame_{current:07d}"

            img_path = dataset_dir / "images" / split / f"{stem}.jpg"
            lbl_path = dataset_dir / "labels" / split / f"{stem}.txt"

            cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            lines = []
            for det in annotations[current]:
                x1, y1, x2, y2 = det["box_xyxy"]
                cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, width, height)
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                w  = max(0.0, min(1.0, w))
                h  = max(0.0, min(1.0, h))
                if w > 0 and h > 0:
                    lines.append(
                        f"{det['cls']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    )
            lbl_path.write_text("\n".join(lines))

            written += 1
            if written % 200 == 0:
                print(f"  {written}/{len(selected_frames)} written ...")

        current += 1

    cap.release()
    print(f"Extraction complete: {written} frames written.")
    return sorted(train_set), sorted(test_set)


def write_data_yaml(dataset_dir: Path, train_path: str, test_path: str) -> Path:
    nc    = max(CLS_NAMES.keys()) + 1
    names = [CLS_NAMES.get(i, f"class_{i}") for i in range(nc)]
    lines = [
        f"path: {dataset_dir.resolve()}",
        f"train: {train_path}",
        f"val:   {test_path}",   # YOLO uses 'val' key; we point it at test set
        f"nc: {nc}",
        f"names: {names}",
    ]
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text("\n".join(lines))
    return yaml_path


# ---------------------------------------------------------------------------
# Step 4 – Hyperparameter search space
# ---------------------------------------------------------------------------

SEARCH_SPACE: Dict[str, list] = {
    "lr0":          [0.0005, 0.005, 0.01],
    "optimizer":    ["AdamW"],
    "patience" : [10],
    "batch": [16,32],
    "imgsz": [640, 960]
}


def grid_params() -> List[dict]:
    """Return all combinations from SEARCH_SPACE (Cartesian product)."""
    keys = list(SEARCH_SPACE.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*SEARCH_SPACE.values())]


# ---------------------------------------------------------------------------
# Step 5 – One training trial
# ---------------------------------------------------------------------------

def run_trial(
    base_model: str,
    data_yaml: Path,
    project: str,
    run_name: str,
    params: dict,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
) -> dict:
    """
    Fine-tune YOLO and return val (test) metrics.
    YOLO evaluates on the 'val' split (which we pointed at the test set).
    """
    model = YOLO(base_model)

    results = model.train(
        data=str(data_yaml),
        project=project,
        name=run_name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        exist_ok=True,
        plots=False,
        save=True,
        verbose=False,
        # Tunable HPs
        lr0=params["lr0"],
        optimizer=params["optimizer"],
    )

    metrics: dict = {}
    try:
        rd = results.results_dict
        metrics["map50"]     = round(float(rd.get("metrics/mAP50(B)",    0.0)), 4)
        metrics["map50_95"]  = round(float(rd.get("metrics/mAP50-95(B)", 0.0)), 4)
        metrics["precision"] = round(float(rd.get("metrics/precision(B)", 0.0)), 4)
        metrics["recall"]    = round(float(rd.get("metrics/recall(B)",   0.0)), 4)
        metrics["best_epoch"]= int(getattr(results, "best_epoch", -1))
    except Exception as e:
        print(f"  Warning: could not parse metrics: {e}")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",      default="/home/brianko/Visual-Preference/test2/2_09_084511_3min.mp4")
    ap.add_argument("--gt",         default="/home/brianko/Visual-Preference/test2/sam3_2_09_084511_3min_cvat.xml")
    ap.add_argument("--base-model", default="yolo11n.pt",
                    help="Weights to fine-tune (e.g. yolo11n.pt or your best.pt)")
    ap.add_argument("--project",    default="runs/hptune")
    ap.add_argument("--out",        default="train_hparam_results.csv")
    ap.add_argument("--epochs",     type=int, default=100)
    ap.add_argument("--imgsz",      type=int, default=640)
    ap.add_argument("--batch",      type=int, default=16)
    ap.add_argument("--patience",   type=int, default=15)
    ap.add_argument("--test-ratio", type=float, default=0.20,
                    help="Fraction of selected frames reserved for test evaluation")
    # Frame selection
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Hard cap on total frames used (after per-class cap)")
    ap.add_argument("--max-frames-per-class", type=int, default=None,
                    help="Max frames per class to balance dataset")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--dataset-dir", default=None,
                    help="Where to store extracted frames. Default: <project>/dataset")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    project_dir = Path(args.project)

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        video_stem  = Path(args.video).stem
        mf          = f"mf{args.max_frames}"          if args.max_frames           else "mfall"
        mfpc        = f"pc{args.max_frames_per_class}" if args.max_frames_per_class else "pcall"
        dataset_dir = project_dir / f"dataset_{video_stem}_{mf}_{mfpc}"

    # ---- Parse GT ----
    print(f"Parsing CVAT XML: {args.gt}")
    annotations, width, height = parse_cvat_xml(args.gt)
    print(f"  {len(annotations)} annotated frames  ({width}x{height})")

    # ---- Frame selection ----
    data_yaml = dataset_dir / "data.yaml"

    if data_yaml.exists():
        print(f"Dataset already exists, skipping extraction: {dataset_dir}")
    else:
        selected = select_frames(
            annotations=annotations,
            max_frames=args.max_frames,
            max_frames_per_class=args.max_frames_per_class,
        )

        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)

        train_frames, test_frames = build_yolo_dataset(
            video_path=args.video,
            annotations=annotations,
            selected_frames=selected,
            width=width,
            height=height,
            dataset_dir=dataset_dir,
            test_ratio=args.test_ratio,
            rng=rng,
        )

        data_yaml = write_data_yaml(
            dataset_dir,
            train_path="images/train",
            test_path="images/test",
        )
        print(f"data.yaml -> {data_yaml}")
        print(f"Train frames: {len(train_frames)},  Test frames: {len(test_frames)}")

    # ---- Grid search trials ----
    trials = grid_params()
    print(f"\nStarting {len(trials)} grid-search trials ...")

    fieldnames = (
        list(SEARCH_SPACE.keys()) +
        ["map50", "map50_95", "precision", "recall", "best_epoch", "run_dir"]
    )
    rows: List[dict] = []

    for idx, params in enumerate(trials):
        run_name = f"trial_{idx:03d}"
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(trials)}]  {run_name}")
        for k, v in params.items():
            print(f"  {k}: {v}")

        try:
            metrics = run_trial(
                base_model=args.base_model,
                data_yaml=data_yaml,
                project=str(project_dir),
                run_name=run_name,
                params=params,
                epochs=args.epochs,
                imgsz=params["imgsz"],
                batch=params["batch"],
                patience=params["patience"],
            )
            run_dir = str(project_dir / run_name)
            row = {**params, **metrics, "run_dir": run_dir}
            rows.append(row)
            print(
                f"  -> mAP50={metrics.get('map50','?')}  "
                f"mAP50-95={metrics.get('map50_95','?')}  "
                f"P={metrics.get('precision','?')}  "
                f"R={metrics.get('recall','?')}  "
                f"best_epoch={metrics.get('best_epoch','?')}"
            )
        except Exception as e:
            print(f"  ERROR in trial {idx}: {e}")
            rows.append({**params, "run_dir": str(project_dir / run_name)})

    # ---- Save results ----
    rows.sort(key=lambda r: r.get("map50_95") or -1, reverse=True)

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {out_path}")
    print("\nTop 5 by mAP50-95:")
    for r in rows[:5]:
        print(
            f"  mAP50-95={r.get('map50_95','?')}  mAP50={r.get('map50','?')}  "
            + "  ".join(f"{k}={r.get(k,'?')}" for k in SEARCH_SPACE)
            + f"  run={Path(r.get('run_dir','')).name}"
        )

    # ---- Copy best weights to project root ----
    best = rows[0] if rows else None
    if best and best.get("run_dir") and best.get("map50_95") is not None:
        best_src = Path(best["run_dir"]) / "weights" / "best.pt"
        if best_src.exists():
            best_dst = project_dir / "best_overall.pt"
            shutil.copy2(best_src, best_dst)
            print(f"\nBest weights ({best['map50_95']} mAP50-95) saved to: {best_dst}")

            # Also save a JSON summary of the best trial's params + metrics
            best_summary = project_dir / "best_overall.json"
            best_summary.write_text(
                json.dumps({k: best.get(k) for k in fieldnames}, indent=2)
            )
            print(f"Best trial summary saved to: {best_summary}")


if __name__ == "__main__":
    main()
