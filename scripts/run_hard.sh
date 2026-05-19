#!/usr/bin/env bash
# run_hard.sh — Hard split: linear + Kalman trajectory prediction + TTC overlay.
#
# Required:
#   VIDEO=/path/to/hard.mp4
#   ANNOT_XML=/path/to/annotations_hard.xml  (CVAT 1.1 ground truth)
# Optional:
#   PYTHON=python3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEO="${VIDEO:?Set VIDEO=/path/to/hard.mp4 (source video not bundled)}"
ANNOT_XML="${ANNOT_XML:?Set ANNOT_XML=/path/to/annotations_hard.xml (CVAT 1.1 ground truth)}"
PYTHON="${PYTHON:-python3}"

PRED_XML="$REPO_ROOT/output/hard/predictions.xml"
TTC_JSON="$REPO_ROOT/output/hard/time_to_collision.json"
OUT_VIDEO="$REPO_ROOT/output/hard/hard_annotated.mp4"
TTC_FRAMES_DIR="$REPO_ROOT/output/hard/ttc_frames"

mkdir -p "$(dirname "$PRED_XML")" "$TTC_FRAMES_DIR"

echo "=== Step 1: Trajectory prediction + TTC ==="
"$PYTHON" "$REPO_ROOT/traj_predict_from_cvat.py" \
    --input "$ANNOT_XML" \
    --output "$PRED_XML" \
    --horizon 30 \
    --lookback 10 \
    --fps 30 \
    --ttc-predictor linear \
    --ttc-margin 0 \
    --ttc-stride 5

echo "=== Step 2: Annotated video ==="
"$PYTHON" "$REPO_ROOT/visualize_traj_predictions.py" \
    --video "$VIDEO" \
    --xml "$PRED_XML" \
    --ttc "$TTC_JSON" \
    --save-ttc-frames "$TTC_FRAMES_DIR" \
    --output "$OUT_VIDEO"
