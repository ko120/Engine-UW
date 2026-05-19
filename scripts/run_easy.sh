#!/usr/bin/env bash
# run_easy.sh — Easy split: linear + Kalman trajectory prediction + TTC overlay.
#
# Required:
#   VIDEO=/path/to/easy.mp4   (source clip; not bundled)
# Optional:
#   PYTHON=python3            (Python interpreter to use)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEO="${VIDEO:?Set VIDEO=/path/to/easy.mp4 (source video not bundled)}"
PYTHON="${PYTHON:-python3}"

ANNOT_XML="$REPO_ROOT/label/easy/annotations_easy.xml"
PRED_XML="$REPO_ROOT/label/easy/predictions.xml"
TTC_JSON="$REPO_ROOT/label/easy/time_to_collision.json"
OUT_VIDEO="$REPO_ROOT/output/easy/easy_annotated.mp4"
TTC_FRAMES_DIR="$REPO_ROOT/output/easy/ttc_frames"

mkdir -p "$(dirname "$OUT_VIDEO")" "$TTC_FRAMES_DIR"

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
