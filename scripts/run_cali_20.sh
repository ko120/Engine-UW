#!/usr/bin/env bash
# run_cali_20.sh — Cali segment 20: trajectory prediction + TTC overlay
# at 10 fps with a looser TTC speed gate.
#
# Required:
#   VIDEO=/path/to/segment_020.avi
#   ANNOT_XML=/path/to/cali_20_annotations.xml
# Optional:
#   PYTHON=python3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEO="${VIDEO:?Set VIDEO=/path/to/segment_020.avi}"
ANNOT_XML="${ANNOT_XML:?Set ANNOT_XML=/path/to/cali_20_annotations.xml}"
PYTHON="${PYTHON:-python3}"

OUT_DIR="$REPO_ROOT/output/cali_20"
PRED_XML="$OUT_DIR/predictions.xml"
TTC_JSON="$OUT_DIR/time_to_collision.json"
OUT_VIDEO="$OUT_DIR/cali_20_annotated.mp4"
TTC_FRAMES_DIR="$OUT_DIR/ttc_frames"

mkdir -p "$OUT_DIR" "$TTC_FRAMES_DIR"

# Classes rendered: car, person, bicycle, skateboard (all tracks in the XML
# are processed — neither traj_predict_from_cvat.py nor
# visualize_traj_predictions.py filters by class).
#
# Video is 10 fps. Skateboard track 242 spans ~80 frames, so lookback=10 is
# fine. ttc-min-speed=1.5 px/frame loosens the TTC speed gate a bit so the
# skateboard (~11 px/frame) and slower agents are not missed.

echo "=== Step 1: Trajectory prediction + TTC ==="
"$PYTHON" "$REPO_ROOT/traj_predict_from_cvat.py" \
    --input "$ANNOT_XML" \
    --output "$PRED_XML" \
    --horizon 30 \
    --lookback 10 \
    --fps 10 \
    --ttc-predictor linear \
    --ttc-margin 0 \
    --ttc-min-speed 1.5 \
    --ttc-stride 5

echo "=== Step 2: Annotated video ==="
"$PYTHON" "$REPO_ROOT/visualize_traj_predictions.py" \
    --video "$VIDEO" \
    --xml "$PRED_XML" \
    --ttc "$TTC_JSON" \
    --save-ttc-frames "$TTC_FRAMES_DIR" \
    --no-transformer \
    --output "$OUT_VIDEO"

echo "=== Done: $OUT_VIDEO ==="
