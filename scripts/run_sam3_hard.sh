#!/usr/bin/env bash
# run_sam3_hard.sh — YOLO track + SAM3 text-prompted segmentation on the hard clip.
#
# Required:
#   VIDEO=/path/to/hard.mp4
#   YOLO_MODEL=/path/to/yolo*.pt
#   SAM3_MODEL=/path/to/sam3.pt
# Optional:
#   PYTHON=python3
#   OUT_BASE=$REPO_ROOT/result/hard
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEO="${VIDEO:?Set VIDEO=/path/to/hard.mp4}"
YOLO_MODEL="${YOLO_MODEL:?Set YOLO_MODEL=/path/to/yolo*.pt}"
SAM3_MODEL="${SAM3_MODEL:?Set SAM3_MODEL=/path/to/sam3.pt}"
PYTHON="${PYTHON:-python3}"

STEM=$(basename "${VIDEO%.*}")
BASE="${OUT_BASE:-$REPO_ROOT/result/hard}"

mkdir -p "$BASE/yolo" "$BASE/all" "$BASE/separate"

echo "=== [yolo] track video ==="
"$PYTHON" "$REPO_ROOT/yolo.py" \
    --mode     track \
    --source   "$VIDEO" \
    --model    "$YOLO_MODEL" \
    --out-path "$BASE/yolo/$STEM.json"

echo "=== [all] person bicycle car truck ==="
"$PYTHON" "$REPO_ROOT/sam3.py" \
    --source   "$VIDEO" \
    --out-path "$BASE/all/$STEM" \
    --model    "$SAM3_MODEL" \
    --text     person bicycle car truck

echo "=== [separate] person+bicycle | car+truck ==="
"$PYTHON" "$REPO_ROOT/sam3.py" \
    --source   "$VIDEO" \
    --out-path "$BASE/separate/$STEM" \
    --model    "$SAM3_MODEL" \
    --text     "person,bicycle" "car,truck"

echo "Done."
echo "  yolo     -> $BASE/yolo/${STEM}.json"
echo "  all      -> $BASE/all/${STEM}_sam3.json"
echo "  separate -> $BASE/separate/${STEM}_sam3.json"
