#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="/Users/brianko/Desktop/Engine final/Fall /Esay/output"
mkdir -p "$OUT_DIR"

# python "$SCRIPT_DIR/annotate_video.py" \
#   --input  "/Users/brianko/Desktop/Engine final/Fall /Esay/124441_10-13min.mp4" \
#   --annot  "$OUT_DIR/predictions.json" \
#   --output "$OUT_DIR/124441_annotated.mp4" \
#   --json-out "$OUT_DIR/124441_annotated.json" \
#   --near-miss "$OUT_DIR/near_misses.json" \
#   --horizon 30 \
#   --lookback 5

python "$SCRIPT_DIR/visualize_traj_predictions.py" \
  --video "/Users/brianko/Desktop/Engine final/Fall /Esay/124441_10-13min.mp4" \
  --xml   "/Users/brianko/Desktop/Engine final/Fall /engineVideo-main/label/predictions.xml" \
  --near-miss "/Users/brianko/Desktop/Engine final/Fall /engineVideo-main/label/near_misses.json" \
  --output "$OUT_DIR/124441_traj_annotated.mp4"