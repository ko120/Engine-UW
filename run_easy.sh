#!/usr/bin/env bash
set -euo pipefail

VIDEO="/Users/brianko/Desktop/Engine final/Fall /Esay/124441_10-13min.mp4"
ANNOT_XML="/Users/brianko/Desktop/Engine final/Fall /engineVideo-main/label/easy/annotations_easy.xml"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"


# # 1 ── YOLO tracking → predictions.json
# echo "=== Step 1: YOLO tracking ==="
# python "$SCRIPT_DIR/yolo.py" \
#     --source "$VIDEO" \
#     --out-path "$OUT_DIR/predictions.json" \
#     --save-video

# 2 ── Trajectory prediction + near-miss detection + time-to-collision
echo "=== Step 2: Trajectory prediction + near-miss + TTC ==="
python "$SCRIPT_DIR/traj_predict_from_cvat.py" \
    --input "$ANNOT_XML" \
    --output  "/Users/brianko/Desktop/Engine final/Fall /engineVideo-main/label/easy/predictions.xml" \
    --horizon 30 \
    --lookback 10 \
    --fps 30 \
    --ttc-predictor linear \
    --ttc-margin 0 \
    --ttc-stride 5

# 3 ── Annotated video with near-miss overlay
echo "=== Step 3: Annotated video ==="
python "$SCRIPT_DIR/visualize_traj_predictions.py" \
  --video "$VIDEO" \
  --xml   "/Users/brianko/Desktop/Engine final/Fall /engineVideo-main/label/easy/predictions.xml" \
  --ttc "/Users/brianko/Desktop/Engine final/Fall /engineVideo-main/label/easy/time_to_collision.json" \
  --save-ttc-frames "/Users/brianko/Desktop/Engine final/Fall /engineVideo-main/output/easy/ttc_frames" \
  --output "/Users/brianko/Desktop/Engine final/Fall /engineVideo-main/output/easy/easy_annotated.mp4"


