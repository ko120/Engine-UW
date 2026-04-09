#!/bin/bash
set -e

VIDEO="/home/brianko/Visual-Preference/test2/data/easy/124441_10-13min.mp4"
STEM=$(basename "${VIDEO%.*}")
SAM3_MODEL="/home/brianko/Visual-Preference/test2/sam3.pt"
YOLO_MODEL="/home/brianko/Visual-Preference/test2/yolo26x.pt"
BASE="/home/brianko/Visual-Preference/test2/result/easy"

# ── YOLO track ────────────────────────────────────────────────────────────────
echo "=== [yolo] track_video ==="
python yolo.py \
    --mode     track \
    --source   "$VIDEO" \
    --model    "$YOLO_MODEL" \
    --out-path "$BASE/yolo/$STEM.json"

# # ── SAM3: all classes (4 single-class chunks) ─────────────────────────────────
echo "=== [all] person bicycle car truck ==="
python sam3.py \
    --source   "$VIDEO" \
    --out-path "$BASE/all/$STEM" \
    --model    "$SAM3_MODEL" \
    --text     person bicycle car truck

# # ── SAM3: grouped chunks ──────────────────────────────────────────────────────
# echo "=== [separate] person+bicycle  |  car+truck ==="
# python sam3_cp.py \
#     --source   "$VIDEO" \
#     --out-path "$BASE/separate/$STEM" \
#     --model    "$SAM3_MODEL" \
#     --text     "person,bicycle" "car,truck"

# echo "Done."
# echo "  yolo     → $BASE/yolo/${STEM}.json"
# echo "  all      → $BASE/all/${STEM}_sam3.json"
# echo "  separate → $BASE/separate/${STEM}_sam3.json"
