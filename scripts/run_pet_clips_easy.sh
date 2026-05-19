#!/usr/bin/env bash
# run_pet_clips_easy.sh — Render one short MP4 per PET event on the easy
# clip. Each output shows ONLY the frozen ground-truth trajectory snapshot
# for the two involved agents plus the PET X marker and a top banner with
# the recomputed PET time. No bounding boxes, no predictions.
#
# Required:
#   VIDEO=/path/to/easy.mp4
#   PRED_XML=/path/to/predictions_with_transformer.xml
#       (output of run_transformer_easy.sh step 2)
# Optional:
#   PYTHON=python3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEO="${VIDEO:?Set VIDEO=/path/to/easy.mp4 (source video not bundled)}"
PRED_XML="${PRED_XML:?Set PRED_XML=/path/to/predictions_with_transformer.xml (produced by run_transformer_easy.sh)}"
PYTHON="${PYTHON:-python3}"

PET_JSON="$REPO_ROOT/output/easy/safety/pet_events.json"
OUT_DIR="$REPO_ROOT/output/easy/pet_clips"

LEAD_IN=30          # frames before the FIRST agent enters the conflict cell
LEAD_OUT=30         # frames after the SECOND agent leaves the conflict cell
PET_CELL_SIZE=30    # matches Step 3 of run_transformer_easy.sh
PET_MAX_SEC=5.0     # matches Step 3 (PET_INTERACTION_THRESHOLD)
MAX_PET_SEC=6.0     # skip events whose recomputed PET exceeds this
ALLOWED_PAIRS="car-person,car-cyclist,car-car"

mkdir -p "$OUT_DIR"

echo "=== Make PET clips -> $OUT_DIR ==="
"$PYTHON" "$REPO_ROOT/make_pet_clips.py" \
    --video         "$VIDEO" \
    --xml           "$PRED_XML" \
    --pet           "$PET_JSON" \
    --out-dir       "$OUT_DIR" \
    --lead-in       "$LEAD_IN" \
    --lead-out      "$LEAD_OUT" \
    --pet-cell-size "$PET_CELL_SIZE" \
    --pet-max-sec   "$PET_MAX_SEC" \
    --max-pet-sec   "$MAX_PET_SEC" \
    --allowed-pairs "$ALLOWED_PAIRS"

echo
echo "Done. Clips: $OUT_DIR"
