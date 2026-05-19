#!/usr/bin/env bash
# run_transformer_easy.sh — Easy split: Transformer trajectory prediction +
# safety analytics (PET, occupancy/conflict heatmaps) + annotated video.
#
# Required:
#   VIDEO=/path/to/easy.mp4
# Optional:
#   PYTHON=python3
#   MODEL=$REPO_ROOT/models/traj_transformer.pt  (trained checkpoint)
#
# Steps 1 and 2 (train + predict) are commented by default. Uncomment them
# if you have not yet produced a checkpoint and predictions XML, then re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEO="${VIDEO:?Set VIDEO=/path/to/easy.mp4 (source video not bundled)}"
PYTHON="${PYTHON:-python3}"

ANNOT_XML="$REPO_ROOT/label/easy/annotations_easy.xml"
MODEL_OUT="${MODEL:-$REPO_ROOT/models/traj_transformer.pt}"
PRED_XML="$REPO_ROOT/output/easy/predictions_with_transformer.xml"
OUT_VIDEO="$REPO_ROOT/output/easy/easy_annotated_with_transformer.mp4"

TTC_JSON="$REPO_ROOT/label/easy/time_to_collision.json"
SAFETY_DIR="$REPO_ROOT/output/easy/safety"

# Hyperparameters
LOOKBACK=10
HORIZON=30
EPOCHS=120
BATCH_SIZE=64
POLY_STRIDE=5
FEATURE_MODE=kinematic
VEL_WEIGHT=1.0
ACC_WEIGHT=0.5
CURVE_AUG_PROB=0.5
CURVE_AUG_OMEGA_MAX=0.025

mkdir -p "$(dirname "$MODEL_OUT")" "$(dirname "$PRED_XML")" "$SAFETY_DIR"

# # 1) Train the Transformer on label/easy
# echo "=== Step 1: Train Transformer on $ANNOT_XML ==="
# "$PYTHON" "$REPO_ROOT/transformer_trajectory.py" train \
#     --input         "$ANNOT_XML" \
#     --model-out     "$MODEL_OUT" \
#     --lookback      "$LOOKBACK" \
#     --horizon       "$HORIZON" \
#     --epochs        "$EPOCHS" \
#     --batch-size    "$BATCH_SIZE" \
#     --feature-mode        "$FEATURE_MODE" \
#     --vel-weight          "$VEL_WEIGHT" \
#     --acc-weight          "$ACC_WEIGHT" \
#     --curve-aug-prob      "$CURVE_AUG_PROB" \
#     --curve-aug-omega-max "$CURVE_AUG_OMEGA_MAX" \
#     --compare
#
# # 2) Predict trajectories and write the CVAT predictions XML
# echo "=== Step 2: Predict trajectories -> $PRED_XML ==="
# "$PYTHON" "$REPO_ROOT/transformer_trajectory.py" eval \
#     --input       "$ANNOT_XML" \
#     --model-in    "$MODEL_OUT" \
#     --output-xml  "$PRED_XML" \
#     --poly-stride "$POLY_STRIDE"

# 3) Safety analytics: PET + occupancy/conflict heatmaps + report
PET_METHOD=footprint
PET_MARKER_SIZE=30
PET_INTERACTION_THRESHOLD=5.0
PET_CRITICAL_THRESHOLD=1.5
PET_MIN_OVERLAP_RATIO=0.15
echo "=== Step 3: Safety analytics (PET + heatmaps) -> $SAFETY_DIR ==="
"$PYTHON" "$REPO_ROOT/safety_analytics.py" \
    --input          "$ANNOT_XML" \
    --ttc            "$TTC_JSON" \
    --video          "$VIDEO" \
    --out-dir        "$SAFETY_DIR" \
    --fps            30 \
    --pet-method     "$PET_METHOD" \
    --cell-size      "$PET_MARKER_SIZE" \
    --pet-interaction-threshold "$PET_INTERACTION_THRESHOLD" \
    --pet-critical-threshold    "$PET_CRITICAL_THRESHOLD" \
    --pet-min-overlap-ratio     "$PET_MIN_OVERLAP_RATIO" \
    --use-agents

# 4) Render annotated video with TTC + PET overlays
PET_JSON="$SAFETY_DIR/pet_events.json"
PET_FRAMES_DIR="$REPO_ROOT/output/easy/pet_frames"
echo "=== Step 4: Render annotated video -> $OUT_VIDEO ==="
"$PYTHON" "$REPO_ROOT/visualize_traj_predictions.py" \
    --video         "$VIDEO" \
    --xml           "$PRED_XML" \
    --output        "$OUT_VIDEO" \
    --ttc           "$TTC_JSON" \
    --pet           "$PET_JSON" \
    --pet-cell-size "$PET_MARKER_SIZE" \
    --pet-max-sec   "$PET_INTERACTION_THRESHOLD" \
    --save-pet-frames "$PET_FRAMES_DIR"

echo
echo "Done."
echo "  model:    $MODEL_OUT"
echo "  xml:      $PRED_XML"
echo "  video:    $OUT_VIDEO"
echo "  safety:   $SAFETY_DIR"
echo "  pet img:  $PET_FRAMES_DIR"
