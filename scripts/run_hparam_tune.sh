#!/usr/bin/env bash
# run_hparam_tune.sh — Random-search hyperparameter tuning for YOLO fine-tuning.
#
# Required:
#   VIDEO=/path/to/video.mp4
#   GT_XML=/path/to/annotations.xml   (CVAT 1.1 ground truth)
# Optional:
#   PYTHON=python3
#   BASE_MODEL=yolo26l.pt
#   PROJECT=$REPO_ROOT/runs/hptune
#   OUT_CSV=$REPO_ROOT/train_hparam_results.csv
#   CUDA_VISIBLE_DEVICES (set to pin a GPU)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEO="${VIDEO:?Set VIDEO=/path/to/video.mp4}"
GT_XML="${GT_XML:?Set GT_XML=/path/to/annotations.xml (CVAT 1.1 ground truth)}"
PYTHON="${PYTHON:-python3}"
BASE_MODEL="${BASE_MODEL:-yolo26l.pt}"
PROJECT="${PROJECT:-$REPO_ROOT/runs/hptune}"
OUT_CSV="${OUT_CSV:-$REPO_ROOT/train_hparam_results.csv}"

"$PYTHON" "$REPO_ROOT/yolo_train_hparam_tune.py" \
    --video                "$VIDEO" \
    --gt                   "$GT_XML" \
    --base-model           "$BASE_MODEL" \
    --project              "$PROJECT" \
    --out                  "$OUT_CSV" \
    --epochs               100 \
    --patience             10 \
    --test-ratio           0.20 \
    --max-frames           400 \
    --max-frames-per-class 100 \
    --seed                 42
