#!/usr/bin/env bash
# eval.sh — Evaluate the bundled easy-split tracker outputs against the
# easy-split ground-truth annotations.
#
# Optional:
#   PYTHON=python3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"

GT="$REPO_ROOT/label/easy/annotations_easy.xml"
BASE="$REPO_ROOT/result/easy"

echo "=== [all] ==="
"$PYTHON" "$REPO_ROOT/eval_metrics.py" \
    --pred "$BASE/all/124441_10-13min_sam3.json" \
    --gt   "$GT"

echo
echo "=== [separate] ==="
"$PYTHON" "$REPO_ROOT/eval_metrics.py" \
    --pred "$BASE/seperate/easy_sam3.json" \
    --gt   "$GT"

echo
echo "=== [yolo] ==="
"$PYTHON" "$REPO_ROOT/eval_metrics.py" \
    --pred "$BASE/yolo/124441_10-13min.json" \
    --gt   "$GT"
