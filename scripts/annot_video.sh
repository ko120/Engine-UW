#!/usr/bin/env bash
# annot_video.sh — Render trajectory predictions onto the easy clip using
# the bundled label/easy/predictions.xml.
#
# Required:
#   VIDEO=/path/to/easy.mp4
# Optional:
#   PYTHON=python3
#   OUT=$REPO_ROOT/output/easy/easy_traj_annotated.mp4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VIDEO="${VIDEO:?Set VIDEO=/path/to/easy.mp4 (source video not bundled)}"
PYTHON="${PYTHON:-python3}"

XML="$REPO_ROOT/label/easy/predictions.xml"
NEAR_MISS="$REPO_ROOT/label/easy/near_misses.json"
OUT="${OUT:-$REPO_ROOT/output/easy/easy_traj_annotated.mp4}"

mkdir -p "$(dirname "$OUT")"

"$PYTHON" "$REPO_ROOT/visualize_traj_predictions.py" \
    --video "$VIDEO" \
    --xml   "$XML" \
    --near-miss "$NEAR_MISS" \
    --output "$OUT"
