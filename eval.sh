#!/bin/bash
set -e

GT="/content/engineVideo/label/annotations_easy.xml"
BASE="/content/engineVideo/result/easy"

echo "=== [all] ==="
python eval_metrics.py \
    --pred "/content/engineVideo/result/easy/all/124441_10-13min_sam3.json" \
    --gt   "$GT"

echo ""
echo "=== [separate] ==="
python eval_metrics.py \
    --pred "/content/engineVideo/result/easy/seperate/easy_sam3.json" \
    --gt   "$GT"

echo ""
echo "=== [yolo] ==="
python eval_metrics.py \
    --pred "/content/engineVideo/result/easy/yolo/124441_10-13min.json" \
    --gt   "$GT"
