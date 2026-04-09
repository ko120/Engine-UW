#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=3
cd /home/brianko/Visual-Preference/test2

python yolo_train_hparam_tune.py \
    --video                /home/brianko/Visual-Preference/test2/2_09_084511_3min.mp4 \
    --gt                   /home/brianko/Visual-Preference/test2/sam3_2_09_084511_3min_cvat.xml \
    --base-model           yolo26l.pt \
    --project              runs/hptune \
    --out                  train_hparam_results.csv \
    --epochs               100 \
    --patience             10 \
    --test-ratio           0.20 \
    --max-frames           400 \
    --max-frames-per-class 100 \
    --seed                 42
