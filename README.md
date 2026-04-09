# engineVideo

Video analytics pipeline for detecting, tracking, and predicting trajectories of road users (person, bicycle, car, truck) in traffic footage.

## Pipeline Overview

```
Video ──► YOLO detect/track ──► SAM3 gap-fill ──► pseudo-labels (JSON)
                                                        │
                              CVAT XML ground-truth ────┤
                                                        │
                                                   eval_metrics.py
                                                   traj_predict_from_cvat.py
```

## Scripts

| Script | Purpose |
|--------|---------|
| `yolo.py` | YOLO detect or track a video; outputs JSON |
| `sam3.py` | SAM3 text-prompted tracking; outputs JSON |
| `yolo_sam2_tracker.py` | Combined YOLO BoT-SORT + SAM3 occlusion-filling pipeline |
| `eval_model.py` | Evaluate YOLO model vs CVAT XML (Precision/Recall/F1/AP50/MOTA) |
| `eval_metrics.py` | Compare tracker JSON output vs CVAT XML ground truth |
| `compare_trackers.py` | Side-by-side comparison of multiple tracker outputs |
| `traj_predict_from_cvat.py` | Linear + Kalman trajectory prediction from CVAT annotations |
| `visualize_traj_predictions.py` | Render predicted trajectories onto video |
| `annotate_video.py` | Draw bounding boxes + trajectory overlays on video |
| `draw_bbx.py` | Draw bounding boxes from JSON onto video frames |
| `converter.py` | Convert between annotation formats |
| `yolo_train_hparam_tune.py` | Hyperparameter search for YOLO fine-tuning |

## Shell Scripts

| Script | Purpose |
|--------|---------|
| `run_sam3_easy.sh` | Run YOLO + SAM3 on easy-split video |
| `run_sam3_hard.sh` | Run YOLO + SAM3 on hard-split video |
| `run_hparam_tune.sh` | Launch YOLO hyperparameter tuning |
| `eval.sh` | Evaluate all tracker variants vs ground truth |
| `annot_video.sh` | Render trajectory predictions onto video |

## Install

```bash
pip install -U ultralytics opencv-python numpy scipy tqdm
```

## Quick Start

**Track a video (YOLO only):**
```bash
python yolo.py --mode track --source input.mp4 --model best.pt --out-path output.json
```

**Track with SAM3 gap-filling:**
```bash
python sam3.py --source input.mp4 --model sam3.pt --out-path result/ --text person bicycle car truck
```

**Evaluate vs ground truth:**
```bash
python eval_metrics.py --pred result/output_sam3.json --gt annotations.xml
```

**Trajectory prediction:**
```bash
python traj_predict_from_cvat.py --input annotations.xml --output predictions.xml --horizon 30 --lookback 10
```

**Visualize predictions:**
```bash
python visualize_traj_predictions.py --video input.mp4 --xml predictions.xml --output annotated.mp4
```

**Fine-tune YOLO with hyperparameter search:**
```bash
python yolo_train_hparam_tune.py --video input.mp4 --gt annotations.xml --base-model yolo26l.pt
```

## Metrics

`eval_model.py` / `eval_metrics.py` report:
- Per-class: Precision, Recall, F1, AP50
- Overall: mAP50, mAP50-95, MOTA, ID Switches

`traj_predict_from_cvat.py` reports per-track and dataset-wide ADE / FDE for linear and Kalman predictors.
