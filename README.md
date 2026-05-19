# engineVideo

Video analytics pipeline for road-user safety analysis: detection, tracking,
trajectory prediction, and post-encroachment-time (PET) / time-to-collision
(TTC) metrics on traffic footage.

## Pipeline

```
Video ──► YOLO / SAM3 tracking ──► tracker JSON
                                          │
              CVAT XML ground truth ──────┤
                                          ▼
                          traj_predict_from_cvat.py   (linear + Kalman)
                          transformer_trajectory.py   (Transformer)
                                          │
                                          ▼
                      safety_analytics.py   (PET + heatmaps + report)
                                          │
                                          ▼
                  visualize_traj_predictions.py   (annotated video)
                  make_pet_clips.py               (one MP4 per PET event)
```

## Install

```bash
pip install -r requirements.txt
```

`ultralytics` brings in `torch`; install the CUDA build separately if you
need GPU.

## Quickstart — easy split

The repo ships small ground-truth annotations and example safety-analytics
outputs for the easy split (`label/easy/`, `output/easy/safety/`). The
source MP4 is not included — supply your own and point `VIDEO` at it:

```bash
export VIDEO=/path/to/your/easy.mp4
./scripts/run_easy.sh                # linear + Kalman + TTC overlay
./scripts/run_transformer_easy.sh    # PET + safety analytics + annotated video
PRED_XML=output/easy/predictions_with_transformer.xml ./scripts/run_pet_clips_easy.sh
./scripts/eval.sh                    # eval bundled tracker JSONs vs ground truth
```

All shell scripts read paths from environment variables — they have no
hardcoded user-specific paths. See the comment at the top of each script
for the required variables. Override the interpreter with `PYTHON=...`
(default `python3`).

## Python entry points

| File | Purpose |
|------|---------|
| `yolo.py` | YOLO detect or track a video; outputs tracker JSON |
| `sam3.py` | SAM3 text-prompted tracking; outputs tracker JSON |
| `yolo_sam2_tracker.py` | YOLO BoT-SORT + SAM3 occlusion-filling pipeline |
| `eval_model.py` | Evaluate a YOLO model vs CVAT XML (P/R/F1/AP50/MOTA) |
| `eval_metrics.py` | Compare tracker JSON vs CVAT XML (HOTA, mAP50, gaps) |
| `compare_trackers.py` | Side-by-side comparison of tracker outputs |
| `traj_predict_from_cvat.py` | Linear + Kalman trajectory prediction from CVAT |
| `transformer_trajectory.py` | Transformer trajectory predictor (train + eval) |
| `safety_analytics.py` | PET + occupancy/conflict heatmaps + text report |
| `visualize_traj_predictions.py` | Render predictions onto video |
| `make_pet_clips.py` | Render a short MP4 per PET event |
| `annotate_video.py` | Bounding-box + trajectory overlay |
| `draw_bbx.py` | Bounding-box-only overlay |
| `converter.py` | Convert YOLO JSON ↔ CVAT XML |
| `yolo_train_hparam_tune.py` | YOLO fine-tuning random-search HP loop |
| `pet/pet_ttc.py` | Standalone PET/TTC reference implementation |

Each script prints `--help`; refer to its module docstring for full usage.

## Shell scripts and required env vars

| Script | Required env vars |
|--------|-------------------|
| `scripts/run_easy.sh`, `scripts/annot_video.sh` | `VIDEO` |
| `scripts/run_transformer_easy.sh` | `VIDEO` |
| `scripts/run_pet_clips_easy.sh` | `VIDEO`, `PRED_XML` |
| `scripts/run_hard.sh` | `VIDEO`, `ANNOT_XML` |
| `scripts/run_cali_20.sh` | `VIDEO`, `ANNOT_XML` |
| `scripts/run_sam3_easy.sh`, `scripts/run_sam3_hard.sh` | `VIDEO`, `YOLO_MODEL`, `SAM3_MODEL` |
| `scripts/run_hparam_tune.sh` | `VIDEO`, `GT_XML` |
| `scripts/eval.sh` | none — uses bundled `result/easy/*` + `label/easy/annotations_easy.xml` |

## Repo layout

```
*.py                 # Python entry points (cross-import, all at root)
scripts/             # Shell driver scripts (run_*.sh, eval.sh, annot_video.sh)
pet/                 # Standalone PET/TTC reference module + demo
label/easy/          # CVAT ground truth + derived predictions/TTC (committed)
output/easy/safety/  # Example PET + heatmap outputs (committed)
result/easy/         # Example tracker JSONs used by eval.sh (committed)
result/hard/         # Example tracker JSONs for the hard split (committed)
models/              # gitignored — produced by training
output/easy/*.mp4    # gitignored — produced by run_*.sh
label/Cali/, label/Cali_raw/, label/hard/, output/cali_20/, output/hard/
                     # gitignored — large data, not bundled
```

Trained checkpoints, source videos, rendered MP4s, and frame dumps are
gitignored — regenerate them locally from the scripts above.
