# PET/TTC Sample

Standalone sample extracted from the RTSSI near-miss logic.

It exposes only two functions:

- `compute_pet(...)`
- `compute_ttc(...)`

Trajectory format:

```python
[(frame_index, x, y), ...]
```

Use the same tracked point you use in production, usually bbox center or foot
point.

Run the demo:

```bash
python3 pet_ttc.py
```

Expected output:

```text
PET: 0.5s
TTC: 1.7s
Minimum distance: 0.22m
```

`compute_ttc()` uses `meters_per_pixel=0.03` by default, matching the pixel
fallback in our codebase. If you have calibration, pass your calibrated scale.
