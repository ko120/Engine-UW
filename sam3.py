import argparse
import json
from pathlib import Path

from ultralytics.models.sam import SAM3VideoSemanticPredictor

CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
}
# Reverse: name -> YOLO class id
_NAME_TO_CLS = {v: k for k, v in CLASS_NAMES.items()}


def _iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _merge_frame(frame: dict, iou_thresh: float = 0.5) -> dict:
    """NMS across all boxes in a frame, keeping highest-score box per overlap group."""
    boxes = frame["boxes_xyxy"]
    if len(boxes) <= 1:
        return frame

    scores = frame["scores"]
    # Sort by score descending
    order = sorted(range(len(boxes)), key=lambda k: scores[k], reverse=True)
    keep = []
    suppressed = set()
    for i, idx in enumerate(order):
        if idx in suppressed:
            continue
        keep.append(idx)
        for jdx in order[i + 1 :]:
            if jdx not in suppressed and _iou(boxes[idx], boxes[jdx]) > iou_thresh:
                suppressed.add(jdx)

    return {
        "frame_idx": frame["frame_idx"],
        "boxes_xyxy":  [frame["boxes_xyxy"][k]  for k in keep],
        "scores":      [frame["scores"][k]       for k in keep],
        "classes":     [frame["classes"][k]      for k in keep],
        "class_names": [frame["class_names"][k]  for k in keep],
        "ids":         [frame["ids"][k]          for k in keep],
        "n_masks":     frame["n_masks"],
    }


def run_sam3(
    source: str,
    bboxes: list | None = None,
    text: list[str] | str | None = None,
    model_path: str = "sam3.pt",
    out_path: str = "sam3_predictions",
):
    # Modified path logic: putting 'sam3' at the end of the filename
    p = Path(out_path)
    base_path = p.parent / f"{p.stem}_sam3"
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # Build chunks: each chunk is a list of class names run together in one predictor pass.
    # Accepts list[list[str]] (chunked), list[str] (flat -> one chunk), or str (one chunk).
    if isinstance(text, str):
        chunks: list[list[str]] = [[t.strip() for t in text.split(",")]]
    elif isinstance(text, list) and text and isinstance(text[0], list):
        chunks = text  # already chunked
    elif isinstance(text, list):
        chunks = [text]  # flat list -> single chunk
    else:
        chunks = [[]]  # bbox-only mode

    # Run predictor once per chunk, then unify results per frame.
    unified: dict[int, dict] = {}
    max_id_seen = 0  # tracks the highest ID used so far across all runs

    for run_idx, chunk in enumerate(chunks):
        # Offset by max ID seen so far — guarantees no collision regardless of how many tracks each run produces
        id_offset = max_id_seen

        prompts: dict = {}
        if bboxes:
            prompts["bboxes"] = bboxes
        if chunk:
            prompts["text"] = chunk

        print(f"[run {run_idx + 1}/{len(chunks)}] detecting: {chunk or '(bbox only)'}")

        # Re-create the predictor for each run so tracking state is fresh
        run_predictor = SAM3VideoSemanticPredictor(
            overrides=dict(
                conf=0.4,
                #imgsz=65,
                task="segment",
                mode="predict",
                model=model_path,
                half=True,
                save=False,
                verbose=False,
            )
        )

        results = run_predictor(source=source, stream=True, **prompts)

        for i, r in enumerate(results):
            r_cpu = r.cpu()

            if i not in unified:
                unified[i] = {
                    "frame_idx": i,
                    "boxes_xyxy": [],
                    "scores": [],
                    "classes": [],
                    "class_names": [],
                    "ids": [],
                    "n_masks": 0,
                }

            boxes = r_cpu.boxes
            if boxes is not None and len(boxes) > 0:
                raw_cls = boxes.cls.int().tolist()
                # Map local cls index (within this chunk) back to name and YOLO class id
                cls_names = [chunk[c] if c < len(chunk) else f"class_{c}" for c in raw_cls]
                mapped_cls = [_NAME_TO_CLS.get(name, run_idx * 10 + c) for name, c in zip(cls_names, raw_cls)]
                raw_ids = boxes.id.int().tolist() if boxes.id is not None else [0] * len(raw_cls)
                offset_ids = [tid + id_offset for tid in raw_ids]

                unified[i]["boxes_xyxy"].extend(boxes.xyxy.tolist())
                unified[i]["scores"].extend(boxes.conf.tolist())
                unified[i]["classes"].extend(mapped_cls)
                unified[i]["class_names"].extend(cls_names)
                unified[i]["ids"].extend(offset_ids)
                if offset_ids:
                    max_id_seen = max(max_id_seen, max(offset_ids) + 1)

            unified[i]["n_masks"] += len(r_cpu.masks.data) if r_cpu.masks is not None else 0

            if i % 100 == 0:
                n = len(r_cpu.masks.data) if r_cpu.masks is not None else 0
                print(f"  frame {i} — {n} masks")

    all_predictions = [_merge_frame(unified[k]) for k in sorted(unified)]

    pred_path = str(base_path.with_suffix(".json"))
    with open(pred_path, "w") as f:
        json.dump(all_predictions, f)
    print(f"Saved {len(all_predictions)} frames to {pred_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",     required=True,  help="input video path")
    ap.add_argument("--out-path",   required=True,  help="output base path (JSON written as <out_path>_sam3.json)")
    ap.add_argument("--model",      default="/home/brianko/Visual-Preference/test2/sam3.pt")
    ap.add_argument("--text",       nargs="+",      default=["person", "bicycle", "car", "truck"],
                                    help="class names; comma-separated groups become separate chunks "
                                         "e.g. --text 'person,bicycle' 'car,truck'")
    args = ap.parse_args()

    # Each --text arg may be comma-separated → one predictor chunk per arg
    chunks = [[t.strip() for t in grp.split(",")] for grp in args.text]

    run_sam3(
        source=args.source,
        text=chunks,
        model_path=args.model,
        out_path=args.out_path,
    )