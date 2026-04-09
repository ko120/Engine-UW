"""
Draw bounding boxes onto a video from a SAM3 JSON annotation file.

Usage:
    python draw_bbx.py \
        --input  2_09_084511_3min.mp4 \
        --annot  result/hard/2_09_084511_sam3.json \
        --output 2_09_084511_bbx.mp4
"""

import argparse
import json

import cv2

CLASS_COLORS = {
    0: (50,  205,  50),   # person   → green  (BGR)
    1: (255, 255,   0),   # bicycle  → cyan   (BGR)
    2: (255, 140,   0),   # car      → orange (BGR)
    7: (0,   0,   255),   # truck    → red    (BGR)
}
CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    7: "truck",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="2_09_084511_3min.mp4")
    ap.add_argument("--annot",  default="result/hard/2_09_084511_sam3.json")
    ap.add_argument("--output", default="2_09_084511_bbx.mp4")
    args = ap.parse_args()

    print(f"Loading {args.annot} ...")
    with open(args.annot) as f:
        data = json.load(f)

    frame_annots = {}
    for entry in data:
        fidx = entry["frame_idx"]
        frame_annots[fidx] = list(zip(entry["ids"],
                                      entry["boxes_xyxy"],
                                      entry["classes"]))

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")

    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    print(f"Video: {W}x{H}  {fps:.1f}fps  {total} frames")

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for tid, box, cls in frame_annots.get(fidx, []):
            color = CLASS_COLORS.get(cls, (200, 200, 200))
            x1, y1, x2, y2 = (int(round(v)) for v in box)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            label = f"{CLASS_NAMES.get(cls, cls)} #{tid}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
            cv2.putText(frame, label, (x1 + 1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.putText(frame, f"frame {fidx}/{total-1}",
                    (6, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (220, 220, 220), 1, cv2.LINE_AA)

        writer.write(frame)

        if fidx % 500 == 0:
            print(f"  {fidx}/{total} frames done ...")

        fidx += 1

    cap.release()
    writer.release()
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
