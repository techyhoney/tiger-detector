"""Evaluate the trained YOLO detector on the TEST split, and optionally
run it on a folder of images so you can eyeball real detections.

The headline numbers for "how good is it":
  - mAP50      : detection quality at IoU 0.5 (the common headline metric)
  - mAP50-95   : stricter, averaged over IoU thresholds
  - precision / recall for the tiger class

Examples:
    python val.py --weights runs/detect/tiger_yolo11n/weights/best.pt
    python val.py --weights .../best.pt --predict path/to/new_frames/
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--conf", type=float, default=0.25,
                    help="confidence threshold for the --predict pass")
    ap.add_argument("--predict", default=None,
                    help="folder of images to run + save annotated outputs")
    args = ap.parse_args()

    model = YOLO(args.weights)

    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz)
    p, r = m.box.mp, m.box.mr           # mean precision / recall
    print("\n=== Detection metrics on the %s split ===" % args.split)
    print(f"  mAP50    : {m.box.map50:.4f}")
    print(f"  mAP50-95 : {m.box.map:.4f}")
    print(f"  precision: {p:.4f}")
    print(f"  recall   : {r:.4f}")

    if args.predict:
        print(f"\nRunning inference on {args.predict} (conf>={args.conf}) ...")
        model.predict(source=args.predict, conf=args.conf, imgsz=args.imgsz,
                      save=True)
        print("Annotated images saved under runs/detect/predict*/")


if __name__ == "__main__":
    main()
