"""Run the trained YOLO detector on image(s) and save annotated outputs.

No dataset/data.yaml needed — this is just inference for eyeballing results.

Examples:
    python predict.py --weights runs/detect/tiger_yolo11n-4/weights/best.pt \
                      --source ../test_images
    python predict.py --weights best.pt --source some_photo.jpg --conf 0.15
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--save-crops", action="store_true",
                    help="also save each detected tiger crop")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        save=True,               # annotated images -> runs/detect/predict*/
        save_crop=args.save_crops,
    )

    # brief text summary per image
    n_boxes = 0
    for r in results:
        c = len(r.boxes)
        n_boxes += c
        confs = [f"{b.conf.item():.2f}" for b in r.boxes]
        print(f"{r.path}: {c} tiger(s) {confs}")
    print(f"\nTotal tigers detected: {n_boxes}")
    print(f"Annotated images saved under: {results[0].save_dir if results else 'runs/detect/'}")


if __name__ == "__main__":
    main()
