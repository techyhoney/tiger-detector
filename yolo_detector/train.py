"""Train a YOLO tiger detector (Ultralytics).

Defaults to YOLO11n — the nano model you'll eventually put on the Pi. Train
yolo11s too if you want to see the accuracy ceiling on a bigger model; the
gap tells you how much the Pi-sized model is leaving on the table.

Runs on the A4000 automatically (device=0). Results, weights and plots land
in runs/detect/<name>/ — including results.png (loss/mAP curves) and the
best checkpoint at weights/best.pt.

Examples:
    python train.py                          # yolo11n, 640, 100 epochs
    python train.py --model yolo11s.pt --epochs 150 --batch 32
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt",
                    help="pretrained weights to start from (yolo11n/s/m.pt)")
    ap.add_argument("--data", default="data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16,
                    help="use -1 for Ultralytics auto-batch on the A4000")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="tiger_yolo11n")
    ap.add_argument("--patience", type=int, default=25,
                    help="early-stop patience (epochs w/o val mAP gain)")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        patience=args.patience,
        # sensible camera-trap augmentation
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # color/brightness (helps day vs IR)
        degrees=5.0, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, close_mosaic=10,
        cos_lr=True,
        plots=True,
    )
    print("\nDone. Best weights: runs/detect/%s/weights/best.pt" % args.name)
    print("Next: python val.py --weights runs/detect/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
