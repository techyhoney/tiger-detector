"""Train a STANDALONE YOLO elephant detector (Ultralytics).

Separate model, separate dataset from the tiger detector — this reads
data_elephant.yaml (class 0 = elephant), not data.yaml. Nothing is shared
with the tiger run except the code style, so the two can be trained,
evaluated and versioned independently.

Hyperparameters are deliberately IDENTICAL to train.py so the two runs are
directly comparable: if elephant mAP lands well above tiger mAP, that's a
statement about the data, not about the recipe.

Starts from COCO-pretrained yolo26n.pt, which already includes an elephant
class — so the backbone has real elephant features to fine-tune, unlike the
tiger run which had to learn the species from scratch. Expect it to converge
faster; the useful signal is per-class recall on night-IR and partial-body
frames, not the headline mAP.

YOLO26 is end-to-end (NMS-free): inference needs no NMS post-processing, so
there is no conf/iou NMS tuning at deploy time. Training API is unchanged.

Runs on the A4000 automatically (device=0). Results, weights and plots land
in runs/detect/<name>/ — including results.png (loss/mAP curves) and the
best checkpoint at weights/best.pt.

Examples:
    python train_elephant.py                          # yolo26n, 640, 100 epochs
    python train_elephant.py --model yolo26s.pt --epochs 150 --batch 32
    python train_elephant.py --imgsz 960              # small/distant elephants
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n.pt",
                    help="pretrained weights to start from (yolo26n/s/m.pt)")
    ap.add_argument("--data", default="data_elephant.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="input size (multiple of 32). 640 is the sweet spot; "
                         "use 960/1280 if elephants are small/distant in frame")
    ap.add_argument("--batch", type=int, default=16,
                    help="use -1 for Ultralytics auto-batch on the A4000")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="elephant_yolo26n")
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
    # Ultralytics auto-increments the run folder (name, name-2, ...), so read
    # the real path back instead of guessing it from --name.
    best = model.trainer.save_dir / "weights" / "best.pt"
    print(f"\nDone. Best weights: {best}")
    print(f"Next: python val.py --weights {best} --data {args.data}")


if __name__ == "__main__":
    main()
