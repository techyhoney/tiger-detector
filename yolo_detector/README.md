# YOLO Tiger Detector (training + evaluation)

Single-stage YOLO detector for tigers. This folder covers **training and
testing only** — Pi export/deployment comes later. Goal: find out how well a
Pi-sized detector (YOLO11n) actually performs before committing to it.

## Recommended datasets (mix for all conditions)

| Dataset | What it gives | Format | Link |
|---------|---------------|--------|------|
| **ATRW** | 4,434 imgs / 9,496 tiger boxes, wild+zoo, daylight | COCO | https://cvwc2019.github.io/challenge.html |
| **LILA WCS Camera Traps** | real camera-trap: **night IR, forest, occlusion** (filter *Panthera tigris*) | COCO | https://lila.science/datasets/wcscameratraps/ |
| **Roboflow Universe "tiger"** | quick variety top-up, already YOLO | YOLO | https://universe.roboflow.com |
| **iNaturalist/GBIF + MegaDetector** | daylight-color volume, auto-boxed | — | run MegaDetector to auto-label |

**Also add non-tiger animals (leopard/elephant/deer) and empty frames as
negatives** (images with no boxes) so the detector doesn't false-fire on them.

## Pipeline

```
COCO datasets ──coco_to_yolo.py──► yolo_dataset/{images,labels}/{train,val,test} ──train.py──► best.pt ──val.py──► mAP
```

## Setup (on the A4000)
```bash
pip install -r requirements.txt
```

## 1. Convert each source to YOLO format
Run once per split per dataset — they accumulate into one `yolo_dataset/`:
```bash
python coco_to_yolo.py --coco atrw/annotations/train.json --images atrw/images \
    --out yolo_dataset --split train --keep tiger --link
python coco_to_yolo.py --coco atrw/annotations/val.json   --images atrw/images \
    --out yolo_dataset --split val   --keep tiger --link
# add LILA WCS the same way (filter --keep "tiger"); add --negatives on
# leopard/elephant/empty sources so they become negative images.
```
Roboflow exports are already YOLO — copy their `images/` + `labels/` straight
into `yolo_dataset/` under the matching split.

**Hold out a real test split** that ideally comes from *different cameras* than
train, so the mAP reflects field performance, not same-camera memorization.

## 2. Point `data.yaml` at the dataset
Edit `path:` in [data.yaml](data.yaml) to the absolute `yolo_dataset/` path.

## 3. Train
```bash
python train.py                      # YOLO11n (Pi-sized), 640px, 100 epochs
python train.py --model yolo11s.pt   # bigger model = the accuracy ceiling
```
Uses the A4000 automatically. Weights + curves land in
`runs/detect/tiger_yolo11n/`.

## 4. Test — "how good is it?"
```bash
python val.py --weights runs/detect/tiger_yolo11n/weights/best.pt
# see it on real frames:
python val.py --weights .../best.pt --predict path/to/new_frames/
```
Reports **mAP50**, **mAP50-95**, precision, recall. Annotated predictions save
under `runs/detect/predict*/`.

## What number is "good"?
- **mAP50 ≥ 0.90** on a *different-camera* test set is strong for tiger detection.
- Compare **yolo11n vs yolo11s** mAP: if n is close to s, the Pi-sized model is
  plenty. If n lags a lot, you may need a bigger input size or the accelerator.

## Next (later)
Export `best.pt` → NCNN/ONNX for the Pi, and benchmark FPS. Not covered here yet.
