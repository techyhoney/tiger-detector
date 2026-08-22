# Tiger Detector (Stage-2 classifier)

**4-class classifier with a tiger-only decision**, for the two-stage
camera-trap pipeline. Trains on the cropped dataset in `dataset/`, targets
95%+ tiger accuracy, and exports a model that runs on a Raspberry Pi 5 (8GB) CPU.
Code lives in `tiger_detector/`.

The model learns all four classes — **tiger / leopard / elephant / negative** —
which gives a cleaner tiger boundary (leopards and elephants become distinct
clusters instead of one blurry "not-tiger" blob) and a running start on
multi-species. At **inference we only threshold on P(tiger)**, so deployment
stays a simple tiger / not-tiger call. The 4-class head costs ~10 KB and no
measurable latency versus binary.

## Pipeline

```
triggered frame ─► MegaDetector v6 compact ─► animal crops ─► THIS classifier ─► tiger? 
                   (pretrained, no training)                  (MobileNetV3-Large)
```

## Layout

| File | Purpose |
|------|---------|
| `config.py` | All settings: paths, classes, hyperparameters. **Edit here.** |
| `data.py` | Scans `../dataset/`, leakage-safe stratified split, transforms |
| `utils.py` | Device select, model build, checkpoint save/load |
| `train.py` | Two-phase transfer learning → best checkpoint by val macro-F1 |
| `evaluate.py` | Test accuracy, per-class report, confusion matrix, threshold tuning |
| `export_onnx.py` | Export to ONNX (+ optional int8) for the Pi |
| `infer.py` | Classify a crop / folder of crops (torch or ONNX) |
| `pipeline.py` | Full-frame → MegaDetector → classifier (runs on the Pi) |

## Setup

Training deps are already installed. For export + on-device inference:

```bash
pip install onnx onnxruntime
```

## 1. Train

```bash
cd tiger_detector
python train.py
```

Device is auto-detected: **CUDA (your A4000) with AMP mixed precision** when
available, else Apple MPS, else CPU. Writes the best model to
`outputs/tiger_classifier.pt` and `outputs/history.json`. On the A4000 the
default 30 epochs run in roughly 8–15 min (early-stops if val macro-F1
plateaus). Bump `BATCH_SIZE`/`NUM_WORKERS` in `config.py` to use the A4000
more fully.

## 2. Evaluate (does it hit 95%?)

```bash
python evaluate.py
```

Prints overall test accuracy + per-class precision/recall/F1, saves
`outputs/confusion_matrix.png`, and reports the best P(tiger) threshold. If
you're short of 95%, in order: raise the minority class with more tiger images
from the net (run them through MegaDetector first for matching crops), try
`MODEL_NAME = "tf_efficientnet_lite0"` in `config.py`, or train longer.

## 3. Export for the Raspberry Pi

```bash
python export_onnx.py --quantize      # writes .onnx and .int8.onnx
python infer.py ../dataset/tiger --onnx   # spot-check the exported model
```

Copy `outputs/tiger_classifier.onnx` (or the int8 one) to the Pi.

## 4. Run on the Pi (full frames)

```bash
pip install PytorchWildlife onnxruntime          # on the Pi
python pipeline.py path/to/frames/ --onnx --save-crops hits/
```

MegaDetector downloads once. For *triggered* camera-trap use the per-event
budget (detector ~0.3–1 s + classifier ~0.01 s) fits comfortably on the Pi 5
CPU — no accelerator needed.

## Add a new species later (e.g. sloth bear)

Drop a `dataset/sloth_bear/` folder of crops, then edit only `config.py`:

```python
CLASSES = ["negative", "tiger", "leopard", "elephant", "sloth_bear"]
FOLDER_TO_CLASS = {
    "tiger": "tiger", "leopard": "leopard", "elephant": "elephant",
    "negative": "negative", "sloth_bear": "sloth_bear",
}
```

Retrain (`train.py`), re-export, drop the new model on the Pi. **The detector
and `pipeline.py` never change** — only the classifier grows one output.
