"""Full-frame end-to-end pipeline: this is what runs on the Raspberry Pi.

  full frame ──► MegaDetector v6 (compact) ──► animal crops ──► tiger classifier ──► verdict

Stage 1 (MegaDetector) is a pretrained, download-once model — no training.
Stage 2 is the classifier you trained here (torch checkpoint or ONNX).

On the Pi, prefer --onnx (lighter than PyTorch). MegaDetector runs via the
PytorchWildlife package; install it with `pip install PytorchWildlife`.
The exact MegaDetector call can differ slightly between PytorchWildlife
versions — if the import/vars below don't match yours, adjust `detect_animals`
only; everything downstream stays the same.

Examples:
  python pipeline.py path/to/full_frame.jpg
  python pipeline.py path/to/frames_folder/ --onnx --save-crops outputs/hits
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import config
from data import IMG_EXTS
from infer import _EVAL_TF

DET_CONF = 0.2   # MegaDetector detection confidence threshold
ANIMAL_CATEGORY = 0  # MegaDetector: 0=animal, 1=person, 2=vehicle


# --------------------------------------------------------------------------
# Stage 1 — animal detector (MegaDetector v6 compact), lazy-loaded singleton.
# --------------------------------------------------------------------------
_detector = None


def _load_detector():
    global _detector
    if _detector is not None:
        return _detector
    try:
        import torch
        from PytorchWildlife.models import detection as pw_detection
    except ImportError as e:
        raise SystemExit(
            "MegaDetector needs PytorchWildlife. Install it with:\n"
            "    pip install PytorchWildlife\n"
            f"(import error: {e})"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # MDV6-yolov9-c = the compact, edge-friendly variant.
    _detector = pw_detection.MegaDetectorV6(
        device=device, pretrained=True, version="MDV6-yolov9-c"
    )
    return _detector


def detect_animals(image_path):
    """Return a list of (x1, y1, x2, y2) pixel boxes for animals in a frame."""
    det = _load_detector()
    result = det.single_image_detection(str(image_path))
    # PytorchWildlife returns a dict with a supervision Detections object.
    detections = result.get("detections", result)
    boxes = []
    xyxy = getattr(detections, "xyxy", None)
    conf = getattr(detections, "confidence", None)
    cls = getattr(detections, "class_id", None)
    if xyxy is None:
        return boxes
    for i in range(len(xyxy)):
        if conf is not None and conf[i] < DET_CONF:
            continue
        if cls is not None and int(cls[i]) != ANIMAL_CATEGORY:
            continue
        x1, y1, x2, y2 = (int(v) for v in xyxy[i][:4])
        boxes.append((x1, y1, x2, y2))
    return boxes


# --------------------------------------------------------------------------
# Stage 2 — tiger classifier (torch checkpoint or ONNX).
# --------------------------------------------------------------------------
class Classifier:
    def __init__(self, use_onnx: bool):
        self.use_onnx = use_onnx
        self.pos = config.positive_idx()
        if use_onnx:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(
                str(config.ONNX_PATH), providers=["CPUExecutionProvider"])
            self.in_name = self.sess.get_inputs()[0].name
        else:
            import utils
            self.device = utils.get_device()
            self.model, _ = utils.load_checkpoint(
                config.CKPT_PATH, map_location=self.device)
            self.model.to(self.device)

    def prob_tiger(self, crop: Image.Image) -> float:
        x = _EVAL_TF(crop).unsqueeze(0)
        if self.use_onnx:
            logits = self.sess.run(None, {self.in_name: x.numpy()})[0][0]
            p = np.exp(logits - logits.max())
            p = p / p.sum()
            return float(p[self.pos])
        import torch
        import torch.nn.functional as F
        with torch.no_grad():
            p = F.softmax(self.model(x.to(self.device)), dim=1)[0]
        return float(p[self.pos].cpu())


def process_frame(frame_path, clf, threshold, save_dir=None):
    with Image.open(frame_path) as im:
        frame = im.convert("RGB")
    boxes = detect_animals(frame_path)
    hits = []
    for j, (x1, y1, x2, y2) in enumerate(boxes):
        crop = frame.crop((x1, y1, x2, y2))
        p = clf.prob_tiger(crop)
        is_tiger = p >= threshold
        if is_tiger:
            hits.append((j, p, (x1, y1, x2, y2)))
            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                crop.save(Path(save_dir) /
                          f"{Path(frame_path).stem}_tiger{j}_{p:.2f}.jpg")
    return boxes, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="full-frame image or a folder of frames")
    ap.add_argument("--onnx", action="store_true",
                    help="use the exported ONNX classifier (recommended on Pi)")
    ap.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD)
    ap.add_argument("--save-crops", default=None,
                    help="folder to save tiger crops that fire")
    args = ap.parse_args()

    p = Path(args.target)
    frames = ([q for q in sorted(p.iterdir()) if q.suffix.lower() in IMG_EXTS]
              if p.is_dir() else [p])
    if not frames:
        print("No frames found.")
        return

    clf = Classifier(use_onnx=args.onnx)
    total_hits = 0
    for f in frames:
        boxes, hits = process_frame(f, clf, args.threshold, args.save_crops)
        total_hits += len(hits)
        tag = f"  >>> {len(hits)} TIGER(S)" if hits else ""
        print(f"{f.name:40s}  animals={len(boxes)}{tag}")
        for j, prob, box in hits:
            print(f"      box{j} P(tiger)={prob:.3f} @ {box}")
    print(f"\nDone. {len(frames)} frame(s), {total_hits} tiger detection(s).")


if __name__ == "__main__":
    main()
