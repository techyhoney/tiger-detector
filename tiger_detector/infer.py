"""Run the trained classifier on a single crop or a folder of crops.

Works with either the PyTorch checkpoint (default) or the exported ONNX
model (--onnx), so you can test the exact artifact you'll ship to the Pi.

Examples:
  python infer.py path/to/crop.jpg
  python infer.py path/to/folder_of_crops/
  python infer.py path/to/crop.jpg --onnx --threshold 0.6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
import utils
from data import IMG_EXTS, LetterboxResize
from PIL import Image
from torchvision import transforms

_EVAL_TF = transforms.Compose([
    LetterboxResize(config.IMG_SIZE, config.LETTERBOX_FILL),
    transforms.ToTensor(),
    transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
])


def load_image_tensor(path) -> torch.Tensor:
    with Image.open(path) as img:
        return _EVAL_TF(img)


def gather_paths(target: str):
    p = Path(target)
    if p.is_dir():
        return [q for q in sorted(p.iterdir()) if q.suffix.lower() in IMG_EXTS]
    return [p]


def predict_torch(paths, threshold):
    device = utils.get_device()
    model, ckpt = utils.load_checkpoint(config.CKPT_PATH, map_location=device)
    model.to(device)
    classes = ckpt["classes"]
    pos = classes.index(config.POSITIVE_CLASS)
    with torch.no_grad():
        for p in paths:
            x = load_image_tensor(p).unsqueeze(0).to(device)
            prob = F.softmax(model(x), dim=1)[0].cpu().numpy()
            report(p, prob, classes, pos, threshold)


def predict_onnx(paths, threshold):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(config.ONNX_PATH),
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    classes = config.CLASSES
    pos = config.positive_idx()
    for p in paths:
        x = load_image_tensor(p).unsqueeze(0).numpy().astype(np.float32)
        logits = sess.run(None, {in_name: x})[0][0]
        prob = np.exp(logits - logits.max())
        prob = prob / prob.sum()
        report(p, prob, classes, pos, threshold)


def report(path, prob, classes, pos, threshold):
    p_tiger = float(prob[pos])
    verdict = "TIGER" if p_tiger >= threshold else "not tiger"
    dist = "  ".join(f"{c}={prob[i]:.3f}" for i, c in enumerate(classes))
    print(f"{Path(path).name:45s}  {verdict:10s}  P(tiger)={p_tiger:.3f}  [{dist}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="image file or folder of crops")
    ap.add_argument("--onnx", action="store_true", help="use exported ONNX model")
    ap.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD)
    args = ap.parse_args()

    paths = gather_paths(args.target)
    if not paths:
        print("No images found.")
        return
    if args.onnx:
        predict_onnx(paths, args.threshold)
    else:
        predict_torch(paths, args.threshold)


if __name__ == "__main__":
    main()
