"""Export the trained checkpoint to ONNX for Raspberry Pi deployment.

Produces tiger_classifier.onnx. With --quantize it also writes an int8
dynamically-quantized model (smaller + faster on the Pi CPU, tiny
accuracy cost). Verify the quantized model with:
    python evaluate.py         (accuracy on Mac, torch)
    python infer.py <folder> --onnx   (spot-check the ONNX artifact)

Run:  python export_onnx.py [--quantize]
"""
from __future__ import annotations

import argparse

import torch

import config
import utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", action="store_true",
                    help="also emit an int8 dynamic-quantized ONNX model")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not config.CKPT_PATH.exists():
        raise FileNotFoundError(f"No checkpoint at {config.CKPT_PATH}.")

    model, _ = utils.load_checkpoint(config.CKPT_PATH, map_location="cpu")
    model.eval()

    dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(config.ONNX_PATH),
        input_names=["input"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX -> {config.ONNX_PATH}")

    # Sanity check that ONNX Runtime agrees with PyTorch.
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(config.ONNX_PATH),
                                    providers=["CPUExecutionProvider"])
        with torch.no_grad():
            torch_out = model(dummy).numpy()
        onnx_out = sess.run(None, {"input": dummy.numpy()})[0]
        max_diff = float(np.abs(torch_out - onnx_out).max())
        print(f"Torch vs ONNX max logit diff: {max_diff:.2e} "
              f"({'OK' if max_diff < 1e-3 else 'CHECK'})")
    except ImportError:
        print("(Install onnxruntime to verify the export.)")

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        q_path = config.ONNX_PATH.with_name("tiger_classifier.int8.onnx")
        quantize_dynamic(str(config.ONNX_PATH), str(q_path),
                         weight_type=QuantType.QInt8)
        print(f"Exported int8 ONNX -> {q_path}")


if __name__ == "__main__":
    main()
