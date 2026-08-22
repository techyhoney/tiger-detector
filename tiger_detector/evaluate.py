"""Evaluate the trained classifier on the held-out TEST split.

Reports overall accuracy, per-class precision/recall/F1, a confusion
matrix (saved as PNG), and — for the binary tiger build — ROC-AUC,
PR-AUC and the probability threshold that maximizes tiger F1.

Run:  python evaluate.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config
import utils
from data import build_dataloaders


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        p = F.softmax(model(x), dim=1).cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main():
    utils.set_seed()
    device = utils.get_device()
    if not config.CKPT_PATH.exists():
        raise FileNotFoundError(
            f"No checkpoint at {config.CKPT_PATH}. Run train.py first."
        )
    model, ckpt = utils.load_checkpoint(config.CKPT_PATH, map_location=device)
    model.to(device)
    print(f"Loaded {config.CKPT_PATH.name} "
          f"(val macroF1={ckpt.get('val_macro_f1', 'n/a')})")

    # Same seed -> identical test split as training.
    _, _, test_dl, meta = build_dataloaders()
    print(f"Test samples: {meta['n_test']}")

    probs, labels = collect(model, test_dl, device)
    preds = probs.argmax(1)

    acc = (preds == labels).mean()
    print(f"\nOverall test accuracy: {acc*100:.2f}%")
    print("\n" + classification_report(
        labels, preds, target_names=config.CLASSES, digits=4, zero_division=0))

    cm = confusion_matrix(labels, preds)
    ConfusionMatrixDisplay(cm, display_labels=config.CLASSES).plot(
        cmap="Blues", values_format="d")
    plt.title("Test confusion matrix")
    plt.tight_layout()
    cm_path = config.OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    print(f"Confusion matrix -> {cm_path}")

    # Tiger-vs-rest analysis on P(tiger) — the metric you actually deploy on.
    # Works for any number of classes (we only look at the tiger column).
    if True:
        pos = config.positive_idx()
        print(f"\n--- Tiger vs rest (deployment metric, P('{config.POSITIVE_CLASS}')) ---")
        y_true = (labels == pos).astype(int)
        y_score = probs[:, pos]
        try:
            auc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            print(f"\nTiger ROC-AUC: {auc:.4f}   PR-AUC: {ap:.4f}")
        except ValueError:
            print("\n(ROC/PR-AUC unavailable — only one class in test set.)")

        prec, rec, thr = precision_recall_curve(y_true, y_score)
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
        if len(thr):
            print(f"Best-F1 threshold on P(tiger): {thr[best]:.3f} "
                  f"(F1={f1[best]:.4f}, precision={prec[best]:.4f}, "
                  f"recall={rec[best]:.4f})")
            print("  -> set config.DECISION_THRESHOLD to this for deployment.")


if __name__ == "__main__":
    main()
