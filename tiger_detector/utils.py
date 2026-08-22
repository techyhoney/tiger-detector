"""Shared helpers: device selection, model building, checkpoint I/O."""
from __future__ import annotations

import random

import numpy as np
import timm
import torch

import config


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(pretrained: bool = True) -> torch.nn.Module:
    return timm.create_model(
        config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=len(config.CLASSES),
    )


def save_checkpoint(model: torch.nn.Module, path, extra: dict | None = None):
    ckpt = {
        "model_state": model.state_dict(),
        "model_name": config.MODEL_NAME,
        "classes": config.CLASSES,
        "img_size": config.IMG_SIZE,
        "norm_mean": config.NORM_MEAN,
        "norm_std": config.NORM_STD,
        "threshold": config.DECISION_THRESHOLD,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = timm.create_model(
        ckpt["model_name"], pretrained=False, num_classes=len(ckpt["classes"])
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
