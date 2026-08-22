"""Dataset scanning, leakage-safe splitting, transforms and dataloaders."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config

# Some camera-trap crops are slightly truncated; allow PIL to load them.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg"}
_CROP_RE = re.compile(r"[_-]?crop[_-]?\d+$", re.IGNORECASE)
_RF_RE = re.compile(r"_jpe?g\.rf\.[0-9a-f]+$", re.IGNORECASE)
_EXT_RE = re.compile(r"\.(jpe?g)$", re.IGNORECASE)


def source_key(filename: str) -> str:
    """Collapse a crop filename to its source-frame id.

    Crops from the same original frame (``..._crop0``, ``..._crop1``) must
    land in the same split, or the model is tested on near-duplicates of
    what it trained on. This strips the Roboflow signature and crop suffix
    so all crops of one frame share a key.
    """
    name = _EXT_RE.sub("", filename)
    name = _RF_RE.sub("", name)
    name = _CROP_RE.sub("", name)
    return name.lower()


def scan_dataset():
    """Return a list of samples: dicts with path, label_idx, group key."""
    c2i = config.class_to_idx()
    samples = []
    for folder in config.SOURCE_FOLDERS:
        cls = config.FOLDER_TO_CLASS[folder]
        label = c2i[cls]
        fdir = config.DATA_ROOT / folder
        if not fdir.is_dir():
            raise FileNotFoundError(f"Expected data folder not found: {fdir}")
        for p in sorted(fdir.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                samples.append(
                    {
                        "path": str(p),
                        "label": label,
                        # group key namespaced by folder to avoid cross-folder
                        # collisions of identical numeric prefixes.
                        "group": f"{folder}/{source_key(p.name)}",
                    }
                )
    if not samples:
        raise RuntimeError(f"No images found under {config.DATA_ROOT}")
    return samples


def split_samples(samples):
    """Group-aware, stratified train/val/test split (no frame leakage)."""
    groups = defaultdict(list)
    for s in samples:
        groups[s["group"]].append(s)

    group_keys = list(groups.keys())
    # A group is "positive" if it contains any tiger crop -> stratify on that
    # so tigers are proportionally represented in every split.
    pos_idx = config.positive_idx()
    group_pos = [
        int(any(s["label"] == pos_idx for s in groups[g])) for g in group_keys
    ]

    test_size = config.TEST_FRAC
    val_size = config.VAL_FRAC / (1.0 - test_size)  # fraction of the remainder

    train_val_g, test_g = train_test_split(
        group_keys,
        test_size=test_size,
        random_state=config.SEED,
        stratify=group_pos,
    )
    tv_pos = [
        int(any(s["label"] == pos_idx for s in groups[g])) for g in train_val_g
    ]
    train_g, val_g = train_test_split(
        train_val_g,
        test_size=val_size,
        random_state=config.SEED,
        stratify=tv_pos,
    )

    def flatten(gkeys):
        out = []
        for g in gkeys:
            out.extend(groups[g])
        return out

    return flatten(train_g), flatten(val_g), flatten(test_g)


class LetterboxResize:
    """Resize preserving aspect ratio, pad to a square with gray fill.

    Keeps the whole animal visible (no squashing, no aggressive crop),
    which matters for tight detector crops.
    """

    def __init__(self, size: int, fill: int = 114):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size),
                           (self.fill, self.fill, self.fill))
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


def build_transforms(train: bool):
    common_tail = [
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ]
    if train:
        return transforms.Compose(
            [
                # Downsize to the 224px square FIRST, then augment the small
                # image. Rotating/color-jittering a 2400px crop and *then*
                # shrinking it wastes most of the CPU work per sample.
                LetterboxResize(config.IMG_SIZE, config.LETTERBOX_FILL),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12, fill=config.LETTERBOX_FILL),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                *common_tail,
                transforms.RandomErasing(p=0.25),
            ]
        )
    return transforms.Compose(
        [LetterboxResize(config.IMG_SIZE, config.LETTERBOX_FILL), *common_tail]
    )


class CropDataset(Dataset):
    def __init__(self, samples, train: bool):
        self.samples = samples
        self.tf = build_transforms(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        with Image.open(s["path"]) as img:
            x = self.tf(img)
        return x, s["label"]


def compute_class_weights(train_samples) -> torch.Tensor:
    """Inverse-frequency weights for class-balanced cross-entropy."""
    n = len(config.CLASSES)
    counts = [0] * n
    for s in train_samples:
        counts[s["label"]] += 1
    total = sum(counts)
    weights = [total / (n * max(1, c)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def build_dataloaders():
    samples = scan_dataset()
    train_s, val_s, test_s = split_samples(samples)

    # Keep workers alive between epochs and let each buffer several batches
    # ahead, so the GPU isn't starved waiting on JPEG decode/augment.
    extra = {}
    if config.NUM_WORKERS > 0:
        extra = {"persistent_workers": True, "prefetch_factor": 4}

    train_dl = DataLoader(
        CropDataset(train_s, train=True),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        **extra,
    )
    val_dl = DataLoader(
        CropDataset(val_s, train=False),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        **extra,
    )
    test_dl = DataLoader(
        CropDataset(test_s, train=False),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        **extra,
    )
    meta = {
        "n_train": len(train_s),
        "n_val": len(val_s),
        "n_test": len(test_s),
        "class_weights": compute_class_weights(train_s),
    }
    return train_dl, val_dl, test_dl, meta
