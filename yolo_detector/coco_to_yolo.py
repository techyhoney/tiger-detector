"""Convert a COCO-format dataset (ATRW, LILA WCS, etc.) to YOLO detection
labels, keeping only the categories you care about (default: tiger).

YOLO wants: one `.txt` per image under labels/<split>/, each line
    <class> <cx> <cy> <w> <h>          # all normalized 0..1
and the image under images/<split>/.

Images with none of the kept categories become NEGATIVES (an empty label
file) when --negatives is set — useful for feeding leopard/elephant/empty
frames so the detector learns not to fire on them.

Example (ATRW train split, tigers only):
    python coco_to_yolo.py \
        --coco  atrw/annotations/train.json \
        --images atrw/images \
        --out   yolo_dataset --split train \
        --keep  tiger --link

Run once per split (train/val/test) and per source dataset; they all
accumulate into the same --out folder.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


def matches(name: str, keep_terms) -> bool:
    name = name.lower()
    return any(t in name for t in keep_terms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="COCO annotation .json")
    ap.add_argument("--images", required=True, help="dir with the image files")
    ap.add_argument("--out", required=True, help="output YOLO dataset root")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--keep", default="tiger",
                    help="comma-separated category name substrings to keep")
    ap.add_argument("--negatives", action="store_true",
                    help="also emit images with no kept category as negatives")
    ap.add_argument("--link", action="store_true",
                    help="symlink images instead of copying (saves disk)")
    args = ap.parse_args()

    keep_terms = [t.strip().lower() for t in args.keep.split(",") if t.strip()]
    coco = json.loads(Path(args.coco).read_text())

    catid_keep = {c["id"] for c in coco["categories"] if matches(c["name"], keep_terms)}
    if not catid_keep:
        raise SystemExit(f"No categories matched {keep_terms}. "
                         f"Available: {[c['name'] for c in coco['categories']]}")
    print(f"Keeping category ids {catid_keep} for terms {keep_terms}")

    images = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    out = Path(args.out)
    img_out = out / "images" / args.split
    lbl_out = out / "labels" / args.split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    src_dir = Path(args.images)
    n_pos = n_neg = n_box = n_skip = 0

    for img_id, im in images.items():
        fname = im["file_name"]
        src = src_dir / fname
        if not src.exists():
            n_skip += 1
            continue

        w = im.get("width"); h = im.get("height")
        if not w or not h:
            with Image.open(src) as _im:
                w, h = _im.size

        lines = []
        for a in anns_by_img.get(img_id, []):
            if a["category_id"] not in catid_keep:
                continue
            x, y, bw, bh = a["bbox"]           # COCO: absolute x,y,w,h (top-left)
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            # clip to [0,1] to survive slightly out-of-bounds boxes
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            nw, nh = min(nw, 1), min(nh, 1)
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines and not args.negatives:
            continue

        # place image (flatten name to avoid subdir collisions)
        stem = Path(fname).name
        dst = img_out / stem
        if not dst.exists():
            if args.link:
                os.symlink(src.resolve(), dst)
            else:
                dst.write_bytes(src.read_bytes())
        (lbl_out / (Path(stem).stem + ".txt")).write_text("\n".join(lines))

        if lines:
            n_pos += 1; n_box += len(lines)
        else:
            n_neg += 1

    print(f"[{args.split}] positives={n_pos} (boxes={n_box})  "
          f"negatives={n_neg}  missing_images_skipped={n_skip}")
    print(f"-> {img_out}  and  {lbl_out}")


if __name__ == "__main__":
    main()
