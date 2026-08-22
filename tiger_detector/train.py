"""Train the tiger classifier.

Two-phase transfer learning:
  Phase 1 (FREEZE_EPOCHS): backbone frozen, train the new head fast.
  Phase 2 (FINETUNE_EPOCHS): unfreeze all, fine-tune at a low LR.
Best checkpoint is chosen on validation macro-F1 (protects the minority
class better than raw accuracy). Early stopping avoids overfitting.

Run:  python train.py
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

import config
import utils
from data import build_dataloaders


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    use_amp = train and scaler is not None and scaler.is_enabled()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in tqdm(loader, leave=False, desc="train" if train else "eval"):
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type="cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    acc = sum(p == t for p, t in zip(all_preds, all_labels)) / max(1, n)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / max(1, n), acc, macro_f1


def make_optimizer(params, lr):
    return torch.optim.AdamW(params, lr=lr, weight_decay=config.WEIGHT_DECAY)


def cosine_lr(optimizer, base_lr, epoch, total):
    lr = 0.5 * base_lr * (1 + math.cos(math.pi * epoch / max(1, total)))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-epochs", type=int, default=config.FREEZE_EPOCHS)
    ap.add_argument("--finetune-epochs", type=int, default=config.FINETUNE_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    args = ap.parse_args()
    config.BATCH_SIZE = args.batch_size

    utils.set_seed()
    device = utils.get_device()
    use_amp = device.type == "cuda"          # mixed precision on the A4000
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"Device: {device}  (AMP mixed-precision: {use_amp})")

    train_dl, val_dl, test_dl, meta = build_dataloaders()
    print(f"Samples  train={meta['n_train']}  val={meta['n_val']}  "
          f"test={meta['n_test']}")
    print(f"Classes: {config.CLASSES}")
    print(f"Class weights: {meta['class_weights'].tolist()}")

    model = utils.build_model(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=meta["class_weights"].to(device),
        label_smoothing=config.LABEL_SMOOTHING,
    )

    history = []
    best_f1, best_epoch, epochs_no_improve = -1.0, -1, 0
    global_epoch = 0

    def evaluate_and_maybe_save():
        nonlocal best_f1, best_epoch, epochs_no_improve
        vloss, vacc, vf1 = run_epoch(model, val_dl, device, criterion)
        print(f"  val   loss={vloss:.4f} acc={vacc:.4f} macroF1={vf1:.4f}")
        history.append({"epoch": global_epoch, "val_loss": vloss,
                        "val_acc": vacc, "val_macro_f1": vf1})
        if vf1 > best_f1:
            best_f1, best_epoch, epochs_no_improve = vf1, global_epoch, 0
            utils.save_checkpoint(model, config.CKPT_PATH,
                                  extra={"val_macro_f1": vf1})
            print(f"  ** saved best (macroF1={vf1:.4f}) -> {config.CKPT_PATH.name}")
        else:
            epochs_no_improve += 1
        return vf1

    t0 = time.time()

    # ---- Phase 1: frozen backbone, train head only ----
    if args.freeze_epochs > 0:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
        opt = make_optimizer(
            [p for p in model.parameters() if p.requires_grad], config.HEAD_LR
        )
        for e in range(args.freeze_epochs):
            global_epoch += 1
            lr = cosine_lr(opt, config.HEAD_LR, e, args.freeze_epochs)
            tl, ta, tf1 = run_epoch(model, train_dl, device, criterion, opt, scaler)
            print(f"[freeze {e+1}/{args.freeze_epochs}] lr={lr:.2e} "
                  f"loss={tl:.4f} acc={ta:.4f} macroF1={tf1:.4f}")
            evaluate_and_maybe_save()

    # ---- Phase 2: unfreeze everything, fine-tune ----
    for p in model.parameters():
        p.requires_grad = True
    opt = make_optimizer(model.parameters(), config.FINETUNE_LR)
    for e in range(args.finetune_epochs):
        global_epoch += 1
        lr = cosine_lr(opt, config.FINETUNE_LR, e, args.finetune_epochs)
        tl, ta, tf1 = run_epoch(model, train_dl, device, criterion, opt, scaler)
        print(f"[finetune {e+1}/{args.finetune_epochs}] lr={lr:.2e} "
              f"loss={tl:.4f} acc={ta:.4f} macroF1={tf1:.4f}")
        evaluate_and_maybe_save()
        if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"Early stopping (no val improvement for "
                  f"{config.EARLY_STOP_PATIENCE} epochs).")
            break

    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min. Best val macroF1={best_f1:.4f} "
          f"at epoch {best_epoch}.")
    with open(config.OUTPUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"History -> {config.OUTPUT_DIR / 'history.json'}")
    print("Next: python evaluate.py   (measures test-set accuracy)")


if __name__ == "__main__":
    main()
