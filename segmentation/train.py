"""U-Net training + mIoU evaluation for the Oxford-Pet segmentation task."""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def per_class_iou(pred, target, n_classes=3):
    """Per-class IoU for a batch of logits; returns ``list`` (``nan`` if absent)."""
    pred = pred.argmax(dim=1)
    ious = []
    for c in range(n_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        ious.append((inter / union).item() if union > 0 else float("nan"))
    return ious


def compute_miou(pred, target, n_classes=3):
    ious = [v for v in per_class_iou(pred, target, n_classes) if not np.isnan(v)]
    return float(np.mean(ious)) if ious else 0.0


@torch.no_grad()
def evaluate_segmentation(model, loader, device, n_classes=3):
    """Return ``(mIoU, per_class_IoU_list)`` over a loader."""
    model.eval()
    mious, per_cls = [], [[] for _ in range(n_classes)]
    for imgs, masks in loader:
        out = model(imgs.to(device))
        masks = masks.to(device)
        mious.append(compute_miou(out, masks, n_classes))
        for c, v in enumerate(per_class_iou(out, masks, n_classes)):
            if not np.isnan(v):
                per_cls[c].append(v)
    miou = float(np.mean(mious)) if mious else 0.0
    pc = [float(np.mean(v)) if v else 0.0 for v in per_cls]
    return miou, pc


def train_segmentation(model, train_loader, test_loader, device, epochs=20, lr=1e-3,
                       ckpt_path="unet_resnet18_pet.pt", tag="unet_resnet18",
                       ckpt_dir="checkpoints"):
    """Train a U-Net variant; save weights + history; return the history dict."""
    os.makedirs(ckpt_dir, exist_ok=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = {"loss": [], "miou": [], "time": [], "per_class": [], "tag": tag}
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        ep_loss = []
        for imgs, masks in tqdm(train_loader, desc=f"[{tag}] epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            loss = criterion(model(imgs), masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_loss.append(loss.item())
        scheduler.step()
        elapsed = time.time() - t0
        miou, pc = evaluate_segmentation(model, test_loader, device)
        history["loss"].append(float(np.mean(ep_loss)))
        history["miou"].append(miou)
        history["time"].append(elapsed)
        history["per_class"].append(pc)
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss {history['loss'][-1]:.4f} | "
              f"mIoU {miou:.4f} | {elapsed:.1f}s")

    torch.save(model.state_dict(), ckpt_path)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{tag}.pt"))
    with open(os.path.join(ckpt_dir, f"history_{tag}.json"), "w") as fp:
        json.dump(history, fp, indent=2)
    print(f"Saved → {ckpt_path}  and  {ckpt_dir}/history_{tag}.json")
    return history
