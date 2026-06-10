"""YOLOv4 training + mAP evaluation.

``train_yolo`` implements Exercise 2b: basic augmentation (in the dataset),
anchor→box decoding (``predict_transform``), MSE/CIoU box loss + BCE
objectness/class, gradient clipping and a per-epoch mAP@0.5 readout. It uses
**bf16 AMP + gradient accumulation** so the full YOLOv4 fits in 16 GB at 608².
"""

import json
import os
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from .losses import compute_yolo_loss


@torch.no_grad()
def compute_map(model, loader, device, img_size, conf_thresh=0.05, nms_thresh=0.45,
                num_classes=80, max_images=None, pre_nms_topk=2000, coco_range=False):
    """mAP over ``loader`` using torchmetrics + per-image NMS (cap 100 dets).

    NMS runs on-device after a top-k score cap so the ~22k raw anchors don't
    bottleneck evaluation when the model is still under-confident.

    ``coco_range=False`` (default) evaluates a single IoU threshold and returns
    the scalar **mAP@0.5** (used for the fast per-epoch readout). ``coco_range=True``
    evaluates the full COCO IoU sweep (0.50:0.05:0.95) and returns the tuple
    ``(mAP@[0.5:0.95], mAP@0.5)`` — the standard COCO primary metric plus the 0.5
    slice, computed in one pass.
    """
    from torchmetrics.detection import MeanAveragePrecision
    from torchvision.ops import nms as tv_nms

    metric = MeanAveragePrecision(iou_type="bbox",
                                  iou_thresholds=None if coco_range else [0.5])
    model.eval()
    seen = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs, device.type == "cuda").float()

        conf = outputs[..., 4]
        cls_score, cls_id = outputs[..., 5:].max(dim=-1)
        scores = conf * cls_score
        xy, wh = outputs[..., 0:2], outputs[..., 2:4]
        boxes = torch.stack([xy[..., 0] - wh[..., 0] / 2, xy[..., 1] - wh[..., 1] / 2,
                             xy[..., 0] + wh[..., 0] / 2, xy[..., 1] + wh[..., 1] / 2], dim=-1)
        boxes = boxes.clamp(0, img_size)

        preds, targets = [], []
        for i in range(imgs.size(0)):
            m = scores[i] > conf_thresh
            if m.any():
                b, s, l = boxes[i][m], scores[i][m], cls_id[i][m]
                if s.numel() > pre_nms_topk:                      # cap before NMS
                    top = s.topk(pre_nms_topk).indices
                    b, s, l = b[top], s[top], l[top]
                keep = tv_nms(b, s, nms_thresh)[:100]
                preds.append(dict(boxes=b[keep].cpu(), scores=s[keep].cpu(), labels=l[keep].cpu()))
            else:
                preds.append(dict(boxes=torch.zeros((0, 4)), scores=torch.zeros(0),
                                  labels=torch.zeros(0, dtype=torch.long)))

            lab = labels[i]
            om = lab[..., 4] > 0
            if om.any():
                g = lab[om]
                gx, gy, gw, gh = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
                gboxes = torch.stack([gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2], dim=-1)
                glabels = g[:, 5:].argmax(dim=-1)
                gt = torch.cat([gboxes, glabels.float().unsqueeze(1)], dim=1)
                gt = torch.unique(gt, dim=0)          # de-duplicate multi-anchor GT
                targets.append(dict(boxes=gt[:, :4], labels=gt[:, 4].long()))
            else:
                targets.append(dict(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.long)))

        metric.update(preds, targets)
        seen += imgs.size(0)
        if max_images is not None and seen >= max_images:
            break

    res = metric.compute()
    if coco_range:
        return max(float(res["map"]), 0.0), max(float(res["map_50"]), 0.0)
    return max(float(res["map"]), 0.0)


def train_yolo(model, train_loader, val_loader, device, img_size, epochs,
               loss_type="mse", lr=1e-3, accum_steps=4, ckpt_dir="checkpoints",
               tag="yolov4_mse", use_amp=True, eval_max_images=200):
    """Train YOLOv4 and return a history dict (also saved to ``ckpt_dir``)."""
    os.makedirs(ckpt_dir, exist_ok=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = use_amp and device.type == "cuda"

    history = {"loss": [], "box": [], "conf": [], "cls": [], "map50": [], "time": [], "loss_type": loss_type}
    print(f"Training {tag}: {epochs} epochs, loss={loss_type}, img={img_size}, "
          f"batch≈{train_loader.batch_size}×{accum_steps} (accum)")

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        run = dict(loss=0.0, box=0.0, conf=0.0, cls=0.0)
        n, skipped = 0, 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"[{tag}] epoch {epoch + 1}/{epochs}")
        for bi, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                outputs = model(imgs, device.type == "cuda")
            total, box, conf, cls = compute_yolo_loss(outputs.float(), labels, img_size, loss_type)

            if not torch.isfinite(total):
                optimizer.zero_grad(); skipped += 1; continue

            (total / accum_steps).backward()
            if (bi + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step(); optimizer.zero_grad()

            bs = imgs.size(0); n += bs
            run["loss"] += total.item() * bs; run["box"] += box.item() * bs
            run["conf"] += conf.item() * bs; run["cls"] += cls.item() * bs
            pbar.set_postfix(loss=f"{run['loss']/n:.3f}", box=f"{run['box']/n:.3f}",
                             conf=f"{run['conf']/n:.3f}", cls=f"{run['cls']/n:.3f}")

        # flush any pending accumulated gradients
        if n > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step(); optimizer.zero_grad()
        scheduler.step()

        elapsed = time.time() - t0
        map50 = compute_map(model, val_loader, device, img_size, max_images=eval_max_images)
        for k in ("loss", "box", "conf", "cls"):
            history[k].append(run[k] / max(n, 1))
        history["map50"].append(map50)
        history["time"].append(elapsed)
        if skipped:
            print(f"  [warn] skipped {skipped} non-finite batches")
        print(f"Epoch {epoch+1:02d}/{epochs} | Loss {history['loss'][-1]:.4f} | "
              f"Box {history['box'][-1]:.4f} | Conf {history['conf'][-1]:.4f} | "
              f"Cls {history['cls'][-1]:.4f} | mAP@50 {map50:.4f} | {elapsed:.1f}s")

        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"{tag}.pt"))

    with open(os.path.join(ckpt_dir, f"history_{tag}.json"), "w") as fp:
        json.dump(history, fp, indent=2)
    print(f"Saved → {ckpt_dir}/{tag}.pt  and  history_{tag}.json")
    return history
