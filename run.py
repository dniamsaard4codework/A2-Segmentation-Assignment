#!/usr/bin/env python3
"""Unified training / evaluation / inference entry point for A2.

Detection (YOLOv3 / YOLOv4) and segmentation (U-Net + ResNet-18) are dispatched
from one script, exactly matching the command lines in the assignment brief::

    # Detection — inference with pretrained weights
    python run.py --model yolov3 --weights weights/yolov3.weights --image dog-cycle-car.png --infer
    python run.py --model yolov4 --weights weights/yolov4.weights --image dog-cycle-car.png --infer

    # Detection — train on COCO (ImageNet backbone init) and compare losses
    python run.py --model yolov4 --dataset coco --epochs 10 --loss mse  --train
    python run.py --model yolov4 --dataset coco --epochs 10 --loss ciou --train
    python run.py --model yolov4 --weights weights/yolov4.weights --dataset coco --evaluate

    # Segmentation — skip vs no-skip ablation
    python run.py --model unet_resnet18         --dataset oxford_pet --epochs 20 --train
    python run.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --train
    python run.py --model unet_resnet18 --weights unet_resnet18_pet.pt --dataset oxford_pet --evaluate
"""

import argparse
import json
import os
import random

import cv2
import numpy as np
import torch

DET_MODELS = {"yolov3", "yolov4"}
SEG_MODELS = {"unet_resnet18", "unet_resnet18_no_skip"}


def set_seed(seed=42):
    """Seed Python / NumPy / PyTorch for reproducible runs.

    Determinism is exact for the default ``num_workers=0`` single-process path
    (the same one the notebooks use); with multiple data-loader workers or
    non-deterministic CUDA kernels small run-to-run variation can remain. The
    notebooks define an identical ``set_seed`` and reseed right before each
    train, so a from-scratch notebook run and ``run.py`` consume the RNG in the
    same order and produce the same numbers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_weights(path):
    """Allow the brief's bare names (e.g. ``yolov4.weights``) to resolve to the
    ``weights/`` directory if not found as given."""
    if path and not os.path.exists(path):
        for alt in (os.path.join("weights", path), os.path.join("weights", os.path.basename(path))):
            if os.path.exists(alt):
                return alt
    return path


def record_metric(key, value, path="results/metrics.json"):
    """Persist a scalar result so the notebooks / README can read real numbers."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    if os.path.exists(path):
        try:
            data = json.load(open(path))
        except json.JSONDecodeError:
            data = {}
    data[key] = value
    json.dump(data, open(path, "w"), indent=2)


# ──────────────────────────────────────────────────────────────────────────────
# Detection
# ──────────────────────────────────────────────────────────────────────────────
def _build_darknet(model, img_size, device):
    from detection.darknet import Darknet
    cfg = f"cfg/{model}.cfg"
    net = Darknet(cfg)
    net.net_info["height"] = str(img_size)
    return net.to(device)


def draw_detections(img_path, output, inp_dim, names, save_path):
    """Rescale letterboxed detections back to the original image and draw boxes."""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    scale = min(inp_dim / w, inp_dim / h)
    pad_x, pad_y = (inp_dim - scale * w) / 2, (inp_dim - scale * h) / 2
    rng = np.random.RandomState(42)
    colors = rng.randint(0, 255, size=(len(names), 3)).tolist()

    if hasattr(output, "shape"):
        for row in output:
            x1 = int((row[1].item() - pad_x) / scale); y1 = int((row[2].item() - pad_y) / scale)
            x2 = int((row[3].item() - pad_x) / scale); y2 = int((row[4].item() - pad_y) / scale)
            x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(w, x2), min(h, y2)
            cls = int(row[-1].item()); color = colors[cls]
            label = f"{names[cls]} {row[5].item()*row[6].item():.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, img)
    return save_path


def detection_infer(args, device):
    from detection.util import prep_image, write_results, COCO_NAMES, load_classes
    net = _build_darknet(args.model, args.img_size, device).eval()
    weights = resolve_weights(args.weights)
    n = net.load_weights(weights)
    print(f"Loaded {n} conv layers from {weights}")
    names = load_classes("data/coco.names") if os.path.exists("data/coco.names") else COCO_NAMES

    inp = prep_image(args.image, args.img_size).to(device)
    with torch.no_grad():
        pred = net(inp, device.type == "cuda")
    out = write_results(pred, args.conf, 80, nms_conf=args.nms)

    save = args.out or f"results/det_{args.model}_{os.path.splitext(os.path.basename(args.image))[0]}.png"
    draw_detections(args.image, out, args.img_size, names, save)
    n_det = out.shape[0] if hasattr(out, "shape") else 0
    objs = sorted({names[int(r[-1])] for r in out}) if n_det else []
    print(f"{n_det} detections: {', '.join(objs)}")
    print(f"Saved → {save}")


def _coco_loaders(args):
    from torch.utils.data import DataLoader
    from detection.dataset import CocoYoloDataset, download_coco_val2017
    images_dir, ann = download_coco_val2017("data/coco")
    train_ds = CocoYoloDataset(images_dir, ann, img_size=args.img_size, train=True)
    val_ds = CocoYoloDataset(images_dir, ann, img_size=args.img_size, train=False)
    total = min(args.total_samples, len(train_ds.ids))
    n_val = args.val_samples
    train_ds.ids = train_ds.ids[: total - n_val]
    val_ds.ids = val_ds.ids[total - n_val: total]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"COCO: {len(train_ds.ids)} train / {len(val_ds.ids)} val  (img={args.img_size})")
    return train_loader, val_loader


def detection_train(args, device):
    from detection.train import train_yolo
    net = _build_darknet(args.model, args.img_size, device)
    # Exercise 2a — initialise the CSPDarknet53 backbone from ImageNet weights.
    backbone = "weights/yolov4.conv.137"
    if os.path.exists(backbone):
        n = net.load_weights(backbone)
        print(f"Initialised backbone from {backbone} ({n} conv layers)")
    else:
        print("[warn] yolov4.conv.137 not found — training backbone from scratch")

    train_loader, val_loader = _coco_loaders(args)
    tag = f"{args.model}_{args.loss}"
    train_yolo(net, train_loader, val_loader, device, args.img_size, args.epochs,
               loss_type=args.loss, lr=args.lr, accum_steps=args.accum, tag=tag)


def detection_evaluate(args, device):
    from detection.train import compute_map
    net = _build_darknet(args.model, args.img_size, device).eval()
    if args.weights:
        net.load_weights(resolve_weights(args.weights))
    _, val_loader = _coco_loaders(args)
    n_eval = args.limit or len(val_loader.dataset)
    m5095, m50 = compute_map(net, val_loader, device, args.img_size,
                             max_images=args.limit, coco_range=True)
    print(f"mAP@0.5 ({args.model}, {n_eval} val images): {m50:.4f} | "
          f"mAP@[0.5:0.95]: {m5095:.4f}")
    record_metric(f"{args.model}_pretrained_map50", round(m50, 4))
    record_metric(f"{args.model}_pretrained_map5095", round(m5095, 4))
    record_metric(f"{args.model}_pretrained_eval_images", n_eval)


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation
# ──────────────────────────────────────────────────────────────────────────────
def segmentation_train(args, device):
    from segmentation.dataset import get_pet_loaders
    from segmentation.models import build_segmentation_model
    from segmentation.train import train_segmentation
    train_loader, test_loader, *_ = get_pet_loaders("data", batch_size=args.batch_size,
                                                    num_workers=args.num_workers)
    model = build_segmentation_model(args.model)
    ckpt = args.out or f"{args.model}_pet.pt"
    train_segmentation(model, train_loader, test_loader, device, epochs=args.epochs,
                       lr=args.lr, ckpt_path=ckpt, tag=args.model)


def segmentation_evaluate(args, device):
    from segmentation.dataset import get_pet_loaders
    from segmentation.models import build_segmentation_model
    from segmentation.train import evaluate_segmentation
    _, test_loader, *_ = get_pet_loaders("data", batch_size=args.batch_size,
                                         num_workers=args.num_workers)
    model = build_segmentation_model(args.model).to(device)
    weights = resolve_weights(args.weights) if args.weights else None
    if weights and os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location=device))
    miou, pc = evaluate_segmentation(model, test_loader, device)
    print(f"Val mIoU ({args.model}): {miou:.4f}  | per-class IoU "
          f"Pet={pc[0]:.3f} Background={pc[1]:.3f} Border={pc[2]:.3f}")
    record_metric(f"{args.model}_miou", round(miou, 4))


# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="A2 — detection & segmentation runner")
    p.add_argument("--model", required=True, choices=sorted(DET_MODELS | SEG_MODELS))
    p.add_argument("--dataset", default=None, help="coco | oxford_pet")
    p.add_argument("--weights", default=None)
    p.add_argument("--image", default="dog-cycle-car.png")
    p.add_argument("--out", default=None, help="output path (image / checkpoint)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--loss", default="mse", choices=["mse", "iou", "ciou"])
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--accum", type=int, default=4, help="gradient-accumulation steps (detection)")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--total-samples", type=int, default=5000)
    p.add_argument("--val-samples", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="cap val images in --evaluate")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--nms", type=float, default=0.4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible runs")
    p.add_argument("--train", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--infer", action="store_true")
    args = p.parse_args()

    is_det = args.model in DET_MODELS
    # sensible per-domain defaults
    if args.img_size is None:
        args.img_size = (608 if args.model == "yolov4" else 416) if is_det else 128
    if args.batch_size is None:
        args.batch_size = 4 if is_det else 16
    if args.lr is None:
        args.lr = 1e-3
    device = torch.device(args.device)
    set_seed(args.seed)
    print(f"Device: {device} | model: {args.model} | seed: {args.seed}")

    if args.infer:
        assert is_det, "--infer is for detection models"
        detection_infer(args, device)
    elif args.train:
        (detection_train if is_det else segmentation_train)(args, device)
    elif args.evaluate:
        (detection_evaluate if is_det else segmentation_evaluate)(args, device)
    else:
        p.error("specify one of --train / --evaluate / --infer")


if __name__ == "__main__":
    main()
