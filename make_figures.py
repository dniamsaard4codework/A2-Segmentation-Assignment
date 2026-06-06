#!/usr/bin/env python3
"""Generate every figure used by the notebooks and README into ``results/``.

Each figure is produced from a real run; functions are independent and skip
themselves cleanly if their inputs (weights / checkpoints / history files) are
not present yet, so this script can be re-run as artifacts appear.
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RESULTS = "results"
CKPT = "checkpoints"
os.makedirs(RESULTS, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save(fig, name):
    path = os.path.join(RESULTS, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _load_history(tag):
    path = os.path.join(CKPT, f"history_{tag}.json")
    return json.load(open(path)) if os.path.exists(path) else None


# ──────────────────────────────────────────────────────────────────────────────
# Static / conceptual figures
# ──────────────────────────────────────────────────────────────────────────────
def fig_mish():
    import torch.nn.functional as F
    x = torch.linspace(-6, 6, 400)
    mish = x * torch.tanh(F.softplus(x))
    relu = F.relu(x)
    leaky = F.leaky_relu(x, 0.1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, mish, label="Mish  (YOLOv4)", lw=2.5, color="crimson")
    ax.plot(x, relu, label="ReLU", lw=1.8, ls="--", color="steelblue")
    ax.plot(x, leaky, label="Leaky ReLU (0.1)  (YOLOv3)", lw=1.8, ls=":", color="green")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_title("Mish vs ReLU vs Leaky-ReLU", fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, "mish_activation.png")


def fig_yolov4_anchors():
    from detection.dataset import ANCHORS_V4, STRIDES
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    names = ["stride 8 (small)", "stride 16 (medium)", "stride 32 (large)"]
    fig, ax = plt.subplots(figsize=(7, 7))
    for scale, (anchors, c, nm) in enumerate(zip(ANCHORS_V4, colors, names)):
        for (w, h) in anchors:
            ax.add_patch(plt.Rectangle((-w / 2, -h / 2), w, h, fill=False, edgecolor=c, lw=2))
        ax.plot([], [], color=c, lw=2, label=f"{nm}: {anchors}")
    ax.set_xlim(-320, 320); ax.set_ylim(-320, 320); ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_title("YOLOv4 anchor boxes (9 priors @ 608 input)", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), fontsize=8)
    _save(fig, "yolov4_anchors.png")


# ──────────────────────────────────────────────────────────────────────────────
# Detection — pretrained inference
# ──────────────────────────────────────────────────────────────────────────────
def _detect_image(net, img_path, dim, conf=0.5, nms=0.4):
    import cv2
    from detection.util import prep_image, write_results
    inp = prep_image(img_path, dim).to(DEVICE)
    with torch.no_grad():
        out = write_results(net(inp, DEVICE.type == "cuda"), conf, 80, nms)
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(dim / w, dim / h); px, py = (dim - scale * w) / 2, (dim - scale * h) / 2
    dets = []
    if hasattr(out, "shape"):
        for r in out:
            x1 = (r[1].item() - px) / scale; y1 = (r[2].item() - py) / scale
            x2 = (r[3].item() - px) / scale; y2 = (r[4].item() - py) / scale
            dets.append((x1, y1, x2, y2, int(r[-1].item()), r[5].item() * r[6].item()))
    return img, dets


def _draw(ax, img, dets, names):
    rng = np.random.RandomState(1)
    colors = rng.rand(80, 3)
    ax.imshow(img)
    for x1, y1, x2, y2, c, s in dets:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=colors[c], lw=2.5))
        ax.text(x1, y1 - 4, f"{names[c]} {s:.2f}", color="white", fontsize=9,
                bbox=dict(facecolor=colors[c], alpha=0.8, pad=1, edgecolor="none"))
    ax.axis("off")


def fig_det_inference_compare():
    from detection.darknet import Darknet
    from detection.util import COCO_NAMES
    if not (os.path.exists("weights/yolov3.weights") and os.path.exists("weights/yolov4.weights")):
        print("  [skip] det compare — weights missing"); return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (model, wt, dim) in zip(axes, [("YOLOv3", "weights/yolov3.weights", 416),
                                           ("YOLOv4", "weights/yolov4.weights", 608)]):
        net = Darknet(f"cfg/{model.lower()}.cfg").to(DEVICE).eval()
        net.net_info["height"] = str(dim); net.load_weights(wt)
        img, dets = _detect_image(net, "dog-cycle-car.png", dim)
        _draw(ax, img, dets, COCO_NAMES)
        ax.set_title(f"{model} (pretrained, {dim}px RGB)", fontweight="bold")
        del net; torch.cuda.empty_cache()
    fig.suptitle("Pretrained detection — YOLOv3 vs YOLOv4", fontsize=13, fontweight="bold")
    _save(fig, "det_inference_compare.png")


def fig_det_coco_grid(n=6):
    from detection.darknet import Darknet
    from detection.util import COCO_NAMES
    if not os.path.exists("weights/yolov4.weights") or not os.path.isdir("data/coco/val2017"):
        print("  [skip] coco grid — weights/data missing"); return
    net = Darknet("cfg/yolov4.cfg").to(DEVICE).eval()
    net.net_info["height"] = "608"; net.load_weights("weights/yolov4.weights")
    imgs = sorted(glob.glob("data/coco/val2017/*.jpg"))[:n]
    fig, axes = plt.subplots(2, n // 2, figsize=(16, 8))
    for ax, p in zip(axes.flat, imgs):
        img, dets = _detect_image(net, p, 608, conf=0.5)
        _draw(ax, img, dets, COCO_NAMES)
    fig.suptitle("YOLOv4 (pretrained) on COCO val2017", fontsize=13, fontweight="bold")
    _save(fig, "det_coco_grid.png")
    del net; torch.cuda.empty_cache()


def fig_det_pretrained_eval(n_images=500):
    """Per-class AP bar + a precision-recall curve for pretrained YOLOv4."""
    from torch.utils.data import DataLoader
    from torchmetrics.detection import MeanAveragePrecision
    from torchvision.ops import nms as tv_nms
    from detection.darknet import Darknet
    from detection.dataset import CocoYoloDataset
    from detection.util import COCO_NAMES
    if not os.path.exists("weights/yolov4.weights") or not os.path.isdir("data/coco/val2017"):
        print("  [skip] pretrained eval — weights/data missing"); return

    net = Darknet("cfg/yolov4.cfg").to(DEVICE).eval()
    net.net_info["height"] = "608"; net.load_weights("weights/yolov4.weights")
    ds = CocoYoloDataset("data/coco/val2017", "data/coco/annotations/instances_val2017.json",
                         img_size=608, train=False)
    ds.ids = ds.ids[:n_images]
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5], class_metrics=True)

    pr_scores, pr_tp = [], []   # for a single-class PR curve (person, class 0)
    n_pos_person = 0
    with torch.no_grad():
        for imgs, labels in loader:
            out = net(imgs.to(DEVICE), DEVICE.type == "cuda").float()
            conf = out[..., 4]; cs, cid = out[..., 5:].max(-1); sc = conf * cs
            xy, wh = out[..., :2], out[..., 2:4]
            boxes = torch.stack([xy[..., 0] - wh[..., 0] / 2, xy[..., 1] - wh[..., 1] / 2,
                                 xy[..., 0] + wh[..., 0] / 2, xy[..., 1] + wh[..., 1] / 2], -1).clamp(0, 608)
            preds, tgts = [], []
            for i in range(imgs.size(0)):
                m = sc[i] > 0.05
                if m.any():
                    b, s, l = boxes[i][m], sc[i][m], cid[i][m]
                    k = tv_nms(b, s, 0.45)[:100]
                    preds.append(dict(boxes=b[k].cpu(), scores=s[k].cpu(), labels=l[k].cpu()))
                else:
                    preds.append(dict(boxes=torch.zeros((0, 4)), scores=torch.zeros(0), labels=torch.zeros(0, dtype=torch.long)))
                lab = labels[i]; om = lab[..., 4] > 0
                if om.any():
                    g = lab[om]; gx, gy, gw, gh = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
                    gb = torch.stack([gx - gw / 2, gy - gh / 2, gx + gw / 2, gy + gh / 2], -1)
                    gl = g[:, 5:].argmax(-1)
                    gt = torch.unique(torch.cat([gb, gl.float().unsqueeze(1)], 1), dim=0)
                    tgts.append(dict(boxes=gt[:, :4], labels=gt[:, 4].long()))
                    _accumulate_person_pr(preds[-1], tgts[-1], pr_scores, pr_tp)
                    n_pos_person += int((gt[:, 4] == 0).sum())
                else:
                    tgts.append(dict(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.long)))
            metric.update(preds, tgts)
    res = metric.compute()
    del net; torch.cuda.empty_cache()

    # per-class AP bar (top 15 by AP)
    ap = res.get("map_per_class"); classes = res.get("classes")
    if ap is not None and ap.ndim > 0:
        ap = ap.numpy(); classes = classes.numpy()
        order = np.argsort(-ap)[:15]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([COCO_NAMES[classes[i]] for i in order], ap[order], color="teal")
        ax.set_ylabel("AP@0.5"); ax.set_title(
            f"YOLOv4 pretrained — top-15 per-class AP@0.5  (mAP={float(res['map']):.3f}, {n_images} val imgs)",
            fontweight="bold")
        ax.tick_params(axis="x", rotation=45); ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), ha="right")
        _save(fig, "det_per_class_ap.png")

    # person PR curve
    if pr_scores:
        s = np.array(pr_scores); tp = np.array(pr_tp)
        order = np.argsort(-s); tp = tp[order]
        cum_tp = np.cumsum(tp); cum_fp = np.cumsum(1 - tp)
        recall = cum_tp / max(n_pos_person, 1)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(recall, precision, lw=2.2, color="darkred")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title("Precision–Recall — 'person' @ IoU 0.5\n(YOLOv4 pretrained)", fontweight="bold")
        ax.grid(alpha=0.3)
        _save(fig, "det_pr_curve.png")


def _accumulate_person_pr(pred, tgt, pr_scores, pr_tp, cls=0, iou_thr=0.5):
    from detection.util import bbox_iou
    pb = pred["boxes"][pred["labels"] == cls]; ps = pred["scores"][pred["labels"] == cls]
    gb = tgt["boxes"][tgt["labels"] == cls]
    matched = torch.zeros(len(gb), dtype=torch.bool)
    order = torch.argsort(ps, descending=True)
    for i in order:
        if len(gb) == 0:
            pr_scores.append(ps[i].item()); pr_tp.append(0); continue
        ious = bbox_iou(pb[i].unsqueeze(0), gb)
        j = int(torch.argmax(ious))
        if ious[j] >= iou_thr and not matched[j]:
            matched[j] = True; pr_scores.append(ps[i].item()); pr_tp.append(1)
        else:
            pr_scores.append(ps[i].item()); pr_tp.append(0)


# ──────────────────────────────────────────────────────────────────────────────
# Detection — training history
# ──────────────────────────────────────────────────────────────────────────────
def fig_det_history(tag):
    h = _load_history(tag)
    if h is None:
        print(f"  [skip] det history {tag} — missing"); return
    ep = range(1, len(h["loss"]) + 1)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"YOLOv4 training history — {h.get('loss_type', tag)} loss", fontsize=13, fontweight="bold")
    for a, key, title, col in [((0, 0), "loss", "Total loss", "steelblue"),
                               ((0, 1), "box", "Box loss", "darkorange"),
                               ((1, 0), "conf", "Objectness (conf) loss", "tomato"),
                               ((1, 1), "map50", "mAP@0.5 (val)", "green")]:
        ax[a].plot(ep, h[key], marker="o", color=col)
        ax[a].set_title(title); ax[a].set_xlabel("epoch"); ax[a].grid(alpha=0.3)
    _save(fig, f"det_history_{tag}.png")


def fig_det_loss_comparison():
    hm, hc = _load_history("yolov4_mse"), _load_history("yolov4_ciou")
    if hm is None or hc is None:
        print("  [skip] det loss comparison — need both mse & ciou histories"); return
    em = range(1, len(hm["loss"]) + 1); ec = range(1, len(hc["loss"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    # Box losses are different functions on different scales — compare *relative*
    # convergence (each normalised to its own epoch-1 value).
    bm = [v / hm["box"][0] for v in hm["box"]]; bc = [v / hc["box"][0] for v in hc["box"]]
    ax[0].plot(em, bm, marker="o", label="MSE/IoU", color="darkorange")
    ax[0].plot(ec, bc, marker="s", label="CIoU", color="purple")
    ax[0].set_title("Box loss — relative convergence (÷ epoch 1)")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("box loss / box loss[0]"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(em, hm["map50"], marker="o", label="MSE/IoU", color="darkorange")
    ax[1].plot(ec, hc["map50"], marker="s", label="CIoU", color="purple")
    ax[1].set_title("mAP@0.5 (val)"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[2].bar(["MSE/IoU", "CIoU"], [max(hm["map50"]), max(hc["map50"])], color=["darkorange", "purple"])
    ax[2].set_title("Best mAP@0.5"); ax[2].grid(axis="y", alpha=0.3)
    fig.suptitle("YOLOv4 loss comparison — MSE/IoU vs CIoU", fontsize=13, fontweight="bold")
    _save(fig, "det_loss_comparison.png")


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation figures
# ──────────────────────────────────────────────────────────────────────────────
def _pet_test_data():
    from segmentation.dataset import get_pet_loaders
    _, _, _, test_data = get_pet_loaders("data", batch_size=16, num_workers=0, download=False)
    return test_data


def fig_seg_dataset_samples():
    from segmentation.dataset import get_pet_loaders, denormalize, CLASS_COLORS, CLASS_NAMES
    try:
        _, _, train_data, _ = get_pet_loaders("data", num_workers=0, download=False)
    except Exception as e:
        print(f"  [skip] seg samples — {e}"); return
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        img, mask = train_data[i * 100]
        axes[i][0].imshow(denormalize(img)); axes[i][0].set_title("Image"); axes[i][0].axis("off")
        axes[i][1].imshow(CLASS_COLORS[mask.numpy()]); axes[i][1].set_title("Mask"); axes[i][1].axis("off")
        axes[i][2].imshow(denormalize(img)); axes[i][2].imshow(CLASS_COLORS[mask.numpy()], alpha=0.5)
        axes[i][2].set_title("Overlay"); axes[i][2].axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[i] / 255) for i in range(3)]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=3)
    fig.suptitle("Oxford-IIIT Pet — image + segmentation mask", fontsize=13)
    _save(fig, "seg_dataset_samples.png")


def fig_seg_curves():
    hs = _load_history("unet_resnet18"); hn = _load_history("unet_resnet18_no_skip")
    if hs is None or hn is None:
        print("  [skip] seg curves — histories missing"); return
    e = range(1, len(hs["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(e, hs["loss"], marker="o", label="skip", color="steelblue")
    ax[0].plot(e, hn["loss"], marker="s", label="no-skip", color="firebrick")
    ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(e, hs["miou"], marker="o", label="skip", color="steelblue")
    ax[1].plot(e, hn["miou"], marker="s", label="no-skip", color="firebrick")
    ax[1].set_title("Validation mIoU"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle("U-Net / ResNet-18 — skip vs no-skip", fontsize=13, fontweight="bold")
    _save(fig, "seg_curves.png")


def _load_seg_model(name):
    from segmentation.models import build_segmentation_model
    ckpt = f"{name}_pet.pt"
    if not os.path.exists(ckpt):
        ckpt = os.path.join(CKPT, f"{name}.pt")
    if not os.path.exists(ckpt):
        return None
    model = build_segmentation_model(name, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)); model.eval()
    return model


def fig_seg_predictions(name):
    from segmentation.dataset import denormalize, CLASS_COLORS, CLASS_NAMES
    model = _load_seg_model(name)
    if model is None:
        print(f"  [skip] seg predictions {name} — checkpoint missing"); return
    test = _pet_test_data()
    fig, axes = plt.subplots(5, 4, figsize=(14, 18))
    for ax, t in zip(axes[0], ["Input", "Ground Truth", "Prediction", "Overlay"]):
        ax.set_title(t, fontweight="bold")
    for row in range(5):
        img, mask = test[row * 50]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(DEVICE)).argmax(1).squeeze().cpu().numpy()
        axes[row][0].imshow(denormalize(img))
        axes[row][1].imshow(CLASS_COLORS[mask.numpy()])
        axes[row][2].imshow(CLASS_COLORS[pred])
        axes[row][3].imshow(denormalize(img)); axes[row][3].imshow(CLASS_COLORS[pred], alpha=0.5)
        for ax in axes[row]:
            ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[i] / 255) for i in range(3)]
    fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=3)
    fig.suptitle(f"U-Net predictions — {name}", fontsize=13)
    _save(fig, f"seg_pred_{name}.png")


def fig_seg_skip_vs_noskip():
    from segmentation.dataset import denormalize, CLASS_COLORS
    ms = _load_seg_model("unet_resnet18"); mn = _load_seg_model("unet_resnet18_no_skip")
    if ms is None or mn is None:
        print("  [skip] seg skip-vs-noskip — checkpoints missing"); return
    test = _pet_test_data()
    rows = 4
    fig, axes = plt.subplots(rows, 4, figsize=(14, 4 * rows))
    for ax, t in zip(axes[0], ["Input", "Ground Truth", "Skip", "No-skip"]):
        ax.set_title(t, fontweight="bold")
    for r in range(rows):
        img, mask = test[r * 60 + 20]
        with torch.no_grad():
            ps = ms(img.unsqueeze(0).to(DEVICE)).argmax(1).squeeze().cpu().numpy()
            pn = mn(img.unsqueeze(0).to(DEVICE)).argmax(1).squeeze().cpu().numpy()
        axes[r][0].imshow(denormalize(img))
        axes[r][1].imshow(CLASS_COLORS[mask.numpy()])
        axes[r][2].imshow(CLASS_COLORS[ps])
        axes[r][3].imshow(CLASS_COLORS[pn])
        for ax in axes[r]:
            ax.axis("off")
    fig.suptitle("Skip vs no-skip predictions — note boundary sharpness", fontsize=13, fontweight="bold")
    _save(fig, "seg_skip_vs_noskip.png")


def fig_seg_per_class_iou():
    from segmentation.dataset import CLASS_NAMES
    hs = _load_history("unet_resnet18"); hn = _load_history("unet_resnet18_no_skip")
    if hs is None or hn is None:
        print("  [skip] seg per-class — histories missing"); return
    pcs = hs["per_class"][-1]; pcn = hn["per_class"][-1]
    x = np.arange(3); w = 0.35
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(x - w / 2, pcs, w, label="skip", color="steelblue")
    ax[0].bar(x + w / 2, pcn, w, label="no-skip", color="firebrick")
    ax[0].set_xticks(x); ax[0].set_xticklabels(CLASS_NAMES); ax[0].set_ylabel("IoU")
    ax[0].set_title("Per-class IoU (final epoch)"); ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)
    ax[1].bar(["skip", "no-skip"], [max(hs["miou"]), max(hn["miou"])], color=["steelblue", "firebrick"])
    ax[1].set_title("Best Val mIoU"); ax[1].set_ylabel("mIoU"); ax[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate([max(hs["miou"]), max(hn["miou"])]):
        ax[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold")
    fig.suptitle("Skip connections — per-class IoU & mIoU", fontsize=13, fontweight="bold")
    _save(fig, "seg_per_class_iou.png")


def fig_maskrcnn_demo():
    import cv2
    import torchvision
    import torchvision.transforms.functional as TF
    try:
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT").eval().to(DEVICE)
    except Exception as e:
        print(f"  [skip] mask r-cnn — {e}"); return
    from detection.util import COCO_NAMES
    names = ["__bg__"] + COCO_NAMES
    img = cv2.cvtColor(cv2.imread("dog-cycle-car.png"), cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        out = model([TF.to_tensor(img).to(DEVICE)])[0]
    keep = out["scores"].cpu().numpy() >= 0.7
    boxes = out["boxes"].cpu().numpy()[keep]; labels = out["labels"].cpu().numpy()[keep]
    masks = out["masks"].cpu().numpy()[keep]
    rng = np.random.RandomState(42); COLORS = rng.randint(0, 255, (91, 3))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img); axes[0].set_title("Original", fontweight="bold"); axes[0].axis("off")
    axes[1].imshow(img)
    for b, l in zip(boxes, labels):
        axes[1].add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False,
                                        edgecolor=COLORS[l] / 255, lw=2.5))
        axes[1].text(b[0], b[1] - 4, names[l], color="white", fontsize=9,
                     bbox=dict(facecolor=COLORS[l] / 255, alpha=0.8, pad=1, edgecolor="none"))
    axes[1].set_title("Bounding boxes (Faster R-CNN head)", fontweight="bold"); axes[1].axis("off")
    overlay = img.copy().astype(np.float32)
    for m, l in zip(masks, labels):
        overlay[m[0] > 0.5] = overlay[m[0] > 0.5] * 0.5 + COLORS[l] * 0.5
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("Instance masks (Mask R-CNN head)", fontweight="bold"); axes[2].axis("off")
    fig.suptitle("Mask R-CNN = Faster R-CNN + mask head (pretrained COCO)", fontsize=13, fontweight="bold")
    _save(fig, "maskrcnn_demo.png")
    del model; torch.cuda.empty_cache()


GROUPS = {
    "static": [fig_mish, fig_yolov4_anchors],
    "det_pretrained": [fig_det_inference_compare, fig_det_coco_grid, fig_det_pretrained_eval],
    "det_train": [lambda: fig_det_history("yolov4_mse"), lambda: fig_det_history("yolov4_ciou"),
                  fig_det_loss_comparison],
    "seg": [fig_seg_dataset_samples, fig_seg_curves,
            lambda: fig_seg_predictions("unet_resnet18"),
            lambda: fig_seg_predictions("unet_resnet18_no_skip"),
            fig_seg_skip_vs_noskip, fig_seg_per_class_iou],
    "maskrcnn": [fig_maskrcnn_demo],
}


def main(groups=None):
    groups = groups or list(GROUPS)
    print(f"Generating figures → {RESULTS}  (groups: {', '.join(groups)})")
    for g in groups:
        for fn in GROUPS.get(g, []):
            try:
                fn()
            except Exception as e:
                print(f"  [error] {getattr(fn, '__name__', 'lambda')}: {e}")
    print("Done.")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
