#!/usr/bin/env python3
"""Programmatically build the two A2 notebooks with nbformat.

The notebooks chdir to the repo root, then display the real figures produced by
``make_figures.py`` and the histories produced by training — plus a couple of
fast live cells (cfg parsing, one YOLOv4 inference). Heavy training is NOT run
inside the notebooks, so ``nbconvert --execute`` is quick and deterministic.
"""

import nbformat as nbf

SETUP = """\
import os, sys, json
# Run from the repository root so all paths match run.py / script.* .
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, os.getcwd())
import torch, numpy as np, pandas as pd
from IPython.display import Image, display

def show(path, width=None):
    if os.path.exists(path):
        display(Image(path, width=width))
    else:
        print(f'[figure not generated yet: {path}] — run: python make_figures.py')

def load_history(tag):
    p = f'checkpoints/history_{tag}.json'
    return json.load(open(p)) if os.path.exists(p) else None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
"""


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text)


# ──────────────────────────────────────────────────────────────────────────────
# A2-01  Object Detection
# ──────────────────────────────────────────────────────────────────────────────
def build_detection():
    c = []
    c.append(md(
        "# A2-01 · Object Detection — YOLOv4 from the Darknet config\n\n"
        "We parse the **YOLOv4** Darknet configuration into PyTorch (the same way the lab parsed "
        "YOLOv3), load the official `yolov4.weights`, run 608×608 **RGB** inference, then train "
        "YOLOv4 on COCO and compare **MSE/IoU vs CIoU** box-regression losses.\n\n"
        "**What was added on top of the YOLOv3 parser** (`detection/darknet.py`, `detection/util.py`):\n"
        "* **Mish** activation (`x·tanh(softplus(x))`) across CSPDarknet53\n"
        "* **`[maxpool]`** layers for the SPP block (sizes 5/9/13, stride 1)\n"
        "* **`[route]` concatenating >2 layers** (SPP fuses four maps) and CSP **`groups`** splits\n"
        "* **`scale_x_y`** grid-sensitivity decoding and **partial weight loading** for the "
        "ImageNet backbone `yolov4.conv.137`"))
    c.append(code(SETUP))

    c.append(md(
        "## 1. The detection lineage: R-CNN → Faster R-CNN → YOLO\n\n"
        "| Model | Proposals | Stages | Speed | Idea |\n|---|---|---|---|---|\n"
        "| R-CNN (2014) | Selective Search + 2000× CNN | 3 separate | ~47 s/img | CNN features per region |\n"
        "| Fast R-CNN (2015) | Selective Search + 1× CNN | 1 joint | ~2 s/img | shared map + RoI pool |\n"
        "| Faster R-CNN (2016) | RPN on GPU | 2 joint | ~0.2 s/img | learned proposals |\n"
        "| **YOLOv3/v4** | **none** | **1 (single-shot)** | **real-time** | **dense grid prediction** |"))

    c.append(md(
        "### Why is YOLOv3 faster than Faster R-CNN?\n\n"
        "**Faster R-CNN is two-stage.** A Region Proposal Network first scans the backbone feature "
        "map and emits ~1–2k candidate regions; a second detection head then crops each proposal "
        "(RoI-Align) and classifies/regresses it *individually*. That per-region second stage runs "
        "hundreds of times per image, doesn't batch cleanly, and sits — together with proposal "
        "generation and its NMS — on the critical path.\n\n"
        "**YOLOv3 is single-shot.** One fully-convolutional forward pass densely predicts, at every "
        "cell of three feature-map scales (strides 8/16/32), the box offsets, objectness and class "
        "scores for a fixed set of anchors. There is **no proposal stage and no per-RoI head** — "
        "detection is one batched CNN pass plus a single cheap NMS. Hence ~30–60 FPS for YOLOv3 vs "
        "~5–7 FPS for Faster R-CNN at comparable accuracy. The historical cost was weaker "
        "localisation on small/overlapping objects, which multi-scale anchors and CIoU loss "
        "(YOLOv4) largely recover."))

    c.append(md(
        "## 2. YOLOv4 architecture\n\n"
        "**CSPDarknet53 backbone** (Mish, cross-stage-partial blocks) → **neck** = SPP "
        "(maxpool 5/9/13) + **PANet** (top-down + bottom-up feature fusion) → **3 YOLO heads** "
        "(strides 8/16/32). The architecture diagram from the course notes:"))
    c.append(code("show('lab_note/img/YOLOv4-Arch.png', width=820)"))
    c.append(md("**Mish** is the backbone activation — smooth and non-monotonic, unlike YOLOv3's Leaky-ReLU:"))
    c.append(code("show('results/mish_activation.png', width=560)"))

    c.append(md(
        "## 3. Parsing `yolov4.cfg` → PyTorch\n\n"
        "`parse_cfg` reads the Darknet blocks; `create_modules` builds an `nn.ModuleList`. The cell "
        "below confirms the YOLOv4-specific blocks are handled and that a forward pass yields the "
        "expected `(N, 22743, 85)` prediction tensor (76²+38²+19², ×3 anchors, ×85)."))
    c.append(code(
        "from detection.darknet import Darknet, parse_cfg\n"
        "blocks = parse_cfg('cfg/yolov4.cfg')\n"
        "counts = {t: sum(1 for b in blocks if b['type']==t) for t in ['convolutional','route','shortcut','maxpool','upsample','yolo']}\n"
        "n_mish = sum(1 for b in blocks if b.get('activation')=='mish')\n"
        "multi_route = [b['layers'] for b in blocks if b['type']=='route' and len(b['layers'].split(','))>2]\n"
        "print('block counts :', counts)\n"
        "print('mish convs   :', n_mish)\n"
        "print('SPP route (>2 layers):', multi_route)\n"
        "print('scale_x_y heads:', [b.get('scale_x_y') for b in blocks if b['type']=='yolo'])\n"
        "net = Darknet('cfg/yolov4.cfg').to(device).eval()\n"
        "with torch.no_grad():\n"
        "    y = net(torch.randn(1,3,608,608, device=device), device.type=='cuda')\n"
        "print('forward output:', tuple(y.shape))"))
    c.append(md("The 9 anchor priors (3 per scale), drawn at the 608 network scale:"))
    c.append(code("show('results/yolov4_anchors.png', width=520)"))

    c.append(md(
        "## 4. Pretrained inference at 608×608 (RGB)\n\n"
        "Loading the official `yolov4.weights` and running the canonical test image. Correct output "
        "is **dog / bicycle / truck** — confirming the parser, partial weight loader, `scale_x_y` "
        "decoding, RGB ordering and NMS are all correct."))
    c.append(code(
        "from detection.util import prep_image, write_results, COCO_NAMES\n"
        "import matplotlib.pyplot as plt\n"
        "net = Darknet('cfg/yolov4.cfg').to(device).eval(); net.net_info['height']='608'\n"
        "print('loaded', net.load_weights('weights/yolov4.weights'), 'conv layers')\n"
        "inp = prep_image('dog-cycle-car.png', 608).to(device)\n"
        "with torch.no_grad():\n"
        "    out = write_results(net(inp, device.type=='cuda'), 0.5, 80, 0.4)\n"
        "objs = sorted({COCO_NAMES[int(r[-1])] for r in out}) if hasattr(out,'shape') else []\n"
        "print('detections:', objs)"))
    c.append(md("YOLOv3 vs YOLOv4, side by side:"))
    c.append(code("show('results/det_inference_compare.png', width=900)"))
    c.append(md("Qualitative YOLOv4 detections on COCO val2017:"))
    c.append(code("show('results/det_coco_grid.png', width=900)"))

    c.append(md(
        "## 5. mAP@0.5 on COCO val2017\n\n"
        "Evaluated with `torchmetrics.MeanAveragePrecision` (NMS, ≤100 dets/img). The pretrained "
        "numbers are the correctness reference for the reimplementation."))
    c.append(code(
        "m = json.load(open('results/metrics.json')) if os.path.exists('results/metrics.json') else {}\n"
        "rows = [['YOLOv3 (pretrained, 416)', m.get('yolov3_pretrained_map50','—')],\n"
        "        ['YOLOv4 (pretrained, 608)', m.get('yolov4_pretrained_map50','—')]]\n"
        "display(pd.DataFrame(rows, columns=['model','mAP@0.5 (COCO val)']))"))
    c.append(code("show('results/det_per_class_ap.png', width=820)\nshow('results/det_pr_curve.png', width=520)"))

    c.append(md(
        "## 6. Training YOLOv4 on COCO — MSE/IoU vs CIoU\n\n"
        "**Exercise 2.** The CSPDarknet53 backbone is initialised from ImageNet (`yolov4.conv.137`, "
        "**2a**); `train_yolo()` (**2b**) applies augmentation, decodes anchor-relative outputs to "
        "boxes, and uses **MSE box + BCE objectness/class** (or **1−CIoU** box). Each anchor is "
        "matched to GT by IoU>0.3. Commands used:\n\n"
        "```bash\n"
        "python run.py --model yolov4 --dataset coco --epochs 10 --loss mse  --train\n"
        "python run.py --model yolov4 --dataset coco --epochs 10 --loss ciou --train\n"
        "```\n\n"
        "*Note:* trained from an ImageNet backbone on the 4k-image val split for a few epochs (vs "
        "118k images × ~300 epochs in the paper), so absolute mAP is modest **by design** — the "
        "point is that the loss **decreases** (the model learns) and CIoU localises better than MSE."))
    c.append(code(
        "hm, hc = load_history('yolov4_mse'), load_history('yolov4_ciou')\n"
        "if hm and hc:\n"
        "    df = pd.DataFrame({\n"
        "        'epoch': range(1, len(hm['loss'])+1),\n"
        "        'MSE total': np.round(hm['loss'],3), 'MSE box': np.round(hm['box'],3), 'MSE mAP@50': np.round(hm['map50'],4),\n"
        "        'CIoU total': np.round(hc['loss'],3), 'CIoU box': np.round(hc['box'],3), 'CIoU mAP@50': np.round(hc['map50'],4)})\n"
        "    display(df)\n"
        "    print(f\"loss decreased: MSE {hm['loss'][0]:.3f}→{hm['loss'][-1]:.3f} | CIoU {hc['loss'][0]:.3f}→{hc['loss'][-1]:.3f}\")\n"
        "else:\n"
        "    print('train histories not found yet — run the two training commands above')"))
    c.append(code("show('results/det_history_yolov4_mse.png', width=820)\nshow('results/det_history_yolov4_ciou.png', width=820)"))

    c.append(md("### MSE/IoU vs CIoU comparison (Exercise 2e)"))
    c.append(code("show('results/det_loss_comparison.png', width=950)"))
    c.append(code(
        "if hm and hc:\n"
        "    comp = pd.DataFrame({'Loss':['MSE / IoU loss','CIoU loss'],\n"
        "        'best mAP@0.5':[round(max(hm['map50']),4), round(max(hc['map50']),4)],\n"
        "        'final box loss':[round(hm['box'][-1],3), round(hc['box'][-1],3)],\n"
        "        'time/epoch (s)':[round(np.mean(hm['time']),1), round(np.mean(hc['time']),1)]})\n"
        "    display(comp)"))

    c.append(md(
        "## 7. Summary\n\n"
        "* The YOLOv3 parser was extended to YOLOv4 (Mish, maxpool/SPP, multi-layer & grouped "
        "routes, `scale_x_y`) and loads `yolov4.weights` → correct 608 RGB detections.\n"
        "* ImageNet backbone init + `train_yolo()` train on COCO with a balanced MSE/BCE (or CIoU) "
        "loss; the loss decreases each epoch and mAP is computed on the val split.\n"
        "* **CIoU** yields a lower box loss and higher mAP@0.5 than coordinate MSE — it optimises "
        "overlap, centre distance and aspect ratio jointly, which is exactly why YOLOv4 adopted it."))
    return c


# ──────────────────────────────────────────────────────────────────────────────
# A2-02  Image Segmentation
# ──────────────────────────────────────────────────────────────────────────────
def build_segmentation():
    c = []
    c.append(md(
        "# A2-02 · Image Segmentation — do skip connections matter?\n\n"
        "Two **U-Net** decoders share the *same* ImageNet-pretrained **ResNet-18** encoder; the only "
        "difference is whether each decoder stage concatenates the matching encoder feature map "
        "(**skip connections**). Training both on Oxford-IIIT Pet for 20 epochs isolates exactly "
        "what skip connections contribute."))
    c.append(code(SETUP))

    c.append(md(
        "## 1. Semantic vs instance segmentation\n\n"
        "| | Semantic | Instance |\n|---|---|---|\n"
        "| Question | class of each pixel | which *object* each pixel belongs to |\n"
        "| Same-class objects | merged | separated |\n"
        "| Typical model | **U-Net / FCN** | **Mask R-CNN** |\n\n"
        "Mask R-CNN = Faster R-CNN + a per-instance mask head (RoI-**Align**, 28×28 mask). Pretrained "
        "COCO Mask R-CNN on the test image:"))
    c.append(code("show('results/maskrcnn_demo.png', width=950)"))

    c.append(md(
        "## 2. U-Net + ResNet-18, and the skip ablation\n\n"
        "The ResNet-18 encoder produces feature maps at H/2, H/4, H/8, H/16, H/32; the decoder "
        "upsamples back to full resolution in 5 stages, each (optionally) concatenating the encoder "
        "map at the same resolution. The course U-Net diagram — grey arrows are the skip "
        "connections:"))
    c.append(code("show('lab_note/img/unet_arch.png', width=720)"))
    c.append(code(
        "from segmentation.models import build_segmentation_model\n"
        "for name in ['unet_resnet18','unet_resnet18_no_skip']:\n"
        "    mdl = build_segmentation_model(name, pretrained=False)\n"
        "    p = sum(x.numel() for x in mdl.parameters())\n"
        "    print(f'{name:24s} use_skip={mdl.use_skip!s:5s} params={p:,}')"))

    c.append(md("## 3. Dataset — Oxford-IIIT Pet (pet / background / border)"))
    c.append(code("show('results/seg_dataset_samples.png', width=560)"))

    c.append(md(
        "## 4. Training & results (20 epochs each)\n\n"
        "```bash\n"
        "python run.py --model unet_resnet18         --dataset oxford_pet --epochs 20 --train\n"
        "python run.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --train\n"
        "```"))
    c.append(code(
        "hs, hn = load_history('unet_resnet18'), load_history('unet_resnet18_no_skip')\n"
        "if hs and hn:\n"
        "    tbl = pd.DataFrame({\n"
        "        'Model':['U-Net + ResNet-18 (skip)','U-Net + ResNet-18 (no skip)'],\n"
        "        'Skip connections':['Yes','No'],\n"
        "        'Best Val mIoU':[round(max(hs['miou']),4), round(max(hn['miou']),4)],\n"
        "        'Time/epoch (s)':[round(np.mean(hs['time']),1), round(np.mean(hn['time']),1)]})\n"
        "    display(tbl)\n"
        "    print(f\"skip improves mIoU by {max(hs['miou'])-max(hn['miou']):+.4f}\")\n"
        "else:\n"
        "    print('histories not found yet — run the two training commands above')"))
    c.append(code("show('results/seg_curves.png', width=900)"))
    c.append(code("show('results/seg_per_class_iou.png', width=900)"))

    c.append(md("### Predictions — skip variant"))
    c.append(code("show('results/seg_pred_unet_resnet18.png', width=720)"))
    c.append(md("### Predictions — no-skip variant"))
    c.append(code("show('results/seg_pred_unet_resnet18_no_skip.png', width=720)"))
    c.append(md("### Side by side (same images) — note how no-skip blurs boundaries"))
    c.append(code("show('results/seg_skip_vs_noskip.png', width=820)"))

    c.append(md(
        "## 5. Exercise answers\n\n"
        "### (c) Why do skip connections help segmentation more than classification?\n\n"
        "Classification only needs *what* — a single global label — so the network deliberately "
        "discards spatial detail: pooling/striding build compact, translation-invariant semantics "
        "and the final answer doesn't depend on *where* an edge sat. Segmentation needs *what* **and** "
        "*exactly where* — a label per pixel — so the decoder must rebuild sharp boundaries at full "
        "resolution, which is impossible from the coarse bottleneck alone (a blurry thumbnail). Skip "
        "connections copy the encoder's high-resolution feature maps (precise edges/texture/location) "
        "straight into the decoder, fusing deep semantics with shallow geometry. For classification "
        "those shallow features barely affect the pooled prediction, so skips add little — they are "
        "far more valuable for dense prediction.\n\n"
        "### (d) Which skip level hurts most when removed — first (64ch, highest res) or last (512ch, lowest res)?\n\n"
        "The **first** skip (64-channel, H/2, highest resolution). It is the only path carrying "
        "full-resolution spatial detail into the final decoder stage; without it boundaries become "
        "blurry/blocky and thin structures (ear tips, legs, fur edges) are lost — precisely where pet "
        "IoU is decided. The last skip (512-channel, H/32) carries deep semantics that the adjacent "
        "bottleneck *already* encodes, so removing it is largely redundant. The highest-resolution "
        "skip supplies information available nowhere else in the decoder; the lowest-resolution one "
        "duplicates the bottleneck."))

    c.append(md(
        "## 6. Discussion\n\n"
        "Skip connections lift Val mIoU by the amount shown in the table above, and the side-by-side "
        "panels show the gain is concentrated at object **boundaries** (the Border/Pet classes) — the "
        "no-skip decoder must hallucinate fine edges from a coarse bottleneck and produces rounded, "
        "leaky masks. **Choose U-Net** when you need dense per-pixel labels for one or a few classes "
        "and don't need to separate instances — medical organ/tumour masks, road/sky 'stuff' — "
        "especially with limited data and a pretrained encoder. **Choose Mask R-CNN** when you must "
        "detect, separate and count individual object **instances** (this cat vs that cat) and also "
        "want boxes + labels, accepting its heavier two-stage cost."))
    return c


def main():
    for cells, path in [(build_detection(), "notebooks/A2-01-Object-Detection.ipynb"),
                        (build_segmentation(), "notebooks/A2-02-Image-Segmentation.ipynb")]:
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}
        nbf.write(nb, path)
        print("wrote", path, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()
