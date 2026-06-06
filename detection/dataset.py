"""COCO-2017 dataset for YOLOv4 training.

* ``download_coco_val2017`` fetches the official val2017 images + annotations
  (direct HTTP — avoids the fiftyone/libcurl issues the lab hit on some setups).
* ``CocoYoloDataset`` performs YOLO label assignment: every GT box is matched to
  the anchor(s) with IoU > 0.3 on each scale (or the single best anchor overall),
  exactly as taught — but the target box is stored in **input-image pixels**
  (the lab stored ``box * stride``, which inflated the loss and kept mAP at 0).

Scales are emitted in **stride-ascending order [8, 16, 32]** to match the order
the YOLOv4 ``[yolo]`` heads appear in the forward pass.
"""

import json
import os
import urllib.request
import zipfile

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

from .losses import iou_xywh_numpy

# YOLOv4 anchors (from yolov4.cfg, pixel sizes at the 608 network scale),
# grouped per detection scale in the head's emission order.
ANCHORS_V4 = [
    [(12, 16), (19, 36), (40, 28)],        # stride 8   — small objects (76×76)
    [(36, 75), (76, 55), (72, 146)],       # stride 16  — medium objects (38×38)
    [(142, 110), (192, 243), (459, 401)],  # stride 32  — large objects (19×19)
]
STRIDES = [8, 16, 32]
NUM_ANCHORS_PER_SCALE = 3

_COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
_COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _download(url, dst):
    if os.path.exists(dst):
        print(f"  [coco] exists: {dst}")
        return
    print(f"  [coco] downloading {url} → {dst}")

    def _hook(block, block_size, total):
        done = block * block_size
        if total > 0:
            pct = min(100, done * 100 // total)
            print(f"\r    {pct:3d}%  ({done // 1_048_576} / {total // 1_048_576} MB)", end="")

    urllib.request.urlretrieve(url, dst, _hook)
    print()


def download_coco_val2017(data_dir="data/coco"):
    """Download + extract COCO val2017. Returns ``(images_dir, ann_file)``."""
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, "val2017")
    ann_file = os.path.join(data_dir, "annotations", "instances_val2017.json")

    if not os.path.isdir(images_dir):
        zip_path = os.path.join(data_dir, "val2017.zip")
        _download(_COCO_IMAGES_URL, zip_path)
        print("  [coco] extracting images …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)

    if not os.path.exists(ann_file):
        zip_path = os.path.join(data_dir, "annotations_trainval2017.zip")
        _download(_COCO_ANN_URL, zip_path)
        print("  [coco] extracting annotations …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)

    return images_dir, ann_file


def build_transforms(img_size, train=True):
    """Albumentations pipeline (COCO bbox format). Basic augmentation when training."""
    ops = [A.Resize(img_size, img_size)]
    if train:
        ops += [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2)]
    return A.Compose(ops, bbox_params=A.BboxParams(
        format="coco", label_fields=["category_ids"], min_visibility=0.0))


class CocoYoloDataset(Dataset):
    def __init__(self, images_dir, ann_file, img_size=608, train=True,
                 anchors=ANCHORS_V4, strides=STRIDES, num_classes=80):
        from pycocotools.coco import COCO

        self.images_dir = images_dir
        self.img_size = img_size
        self.anchors = anchors
        self.strides = strides
        self.num_classes = num_classes
        self.transform = build_transforms(img_size, train)

        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats = sorted(cats, key=lambda c: c["id"])
        self.cat2idx = {c["id"]: i for i, c in enumerate(cats)}      # COCO id → 0..79

    def __len__(self):
        return len(self.ids)

    def set_length(self, n):
        """Restrict to the first ``n`` images (keeps a deterministic split)."""
        self.ids = self.ids[:n]
        return self

    def __getitem__(self, index):
        import cv2

        img_id = self.ids[index]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        path = self.coco.loadImgs(img_id)[0]["file_name"]

        img = cv2.imread(os.path.join(self.images_dir, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        bboxes, cat_ids = [], []
        for obj in anns:
            x, y, bw, bh = obj["bbox"]
            if bw <= 1 or bh <= 1:
                continue
            # clip to image bounds so albumentations never rejects a box
            x = max(0.0, min(x, w - 1)); y = max(0.0, min(y, h - 1))
            bw = min(bw, w - x); bh = min(bh, h - y)
            if bw <= 1 or bh <= 1:
                continue
            bboxes.append([x, y, bw, bh])
            cat_ids.append(obj["category_id"])

        try:
            t = self.transform(image=img, bboxes=bboxes, category_ids=cat_ids)
            img_t = t["image"]
            boxes = np.array(t["bboxes"], dtype=np.float32).reshape(-1, 4)
            classes = np.array(t["category_ids"], dtype=np.int64).reshape(-1)
        except Exception:
            img_t = A.Resize(self.img_size, self.img_size)(image=img)["image"]
            boxes = np.zeros((0, 4), np.float32)
            classes = np.zeros((0,), np.int64)

        img_t = torch.from_numpy(img_t.transpose(2, 0, 1).copy()).float().div(255.0)
        label = self._create_label(boxes, classes)
        return img_t, label

    def _create_label(self, bboxes, class_inds):
        """Assign GT boxes to anchors → flat label tensor ``(N_total, 5+C)``."""
        strides = np.array(self.strides)
        out_sizes = (self.img_size / strides).astype(int)
        C = self.num_classes
        A_ = NUM_ANCHORS_PER_SCALE

        label = [np.zeros((out_sizes[i], out_sizes[i], A_, 5 + C), np.float32) for i in range(3)]

        for b in range(len(bboxes)):
            x, y, bw, bh = bboxes[b]
            if bw <= 0 or bh <= 0:
                continue
            cls_idx = self.cat2idx.get(int(class_inds[b]))
            if cls_idx is None:
                continue
            one_hot = np.zeros(C, np.float32)
            one_hot[cls_idx] = 1.0

            # COCO xywh (top-left) → centre xywh, in pixels
            bbox_xywh = np.array([x + bw / 2, y + bh / 2, bw, bh], np.float32)
            bbox_scaled = bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            ious, exist_positive = [], False
            for i in range(3):
                anchors_xywh = np.zeros((A_, 4), np.float32)
                anchors_xywh[:, 0:2] = np.floor(bbox_scaled[i, 0:2]) + 0.5
                anchors_xywh[:, 2:4] = np.array(self.anchors[i]) / strides[i]
                iou_scale = iou_xywh_numpy(bbox_scaled[i][np.newaxis, :], anchors_xywh)
                ious.append(iou_scale)
                mask = iou_scale > 0.3
                if np.any(mask):
                    xi, yi = np.floor(bbox_scaled[i, 0:2]).astype(int)
                    xi = min(xi, out_sizes[i] - 1); yi = min(yi, out_sizes[i] - 1)
                    label[i][yi, xi, mask, 0:4] = bbox_xywh   # pixels (FIX vs lab)
                    label[i][yi, xi, mask, 4:5] = 1.0
                    label[i][yi, xi, mask, 5:] = one_hot
                    exist_positive = True

            if not exist_positive:
                best = int(np.argmax(np.array(ious).reshape(-1)))
                s, a = best // A_, best % A_
                xi, yi = np.floor(bbox_scaled[s, 0:2]).astype(int)
                xi = min(xi, out_sizes[s] - 1); yi = min(yi, out_sizes[s] - 1)
                label[s][yi, xi, a, 0:4] = bbox_xywh
                label[s][yi, xi, a, 4:5] = 1.0
                label[s][yi, xi, a, 5:] = one_hot

        flat = [torch.from_numpy(label[i].reshape(-1, 5 + C)) for i in range(3)]
        return torch.cat(flat, 0)   # stride order [8, 16, 32] → matches YOLOv4 heads
