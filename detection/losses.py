"""YOLO training losses and IoU helpers.

Two box-regression variants share one objectness/class BCE term so the
**MSE/IoU vs CIoU** comparison (Exercise 2e) is a clean A/B swap:

* ``loss_type="mse"``  → MSE on normalised centre ``(x,y,w,h)`` (YOLOv3 paper).
* ``loss_type="ciou"`` → ``1 − CIoU`` (Zheng et al., 2020), the YOLOv4 default —
  jointly optimises overlap, centre distance and aspect ratio.
"""

import math

import numpy as np
import torch
import torch.nn as nn


def iou_xywh_numpy(boxes1, boxes2):
    """IoU between centre-format ``(x,y,w,h)`` boxes (numpy) — anchor matching."""
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter = np.maximum(right_down - left_up, 0.0)
    inter_area = inter[..., 0] * inter[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return inter_area / (union_area + 1e-16)


def ciou_xywh_torch(boxes1, boxes2):
    """CIoU for centre-format ``(x,y,w,h)`` torch boxes (returns one value/box)."""
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)
    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    inter_lu = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter_rd = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter = torch.clamp(inter_rd - inter_lu, min=0.0)
    inter_area = inter[..., 0] * inter[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = inter_area / (union_area + 1e-16)

    outer_lu = torch.min(boxes1[..., :2], boxes2[..., :2])
    outer_rd = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    outer = torch.clamp(outer_rd - outer_lu, min=0.0)
    outer_diag = outer[..., 0] ** 2 + outer[..., 1] ** 2 + 1e-16

    c1 = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
    c2 = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
    center_dis = (c1[..., 0] - c2[..., 0]) ** 2 + (c1[..., 1] - c2[..., 1]) ** 2

    s1 = torch.clamp(boxes1[..., 2:] - boxes1[..., :2], min=0.0)
    s2 = torch.clamp(boxes2[..., 2:] - boxes2[..., :2], min=0.0)
    v = (4 / math.pi ** 2) * torch.pow(
        torch.atan(s1[..., 0] / torch.clamp(s1[..., 1], min=1e-6)) -
        torch.atan(s2[..., 0] / torch.clamp(s2[..., 1], min=1e-6)), 2)
    alpha = v / torch.clamp(1 - ious + v, min=1e-6)

    return ious - (center_dis / outer_diag + alpha * v)


def compute_yolo_loss(outputs, labels, img_size, loss_type="mse",
                      lambda_coord=5.0, lambda_noobj=0.5):
    """Total YOLO loss = box + objectness(BCE) + class(BCE).

    ``outputs`` are the **decoded** predictions from ``predict_transform``
    (centre ``(x,y,w,h)`` in pixels; objectness/class already sigmoid-activated).
    ``labels`` use the same anchor layout from the dataset (xywh pixels, obj 0/1,
    one-hot class).

    Each term is **mean-normalised over the cells that contribute to it** —
    box/class over the handful of positive (object) anchors, and objectness over
    positives and negatives separately. Without this, the ~22k background anchors
    swamp the loss (the lab's sum-over-batch left conf ≈ 7700 and mAP at 0).
    Returns ``(total, box, conf, cls)``.
    """
    pred_xywh = outputs[..., 0:4] / img_size
    pred_conf = outputs[..., 4:5].clamp(1e-6, 1 - 1e-6)
    pred_cls = outputs[..., 5:].clamp(1e-6, 1 - 1e-6)

    label_xywh = labels[..., 0:4] / img_size
    obj = labels[..., 4:5].clamp(0, 1)
    noobj = 1.0 - obj
    label_cls = labels[..., 5:].clamp(0, 1)
    C = pred_cls.shape[-1]

    n_obj = obj.sum().clamp(min=1.0)
    n_noobj = noobj.sum().clamp(min=1.0)
    bce = nn.BCELoss(reduction="none")

    if loss_type == "ciou":
        ciou = ciou_xywh_torch(pred_xywh, label_xywh).unsqueeze(-1)        # (B,N,1)
        box_term = 1.0 - ciou
    else:  # "mse" / "iou"
        box_term = ((pred_xywh - label_xywh) ** 2).sum(-1, keepdim=True)   # (B,N,1)
    loss_box = lambda_coord * (obj * box_term).sum() / n_obj

    bce_conf = bce(pred_conf, obj)
    loss_conf = (obj * bce_conf).sum() / n_obj + lambda_noobj * (noobj * bce_conf).sum() / n_noobj
    loss_cls = (obj * bce(pred_cls, label_cls)).sum() / (n_obj * C)

    total = loss_box + loss_conf + loss_cls
    return total, loss_box, loss_conf, loss_cls
