"""Decoding, NMS and image utilities for the Darknet detector.

``predict_transform`` is written **functionally** (no in-place ops on tensors
that require grad) so the very same forward pass is used for inference *and* for
training the box-regression / objectness losses.
"""

from __future__ import division

import cv2
import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Feature map → boxes
# ──────────────────────────────────────────────────────────────────────────────
def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True, scale_x_y=1.0):
    """Decode one YOLO feature map into ``(B, H*W*num_anchors, 5+C)``.

    Output box format is centre ``(x, y, w, h)`` in **input-image pixels**;
    objectness and class scores are sigmoid probabilities. ``scale_x_y`` applies
    YOLOv4 grid-sensitivity: ``b_xy = scale·σ(t) − (scale−1)/2 + grid``.
    """
    batch_size = prediction.size(0)
    grid_size = prediction.size(2)
    stride = inp_dim // grid_size
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    dtype, device = prediction.dtype, prediction.device

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # grid cell offsets (x = col, y = row), matched to the row-major flatten above
    grid = torch.arange(grid_size, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(grid, grid, indexing="ij")
    x_offset = gx.reshape(-1, 1)
    y_offset = gy.reshape(-1, 1)
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    anchors_t = torch.as_tensor(anchors, dtype=dtype, device=device)          # pixels
    anchors_t = anchors_t.repeat(grid_size * grid_size, 1).unsqueeze(0)

    xy = (torch.sigmoid(prediction[..., 0:2]) * scale_x_y - (scale_x_y - 1) / 2 + x_y_offset) * stride
    wh = torch.exp(prediction[..., 2:4]) * anchors_t
    conf = torch.sigmoid(prediction[..., 4:5])
    cls = torch.sigmoid(prediction[..., 5:])

    return torch.cat((xy, wh, conf, cls), dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# IoU + NMS
# ──────────────────────────────────────────────────────────────────────────────
def bbox_iou(box1, box2):
    """IoU between ``box1`` (1×4) and each row of ``box2`` (N×4), corner format."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def unique(tensor):
    t_np = tensor.cpu().numpy()
    uniq = np.unique(t_np)
    out = torch.from_numpy(uniq)
    res = tensor.new(out.shape)
    res.copy_(out)
    return res


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    """Confidence-threshold + per-class NMS.

    Returns a ``(num_detections, 8)`` tensor:
    ``[image_idx, x1, y1, x2, y2, obj_conf, class_score, class_idx]`` — or the
    int ``0`` when nothing survives.
    """
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    # centre-xywh → corner xyxy
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)
    write = False
    output = 0

    for ind in range(batch_size):
        image_pred = prediction[ind]
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        image_pred = torch.cat((image_pred[:, :5], max_conf, max_conf_score), 1)

        non_zero_ind = torch.nonzero(image_pred[:, 4], as_tuple=False)
        if non_zero_ind.numel() == 0:
            continue
        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)

        img_classes = unique(image_pred_[:, -1])
        for cls in img_classes:
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2], as_tuple=False).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            image_pred_class = image_pred_class[conf_sort_index]

            idx = image_pred_class.size(0)
            for i in range(idx):
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except (ValueError, IndexError):
                    break
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask
                non_zero_ind = torch.nonzero(image_pred_class[:, 4], as_tuple=False).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = torch.cat((batch_ind, image_pred_class), 1)
            if not write:
                output = seq
                write = True
            else:
                output = torch.cat((output, seq))

    return output


# ──────────────────────────────────────────────────────────────────────────────
# Image preparation  (RGB, letterboxed to inp_dim — YOLOv4 wants RGB not BGR)
# ──────────────────────────────────────────────────────────────────────────────
def letterbox_image(img, inp_dim):
    """Resize image preserving aspect ratio, pad the rest with 128 (grey)."""
    img_h, img_w = img.shape[0], img.shape[1]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((h, w, 3), 128, dtype=np.uint8)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h,
           (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized
    return canvas


def prep_image(img, inp_dim):
    """OpenCV BGR image (or path) → letterboxed **RGB** CHW float tensor in [0,1]."""
    if isinstance(img, str):
        img = cv2.imread(img)
    img = letterbox_image(img, (inp_dim, inp_dim))
    img = img[:, :, ::-1]                       # BGR → RGB  (YOLOv4 requirement)
    img = img.transpose((2, 0, 1)).copy()       # HWC → CHW
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classes(namesfile):
    with open(namesfile, "r") as fp:
        names = fp.read().split("\n")
    return [n for n in names if n]


def get_test_input(img_path, inp_dim, CUDA=False):
    img = cv2.imread(img_path)
    img_ = prep_image(img, inp_dim)
    return img_.cuda() if CUDA else img_


# COCO 80-class names (matches the official ``coco.names`` ordering).
COCO_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
