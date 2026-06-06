"""Darknet config parser and PyTorch model — YOLOv3 **and** YOLOv4.

This extends the classic YOLOv3 "from scratch" parser (Kathuria) so that the
exact same code parses ``yolov4.cfg``. The YOLOv4-specific additions are:

* **Mish** activation (``activation=mish``) used throughout CSPDarknet53.
* **``[maxpool]``** layers — required for the SPP block (sizes 5/9/13, stride 1).
* **``[route]`` with more than two layers** — SPP concatenates four maps.
* **``[route]`` with ``groups``/``group_id``** — the CSP "channel split" that
  feeds half of a feature map down the dense branch.
* **``scale_x_y``** grid-sensitivity on the ``[yolo]`` layers.
* **Partial weight loading** so the ImageNet-pretrained backbone file
  ``yolov4.conv.137`` initialises the first 137 conv layers (Exercise 2a).

The decoding of raw feature maps into boxes lives in :mod:`detection.util`
(``predict_transform``); this module only builds the graph and runs the forward
pass / weight loading.
"""

from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Activations / placeholder layers
# ──────────────────────────────────────────────────────────────────────────────
class Mish(nn.Module):
    """Mish activation: ``x * tanh(softplus(x))`` (Misra, 2019).

    Smooth, non-monotonic, and self-regularising — YOLOv4 uses it across the
    CSPDarknet53 backbone in place of Leaky-ReLU. ``softplus`` is computed in a
    numerically stable way by ``F.softplus``.
    """

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class EmptyLayer(nn.Module):
    """Placeholder for ``route`` and ``shortcut`` — the real op happens in
    :meth:`Darknet.forward` where we have access to cached layer outputs."""

    def __init__(self):
        super().__init__()


class DetectionLayer(nn.Module):
    """Holds the anchors (and ``scale_x_y``) for one ``[yolo]`` head."""

    def __init__(self, anchors, scale_x_y=1.0):
        super().__init__()
        self.anchors = anchors
        self.scale_x_y = scale_x_y


# ──────────────────────────────────────────────────────────────────────────────
# Config parsing
# ──────────────────────────────────────────────────────────────────────────────
def parse_cfg(cfgfile):
    """Parse a Darknet ``.cfg`` into a list of block dicts.

    The first block is the ``[net]`` hyper-parameter section; each subsequent
    block is one layer (``convolutional``, ``route``, ``shortcut``, ``upsample``,
    ``maxpool`` or ``yolo``).
    """
    with open(cfgfile, "r") as fp:
        lines = [ln.strip() for ln in fp.read().split("\n")]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]

    block = {}
    blocks = []
    for line in lines:
        if line.startswith("["):
            if block:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].strip()
        else:
            key, value = line.split("=", 1)
            block[key.strip()] = value.strip()
    blocks.append(block)
    return blocks


def create_modules(blocks):
    """Translate parsed blocks into an ``nn.ModuleList``.

    Returns ``(net_info, module_list)``. ``output_filters`` tracks the channel
    count after every layer so that ``route``/``shortcut`` channel bookkeeping is
    correct (including CSP ``groups`` splits).
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        btype = x["type"]

        # ── convolutional ────────────────────────────────────────────────────
        if btype == "convolutional":
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except KeyError:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            pad = (kernel_size - 1) // 2 if int(x["pad"]) else 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f"conv_{index}", conv)

            if batch_normalize:
                module.add_module(f"batch_norm_{index}", nn.BatchNorm2d(filters))

            if activation == "leaky":
                module.add_module(f"leaky_{index}", nn.LeakyReLU(0.1, inplace=True))
            elif activation == "mish":
                module.add_module(f"mish_{index}", Mish())
            # "linear" → no activation

        # ── upsample ─────────────────────────────────────────────────────────
        elif btype == "upsample":
            stride = int(x["stride"])
            module.add_module(f"upsample_{index}", nn.Upsample(scale_factor=stride, mode="nearest"))

        # ── maxpool (SPP / tiny) ─────────────────────────────────────────────
        elif btype == "maxpool":
            size = int(x["size"])
            stride = int(x["stride"])
            pad = (size - 1) // 2  # "same" padding keeps spatial size when stride==1
            module.add_module(f"maxpool_{index}", nn.MaxPool2d(size, stride, padding=pad))

        # ── route (concat ≥1 layers, optional CSP channel split) ─────────────
        elif btype == "route":
            layers = [int(a) for a in x["layers"].split(",")]
            # normalise positive (absolute) indices to relative-from-here
            layers = [(l - index) if l > 0 else l for l in layers]
            x["layers"] = ",".join(str(l) for l in layers)  # store normalised
            filters = sum(output_filters[index + l] for l in layers)
            groups = int(x.get("groups", 1))
            if groups > 1:
                filters = filters // groups
            module.add_module(f"route_{index}", EmptyLayer())

        # ── shortcut (residual add) ──────────────────────────────────────────
        elif btype == "shortcut":
            filters = output_filters[index - 1]
            module.add_module(f"shortcut_{index}", EmptyLayer())

        # ── yolo detection head ──────────────────────────────────────────────
        elif btype == "yolo":
            mask = [int(m) for m in x["mask"].split(",")]
            anchors = [int(a) for a in x["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            scale_x_y = float(x.get("scale_x_y", 1.0))
            module.add_module(f"Detection_{index}", DetectionLayer(anchors, scale_x_y))

        else:
            raise ValueError(f"Unknown layer type: {btype!r}")

        module_list.append(module)
        prev_filters = filters if btype not in ("yolo",) else prev_filters
        output_filters.append(prev_filters)

    return net_info, module_list


# ──────────────────────────────────────────────────────────────────────────────
# The model
# ──────────────────────────────────────────────────────────────────────────────
class Darknet(nn.Module):
    """A Darknet model built from a ``.cfg`` file (YOLOv3 or YOLOv4)."""

    def __init__(self, cfgfile):
        super().__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
        # default input size from cfg (608 for YOLOv4)
        self.net_info.setdefault("height", "608")

    # -- forward -------------------------------------------------------------
    def forward(self, x, CUDA=None):
        from .util import predict_transform

        if CUDA is None:
            CUDA = x.is_cuda

        modules = self.blocks[1:]
        outputs = {}            # cache for route / shortcut
        detections = None

        for i, module in enumerate(modules):
            mtype = module["type"]

            if mtype in ("convolutional", "upsample", "maxpool"):
                x = self.module_list[i](x)

            elif mtype == "route":
                layers = [int(a) for a in module["layers"].split(",")]
                maps = [outputs[i + l] for l in layers]
                x = maps[0] if len(maps) == 1 else torch.cat(maps, 1)
                groups = int(module.get("groups", 1))
                if groups > 1:
                    group_id = int(module["group_id"])
                    ch = x.shape[1] // groups
                    x = x[:, group_id * ch:(group_id + 1) * ch].contiguous()

            elif mtype == "shortcut":
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]

            elif mtype == "yolo":
                det_layer = self.module_list[i][0]
                anchors = det_layer.anchors
                scale_x_y = det_layer.scale_x_y
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA, scale_x_y)
                detections = x if detections is None else torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    # -- pretrained weights --------------------------------------------------
    def load_weights(self, weightfile):
        """Load Darknet binary weights.

        Works for full detector weights (``yolov3.weights`` / ``yolov4.weights``)
        and for the **partial** ImageNet backbone (``yolov4.conv.137``): loading
        stops cleanly once the file is exhausted and the number of initialised
        conv layers is reported.
        """
        with open(weightfile, "rb") as fp:
            header = np.fromfile(fp, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        loaded = 0
        for i in range(len(self.module_list)):
            if self.blocks[i + 1]["type"] != "convolutional":
                continue

            model = self.module_list[i]
            try:
                batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
            except KeyError:
                batch_normalize = 0

            conv = model[0]

            # Stop if the (possibly partial) file can't fill this layer.
            need = conv.weight.numel()
            if batch_normalize:
                need += 4 * model[1].bias.numel()
            else:
                need += conv.bias.numel()
            if ptr + need > weights.size:
                print(f"  [load_weights] partial file: initialised {loaded} conv layers, "
                      f"stopped at layer index {i}")
                break

            if batch_normalize:
                bn = model[1]
                n = bn.bias.numel()
                bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + n]).view_as(bn.bias.data)); ptr += n
                bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + n]).view_as(bn.weight.data)); ptr += n
                bn.running_mean.copy_(torch.from_numpy(weights[ptr:ptr + n]).view_as(bn.running_mean)); ptr += n
                bn.running_var.copy_(torch.from_numpy(weights[ptr:ptr + n]).view_as(bn.running_var)); ptr += n
            else:
                n = conv.bias.numel()
                conv.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + n]).view_as(conv.bias.data)); ptr += n

            n = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + n]).view_as(conv.weight.data)); ptr += n
            loaded += 1

        return loaded


# The lab notebook refers to the class as ``MyDarknet`` — keep that name working.
MyDarknet = Darknet
