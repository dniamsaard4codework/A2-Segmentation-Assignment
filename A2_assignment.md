# Exercises A2-01-Object-Detection

## Exercise 1

1. In the lab, we saw how the YOLOv3 Darknet configuration file could be parsed and mapped to PyTorch modules. Do the same for **YOLOv4**. Download `yolov4.cfg` from the [YOLOv4 GitHub repository](https://github.com/AlexeyAB/darknet) and modify `MyDarknet`, `darknet.py`, and `util.py` as necessary.

   Changes required:
   - Implement the **Mish** activation function
   - Add support for `maxpool` in `create_modules()` and `forward()`
   - Enable `[route]` to concatenate more than two previous layers
   - Load pretrained weights from the [YOLOv4 release](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
   - Scale inputs to 608×608 in RGB order (not BGR)

2. Train the YOLOv4 model on the COCO dataset (or another dataset of your choice).

   a) Load ImageNet pretrained weights for CSPDarknet53 and initialize the backbone.

   b) Implement a `train_yolo()` function that:
      - Applies basic data augmentation
      - Converts anchor-relative outputs to bounding box coordinates
      - Computes loss (MSE for bbox + BCE for objectness/class)
      - Uses IoU thresholds to filter predictions

   c) Train for at least 5 epochs and verify the model is learning (loss decreasing).

   d) Compute **mAP** on the COCO validation set.

   e) Replace the standard IoU loss with **CIoU loss** and compare mAP:

   | Loss | mAP |
   |---|---|
   | MSE / IoU loss | ? |
   | CIoU loss | ? |

3. **Why is YOLO v3 faster than Faster R-CNN?** Explain the architectural difference that enables single-shot detection.

---

## Submission

Submit your work to GitHub. Your repository should contain:

### 1. Training Script (`run.py`)

```bash
# Inference with pretrained weights
python3 run.py --model yolov3 --weights yolov3.weights --image dog-cycle-car.png --infer

# Train on COCO
python3 run.py --model yolov4 --dataset coco --epochs 5 --train

# Evaluate mAP
python3 run.py --model yolov4 --weights yolov4.weights --dataset coco --evaluate
```

### 2. `README.md`

Your `README.md` must include:

**Commands used** (exact commands you ran)

**Results table:**

| Model | Dataset | mAP | Time/epoch | Notes |
|---|---|---|---|---|
| YOLOv3 (pretrained) | COCO | ? | — | inference only |
| YOLOv4 (IoU loss) | COCO | ? | ? | trained from scratch |
| YOLOv4 (CIoU loss) | COCO | ? | ? | loss comparison |

**Discussion** (3–5 sentences): What was the effect of CIoU vs standard loss? What challenges did you encounter training on COCO?

# Exercises A2-02-Image-Segmentation

## Exercise 1

1. **Skip Connections — Do They Matter?**

   Both models use a **pretrained ResNet-18 encoder** (ImageNet weights). The only difference is whether the decoder uses skip connections or not.

   a) Train **two variants** using the `run.py` script:

   ```bash
   # With skip connections (baseline)
   python3 run.py --model unet_resnet18         --dataset oxford_pet --epochs 20 --train

   # Without skip connections (ablation — same ResNet-18 encoder, skip connections removed)
   python3 run.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --train
   ```

   b) Compare mIoU:

   | Model | Encoder | Skip connections | Val mIoU | Time/epoch |
   |---|---|---|---|---|
   | U-Net + ResNet-18 | ResNet-18 (ImageNet) | ✅ Yes | ? | ? |
   | U-Net + ResNet-18 (no skip) | ResNet-18 (ImageNet) | ❌ No | ? | ? |

   c) Why do skip connections help segmentation more than they would help classification?

   d) Which skip connection level do you think hurts the most when removed — the first (64ch, highest resolution) or the last (512ch, lowest resolution)? Why?
   
---

## Submission

Submit your work to GitHub. Your repository should contain:

### 1. Training Script (`run.py`)

There are **2 models** to train — same ResNet-18 encoder, skip connections differ:

| Model flag | Encoder | Skip connections | Pretrained |
|---|---|---|---|
| `unet_resnet18` | ResNet-18 | ✅ Yes | ✅ ImageNet |
| `unet_resnet18_no_skip` | ResNet-18 | ❌ No | ✅ ImageNet |

```bash
# 1. Baseline — ResNet-18 encoder + skip connections
python3 run.py --model unet_resnet18         --dataset oxford_pet --epochs 20 --train

# 2. Ablation — same ResNet-18 encoder, skip connections REMOVED
python3 run.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs 20 --train

# Evaluate saved model
python3 run.py --model unet_resnet18 --weights unet_resnet18_pet.pt --dataset oxford_pet --evaluate
```

### 2. `README.md`

Your `README.md` must include:

**Commands used** (exact commands you ran)

**Results table:**

| Model | Encoder | Skip connections | Val mIoU | Time/epoch |
|---|---|---|---|---|
| `unet_resnet18` | ResNet-18 (ImageNet) | ✅ | ? | ? |
| `unet_resnet18_no_skip` | ResNet-18 (ImageNet) | ❌ | ? | ? |

**Discussion** (3–5 sentences): How much did skip connections improve mIoU? When would you choose U-Net over Mask R-CNN?
