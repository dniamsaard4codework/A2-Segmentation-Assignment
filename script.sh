#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# A2 — one-shot reproducible pipeline (Linux / macOS).
#
#   ./script.sh                 # full pipeline (segmentation + detection + notebooks)
#   SKIP_DET_TRAIN=1 ./script.sh   # everything except the long YOLOv4 training
#   DET_EPOCHS=12 SEG_EPOCHS=20 ./script.sh
#
# Idempotent: existing weights / datasets / checkpoints are reused, not redone.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"

PY="uv run python"
DET_EPOCHS="${DET_EPOCHS:-10}"
SEG_EPOCHS="${SEG_EPOCHS:-20}"
IMG_SIZE="${IMG_SIZE:-608}"
WORKERS="${WORKERS:-8}"
SKIP_DET_TRAIN="${SKIP_DET_TRAIN:-0}"

dl() { [ -f "$2" ] && echo "  exists: $2" || { echo "  downloading $2"; curl -L --retry 3 -s -o "$2" "$1"; }; }

echo "==> 1/6  Environment (uv sync + GPU check)"
uv sync
$PY -c "import torch;assert torch.cuda.is_available();print('GPU:',torch.cuda.get_device_name(0),'| capability',torch.cuda.get_device_capability(0))"

echo "==> 2/6  Assets (cfg, weights, test image)"
mkdir -p cfg weights results notebooks/img
dl https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/cfg/yolov3.cfg cfg/yolov3.cfg
dl https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg cfg/yolov4.cfg
dl https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png dog-cycle-car.png
dl https://pjreddie.com/media/files/yolov3.weights weights/yolov3.weights
dl https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights weights/yolov4.weights
dl https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137 weights/yolov4.conv.137

echo "==> 3/6  Segmentation — skip vs no-skip ($SEG_EPOCHS epochs each)"
$PY run.py --model unet_resnet18         --dataset oxford_pet --epochs "$SEG_EPOCHS" --train
$PY run.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs "$SEG_EPOCHS" --train

echo "==> 4/6  Detection — pretrained inference + mAP"
$PY run.py --model yolov3 --weights weights/yolov3.weights --image dog-cycle-car.png --infer
$PY run.py --model yolov4 --weights weights/yolov4.weights --image dog-cycle-car.png --infer
$PY run.py --model yolov3 --weights weights/yolov3.weights --dataset coco --evaluate --img-size 416 --limit 1000
$PY run.py --model yolov4 --weights weights/yolov4.weights --dataset coco --evaluate --img-size 608 --limit 1000

if [ "$SKIP_DET_TRAIN" != "1" ]; then
  echo "==> 5/6  Detection — train YOLOv4 (MSE vs CIoU, $DET_EPOCHS epochs, ${IMG_SIZE}px)"
  $PY run.py --model yolov4 --dataset coco --epochs "$DET_EPOCHS" --loss mse  --train --img-size "$IMG_SIZE" --num-workers "$WORKERS"
  $PY run.py --model yolov4 --dataset coco --epochs "$DET_EPOCHS" --loss ciou --train --img-size "$IMG_SIZE" --num-workers "$WORKERS"
else
  echo "==> 5/6  Detection training SKIPPED (SKIP_DET_TRAIN=1)"
fi

echo "==> 6/6  Figures + execute notebooks"
$PY make_figures.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/A2-02-Image-Segmentation.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/A2-01-Object-Detection.ipynb
echo "Done."
