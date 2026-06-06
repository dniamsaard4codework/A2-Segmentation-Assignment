# ──────────────────────────────────────────────────────────────────────────────
# A2 — one-shot reproducible pipeline (Windows PowerShell).
#
#   ./script.ps1                  # full pipeline (segmentation + detection + notebooks)
#   ./script.ps1 -SkipDetTrain    # everything except the long YOLOv4 training
#   ./script.ps1 -DetEpochs 12 -SegEpochs 20
#
# Idempotent: existing weights / datasets / checkpoints are reused, not redone.
# ──────────────────────────────────────────────────────────────────────────────
param(
  [int]$DetEpochs = 10,
  [int]$SegEpochs = 20,
  [int]$ImgSize   = 608,
  [int]$Workers   = 8,
  [switch]$SkipDetTrain
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Dl($url, $dst) {
  if (Test-Path $dst) { Write-Host "  exists: $dst" }
  else { Write-Host "  downloading $dst"; curl.exe -L --retry 3 -s -o $dst $url }
}

Write-Host "==> 1/6  Environment (uv sync + GPU check)"
uv sync
uv run python -c "import torch;assert torch.cuda.is_available();print('GPU:',torch.cuda.get_device_name(0),'| capability',torch.cuda.get_device_capability(0))"

Write-Host "==> 2/6  Assets (cfg, weights, test image)"
New-Item -ItemType Directory -Force -Path cfg, weights, results, notebooks/img | Out-Null
Dl "https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/cfg/yolov3.cfg" "cfg/yolov3.cfg"
Dl "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg" "cfg/yolov4.cfg"
Dl "https://github.com/ayooshkathuria/pytorch-yolo-v3/raw/master/dog-cycle-car.png" "dog-cycle-car.png"
Dl "https://pjreddie.com/media/files/yolov3.weights" "weights/yolov3.weights"
Dl "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights" "weights/yolov4.weights"
Dl "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137" "weights/yolov4.conv.137"

Write-Host "==> 3/6  Segmentation - skip vs no-skip ($SegEpochs epochs each)"
uv run python run.py --model unet_resnet18         --dataset oxford_pet --epochs $SegEpochs --train
uv run python run.py --model unet_resnet18_no_skip --dataset oxford_pet --epochs $SegEpochs --train

Write-Host "==> 4/6  Detection - pretrained inference + mAP"
uv run python run.py --model yolov3 --weights weights/yolov3.weights --image dog-cycle-car.png --infer
uv run python run.py --model yolov4 --weights weights/yolov4.weights --image dog-cycle-car.png --infer
uv run python run.py --model yolov3 --weights weights/yolov3.weights --dataset coco --evaluate --img-size 416 --limit 1000
uv run python run.py --model yolov4 --weights weights/yolov4.weights --dataset coco --evaluate --img-size 608 --limit 1000

if (-not $SkipDetTrain) {
  Write-Host "==> 5/6  Detection - train YOLOv4 (MSE vs CIoU, $DetEpochs epochs, ${ImgSize}px)"
  uv run python run.py --model yolov4 --dataset coco --epochs $DetEpochs --loss mse  --train --img-size $ImgSize --num-workers $Workers
  uv run python run.py --model yolov4 --dataset coco --epochs $DetEpochs --loss ciou --train --img-size $ImgSize --num-workers $Workers
} else {
  Write-Host "==> 5/6  Detection training SKIPPED (-SkipDetTrain)"
}

Write-Host "==> 6/6  Figures + execute notebooks"
uv run python make_figures.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/A2-02-Image-Segmentation.ipynb
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/A2-01-Object-Detection.ipynb
Write-Host "Done."
