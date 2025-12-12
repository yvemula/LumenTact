#!/usr/bin/env bash
set -euo pipefail

# Simple end-to-end helper:
# 1) Build/refresh YOLO dataset (skips existing outputs for faster reruns).
# 2) Train with Ultralytics YOLO.
# 3) Validate best checkpoint and export ONNX.

SRC=${SRC:-sanpo_subset_auto}
OUT=${OUT:-dataset_yolo_train}
MODEL=${MODEL:-yolo11n.pt}
NAME=${NAME:-sanpo_yolo}
EPOCHS=${EPOCHS:-50}
IMGSZ=${IMGSZ:-1280}
BATCH=${BATCH:-16}
DEVICE=${DEVICE:-0}
VAL_RATIO=${VAL_RATIO:-0.2}
VENV_BIN=${VENV_BIN:-./venv/bin}

build_dataset() {
  echo "==> Building dataset at '$OUT' from '$SRC' (skip existing outputs)..."
  "$VENV_BIN/python" make_yolo_dataset.py \
    --src "$SRC" \
    --out "$OUT" \
    --val-ratio "$VAL_RATIO" \
    --write-empty-labels \
    --symlink \
    --seed 0 \
    --skip-existing
}

ensure_data_yaml() {
  echo "==> Ensuring data.yaml exists for '$OUT'..."
  "$VENV_BIN/python" tools/ensure_data_yaml.py --data "$OUT" || true
}

train_model() {
  echo "==> Training model '$MODEL' -> run name '$NAME'..."
  "$VENV_BIN/yolo" train \
    model="$MODEL" \
    data="$OUT/data.yaml" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    device="$DEVICE" \
    name="$NAME"
}

validate_best() {
  local best="runs/detect/$NAME/weights/best.pt"
  if [[ -f "$best" ]]; then
    echo "==> Validating best checkpoint: $best"
    "$VENV_BIN/yolo" val \
      model="$best" \
      data="$OUT/data.yaml" \
      batch="$BATCH" \
      device="$DEVICE" \
      name="${NAME}_val" || true
  else
    echo "!! Skipping val: best checkpoint not found at $best"
  fi
}

export_onnx() {
  local best="runs/detect/$NAME/weights/best.pt"
  if [[ -f "$best" ]]; then
    echo "==> Exporting ONNX for $best"
    "$VENV_BIN/yolo" export \
      model="$best" \
      format=onnx \
      imgsz="$IMGSZ" \
      device="$DEVICE" || true
  else
    echo "!! Skipping export: best checkpoint not found at $best"
  fi
}

build_dataset
ensure_data_yaml
train_model
validate_best
export_onnx
