#!/usr/bin/env bash
set -euo pipefail

# End-to-end helper:
# 1) Build/refresh YOLO dataset (skips existing outputs for faster reruns).
# 2) Ensure data.yaml exists.
# 3) Train and validate.
# 4) Export to chosen formats (ONNX/engine/etc.).
# 5) Run quick predictions and throughput benchmark.

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
EXPORT_FORMATS=${EXPORT_FORMATS:-"onnx"}

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

run_predict() {
  local best="runs/detect/$NAME/weights/best.pt"
  local predict_src=${PREDICT_SOURCE:-"$OUT/images/val"}
  local predict_out=${PREDICT_OUT:-"runs/predict_sanpo"}
  local predict_name=${PREDICT_NAME:-"$NAME"}
  if [[ -f "$best" ]]; then
    echo "==> Running quick prediction on ${predict_src} ..."
    "$VENV_BIN/python" tools/predict_sanpo.py \
      --weights "$best" \
      --source "$predict_src" \
      --out "$predict_out" \
      --name "$predict_name" \
      --imgsz "$IMGSZ" \
      --device "$DEVICE" \
      --max "${PREDICT_MAX:-0}" \
      --conf "${PREDICT_CONF:-0.25}" || true
  else
    echo "!! Skipping predict: best checkpoint not found at $best"
  fi
}

export_formats() {
  local best="runs/detect/$NAME/weights/best.pt"
  if [[ ! -f "$best" ]]; then
    echo "!! Skipping export: best checkpoint not found at $best"
    return
  fi
  for fmt in ${EXPORT_FORMATS}; do
    echo "==> Exporting $fmt for $best"
    "$VENV_BIN/yolo" export \
      model="$best" \
      format="$fmt" \
      imgsz="$IMGSZ" \
      device="$DEVICE" || true
  done
}

benchmark_model() {
  local best="runs/detect/$NAME/weights/best.pt"
  local bench_src=${BENCHMARK_SOURCE:-"$OUT/images/val"}
  if [[ -f "$best" ]]; then
    echo "==> Benchmarking throughput on ${bench_src} ..."
    "$VENV_BIN/python" tools/benchmark_inference.py \
      --weights "$best" \
      --source "$bench_src" \
      --imgsz "$IMGSZ" \
      --device "$DEVICE" \
      --max "${BENCHMARK_MAX:-0}" \
      --conf "${BENCHMARK_CONF:-0.25}" \
      --repeats "${BENCHMARK_REPEATS:-1}" || true
  else
    echo "!! Skipping benchmark: best checkpoint not found at $best"
  fi
}

build_dataset
ensure_data_yaml
train_model
validate_best
run_predict
export_formats
benchmark_model
