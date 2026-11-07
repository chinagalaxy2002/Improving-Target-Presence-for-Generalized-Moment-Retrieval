#!/usr/bin/env bash
set -euo pipefail

# Argument check
if [ $# -lt 4 ]; then
  echo "Usage: $0 <gpu_physical_id> <backbone> <variant> <revision_dir>"
  exit 1
fi

GPU_PHYSICAL_ID="$1"
BACKBONE="$2"
VARIANT="$3"
REV_DIR="$4"

export LAB_ROOT=/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab
REV_NAME=$(basename "$REV_DIR")
RUN_ID="${REV_NAME}_${BACKBONE}_${VARIANT}"
OUT="$LAB_ROOT/runs/${REV_NAME}/${BACKBONE}/${VARIANT}"
LOG_DIR="$LAB_ROOT/logs/${RUN_ID}"
TMP_DIR="$LAB_ROOT/tmp/${RUN_ID}"
CACHE_DIR="$LAB_ROOT/cache/${RUN_ID}"
ENV_HOME="$LAB_ROOT/envs/runtime_home/${RUN_ID}"

mkdir -p "$OUT/gmr" "$LOG_DIR" "$TMP_DIR" "$CACHE_DIR"/{xdg,pycache,torch,hf,pip,matplotlib,cuda,triton,torch_extensions} "$ENV_HOME"

# Runtime isolation environment variables
export PYTHONUNBUFFERED=1
export TMPDIR="$TMP_DIR"
export TEMP="$TMPDIR" TMP="$TMPDIR"
export HOME="$ENV_HOME"
export XDG_CACHE_HOME="$CACHE_DIR/xdg"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$CACHE_DIR/pycache"
export TORCH_HOME="$CACHE_DIR/torch"
export HF_HOME="$CACHE_DIR/hf"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export MPLCONFIGDIR="$CACHE_DIR/matplotlib"
export CUDA_CACHE_PATH="$CACHE_DIR/cuda"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export TORCH_EXTENSIONS_DIR="$CACHE_DIR/torch_extensions"
export WANDB_DIR="$LOG_DIR/wandb"
export WANDB_MODE=disabled

# Python environment
PYTHON_BIN="/home/guoxiangyu/miniconda3/envs/gmr/bin/python"
export PYTHONPATH="$REV_DIR/src"
export CUDA_VISIBLE_DEVICES="$GPU_PHYSICAL_ID"
export OMP_NUM_THREADS=1

CONFIG_FILE="$REV_DIR/configs/experiments/interlayer/${BACKBONE}/${VARIANT}.json"

# Check write scope
"$PYTHON_BIN" "$LAB_ROOT/scripts/check_write_scope.py" \
  --lab "$LAB_ROOT" \
  --check-paths "$OUT" "$LOG_DIR" "$TMP_DIR" "$CACHE_DIR" "$ENV_HOME" \
  --check-env

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting run ${RUN_ID} on GPU ${GPU_PHYSICAL_ID} (device cuda:0)..."

# Execute training
"$PYTHON_BIN" -m soccergmr train \
  --config "$CONFIG_FILE" \
  --stage gmr \
  --output "$OUT/gmr" \
  --device cuda:0 \
  > "$LOG_DIR/train.stdout.log" 2> "$LOG_DIR/train.stderr.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training finished for ${RUN_ID}. Starting Val diagnostics..."

# Execute strict Val diagnostics
"$PYTHON_BIN" "$REV_DIR/diagnosed/run_variant_diagnostics.py" \
  --config "$CONFIG_FILE" \
  --checkpoint "$OUT/gmr/best.pt" \
  --split val \
  --iou-threshold 0.4 \
  --output-dir "$OUT/val_diagnosis" \
  --device cuda:0 \
  > "$LOG_DIR/diagnosis.stdout.log" 2> "$LOG_DIR/diagnosis.stderr.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Diagnostics completed for ${RUN_ID}."
