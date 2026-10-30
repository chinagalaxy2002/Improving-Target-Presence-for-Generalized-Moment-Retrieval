#!/usr/bin/env bash
set -euo pipefail

KIT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
SPLIT=${SPLIT:-test}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
variants=(baseline shared sepffn instruction multi adaptive)
gpus=("$GPU0" "$GPU1" "$GPU0" "$GPU1" "$GPU0" "$GPU1")
pids=()

for i in "${!variants[@]}"; do
  variant=${variants[$i]}
  log="$KIT_ROOT/mr_experiments/$variant/$SPLIT/evaluate.log"
  mkdir -p "$(dirname "$log")"
  CUDA_VISIBLE_DEVICES=${gpus[$i]} OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    python -u "$KIT_ROOT/gmr_multiroute_kit/run_eval.py" \
      --variant "$variant" --split "$SPLIT" > "$log" 2>&1 &
  pids+=("$!")
  printf '%s GPU=%s PID=%s\n' "$variant" "${gpus[$i]}" "$!"
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
