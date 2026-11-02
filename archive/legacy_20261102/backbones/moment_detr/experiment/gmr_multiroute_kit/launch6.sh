#!/usr/bin/env bash
set -euo pipefail
KIT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=${GMR_PROJECT_ROOT:-$(cd "$KIT_DIR/.." && pwd)}
mkdir -p "$ROOT/mr_experiments/logs"
variants=(baseline shared sepffn instruction multi adaptive)
gpus=("${GPU0:-0}" "${GPU1:-1}" "${GPU0:-0}" "${GPU1:-1}" "${GPU0:-0}" "${GPU1:-1}")
for i in "${!variants[@]}"; do
  variant=${variants[$i]}
  CUDA_VISIBLE_DEVICES=${gpus[$i]} OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    nohup python -u "$KIT_DIR/run_train.py" --variant "$variant" \
    > "$ROOT/mr_experiments/logs/$variant.log" 2>&1 < /dev/null &
  echo "$!" > "$ROOT/mr_experiments/logs/$variant.pid"
  echo "$variant GPU=${gpus[$i]} PID=$!"
done
wait
