#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_RUN_DIR="${1:?Usage: bash train_stage2.sh /absolute/fresh/run/directory [cuda:0]}"
TASK_DEVICE="${2:-cuda:0}"
if [[ "$TASK_RUN_DIR" != /* || -e "$TASK_RUN_DIR" ]]; then
  echo 'Choose a fresh absolute run directory.' >&2
  exit 2
fi
export DGMR_RUNS_DIR="$TASK_RUN_DIR"
export DGMR_STAGE1_ROOT="${DGMR_STAGE1_ROOT:-$SCRIPT_DIR/stage1/runs}"
for TASK_BACKBONE in moment_detr qd_detr cg_detr eatr flashvtg; do
  test -f "$DGMR_STAGE1_ROOT/$TASK_BACKBONE/E1/epoch_004.ckpt"
done
for TASK_BACKBONE in moment_detr qd_detr cg_detr eatr flashvtg; do
  python "$SCRIPT_DIR/run.py" preflight --backbone "$TASK_BACKBONE" --device "$TASK_DEVICE"
  python "$SCRIPT_DIR/run.py" train --backbone "$TASK_BACKBONE" --device "$TASK_DEVICE"
  TASK_CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_path"])' "$TASK_RUN_DIR/$TASK_BACKBONE/best_epoch.json")"
  python "$SCRIPT_DIR/run.py" test --backbone "$TASK_BACKBONE" \
    --checkpoint "$TASK_CHECKPOINT" --split test --threshold 0.4 \
    --device "$TASK_DEVICE" --formal-test
done
python "$SCRIPT_DIR/run.py" report
