#!/usr/bin/env bash
set -euo pipefail
KIT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=${GMR_PROJECT_ROOT:-/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval}
OUTPUT_DIR=${GMR_GEQ_OUTPUT:-$PROJECT_DIR/geq_finetune_experiments/20261031}
SESSION_NAME=gmr_baseline_geq_20261031
mkdir -p "$OUTPUT_DIR"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Queue session already exists: $SESSION_NAME"
    exit 0
fi
tmux new-session -d -s "$SESSION_NAME" -c "$KIT_DIR" \
    "/home/guoxiangyu/miniconda3/bin/python -u launch_queue.py --variants baseline --state-name BASELINE_GEQ_QUEUE.json --no-report >> '$OUTPUT_DIR/baseline_geq_queue.log' 2>&1 && /home/guoxiangyu/miniconda3/bin/python -u plot_complete_comparison.py --refresh >> '$OUTPUT_DIR/baseline_geq_queue.log' 2>&1"
echo "Started four baseline-only GEQ jobs: $SESSION_NAME"
