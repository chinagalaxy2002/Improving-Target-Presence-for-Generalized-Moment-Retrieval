#!/usr/bin/env bash
set -euo pipefail
KIT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "$KIT_DIR")
OUTPUT_DIR="$PROJECT_DIR/geq_protocol_experiments/20261031"
SESSION_NAME=gmr_geq_val_protocol_20261031
mkdir -p "$OUTPUT_DIR"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Queue already exists: $SESSION_NAME"
    exit 0
fi
tmux new-session -d -s "$SESSION_NAME" -c "$KIT_DIR" \
    "/home/guoxiangyu/miniconda3/bin/python -u launch_queue.py >> '$OUTPUT_DIR/queue.log' 2>&1"
echo "Started: $SESSION_NAME"
