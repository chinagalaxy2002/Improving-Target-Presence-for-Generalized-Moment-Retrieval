#!/usr/bin/env bash
set -euo pipefail
KIT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_DIR=$(dirname "$KIT_DIR")
OUTPUT_DIR="$PROJECT_DIR/geq_epoch_study/20261031"
SESSION_NAME=gmr_geq_parallel_20261031
mkdir -p "$OUTPUT_DIR"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Study queue already exists: $SESSION_NAME"
    exit 0
fi
TAKEOVER_ARG=""
if [[ -n "${TAKEOVER_PID:-}" ]]; then
    [[ "$TAKEOVER_PID" =~ ^[0-9]+$ ]] || { echo "Invalid TAKEOVER_PID"; exit 1; }
    TAKEOVER_ARG="--takeover-pid $TAKEOVER_PID"
fi
tmux new-session -d -s "$SESSION_NAME" -c "$KIT_DIR" \
    "/home/guoxiangyu/miniconda3/bin/python -u launch_study_queue.py $TAKEOVER_ARG >> '$OUTPUT_DIR/queue.log' 2>&1"
echo "Started: $SESSION_NAME"
