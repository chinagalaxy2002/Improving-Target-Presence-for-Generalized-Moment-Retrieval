#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
SESSION="${SESSION:-gmr_multiroute_backbones_20261030}"
mkdir -p "$HERE/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Queue already running in tmux session $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$HERE/../..' && exec python -u '$HERE/queue.py' --gpu0 '$GPU0' --gpu1 '$GPU1' >> '$HERE/logs/queue.log' 2>&1"
echo "Started multi-route queue in tmux=$SESSION GPU0=$GPU0 GPU1=$GPU1 (max two tasks/GPU)"
echo "Status: $HERE/QUEUE.json"
echo "Log:    $HERE/logs/queue.log"
