#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_DEVICE="${1:-cuda:0}"
if [[ -n "${DGMR_FLASHVTG_CHECKPOINT:-}" ]]; then
  export GMR_FLASH_CHECKPOINT="${GMR_FLASH_CHECKPOINT:-$DGMR_FLASHVTG_CHECKPOINT}"
fi
python "$SCRIPT_DIR/stage1/audit_assets.py"
for TASK_BACKBONE in moment_detr qd_detr cg_detr eatr flashvtg; do
  python "$SCRIPT_DIR/stage1/run.py" --setting E1 --backbone "$TASK_BACKBONE" --device "$TASK_DEVICE" --preflight-only
  python "$SCRIPT_DIR/stage1/run.py" --setting E1 --backbone "$TASK_BACKBONE" --device "$TASK_DEVICE"
done
