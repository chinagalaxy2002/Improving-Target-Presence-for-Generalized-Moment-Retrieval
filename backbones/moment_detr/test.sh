#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VARIANT=${1:?usage: test.sh VARIANT}
exec python -u "$HERE/experiment/gmr_multiroute_kit/run_eval.py" --variant "$VARIANT" --split "${SPLIT:-test}" --device "${DEVICE:-cuda}"
