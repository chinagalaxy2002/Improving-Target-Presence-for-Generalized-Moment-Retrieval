#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VARIANT=${1:?usage: train.sh VARIANT}
exec python -u "$HERE/experiment/gmr_multiroute_kit/run_train.py" --variant "$VARIANT"
