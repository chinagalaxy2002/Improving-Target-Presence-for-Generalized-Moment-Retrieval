#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
VARIANT=${1:?usage: train.sh VARIANT}
exec python -u "$HERE/run.py" --variant "$VARIANT" --mode train
