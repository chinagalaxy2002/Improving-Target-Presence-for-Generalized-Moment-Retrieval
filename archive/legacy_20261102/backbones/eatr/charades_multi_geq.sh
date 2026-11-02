#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$repo_root/shared/charades_multi_geq/runner.py" --backbone eatr "$@"
