#!/usr/bin/env bash
set -euo pipefail
cd /home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/eatr
exec /home/guoxiangyu/miniconda3/envs/gmr/bin/python -u deploy.py --mode train --variant both
