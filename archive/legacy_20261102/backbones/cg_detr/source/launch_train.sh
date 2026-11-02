#!/usr/bin/env bash
set -euo pipefail
cd /home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec /home/guoxiangyu/miniconda3/envs/gmr/bin/python -u train_soccer.py
