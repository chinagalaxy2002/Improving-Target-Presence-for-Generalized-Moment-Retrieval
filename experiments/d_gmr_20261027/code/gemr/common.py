"""Paths and ordinary serialization; no digest-based verification."""
import importlib.util
import json
import os
import random
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
ROOT = REPO / 'gemr'
RUNS = Path(os.getenv('GEMR_RUNS_DIR', REPO / 'gemr_runs')).resolve()
REFERENCE = REPO / 'ablations/component_ablation_20261023'
LABELS = REFERENCE / 'data'
ASSETS = Path(os.getenv('GMR_ASSETS_DIR', '/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr'))
PRETRAINED = Path(os.getenv('GMR_PRETRAINED_ROOT', '/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder'))
GEQ_ROOT = Path(os.getenv('GEMR_GEQ_ROOT', '/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/experiments/component_ablation_20261023/runs'))
BACKBONES = ('moment_detr', 'qd_detr', 'cg_detr', 'eatr', 'flashvtg')
sys.dont_write_bytecode = True


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def dump(path, obj, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if exclusive:
        # link() publishes atomically and fails if the destination exists.
        os.link(temp, path)
        temp.unlink()
    else:
        os.replace(temp, path)


def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configuration():
    return json.loads((ROOT / 'configs/gemr_config.json').read_text())
