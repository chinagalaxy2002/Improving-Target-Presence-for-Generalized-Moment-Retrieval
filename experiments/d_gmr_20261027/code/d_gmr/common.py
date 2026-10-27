"""Experiment-local paths and serialization for D-GMR."""
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
ROOT = Path(__file__).resolve().parent
RUNS = Path(os.getenv('DGMR_RUNS_DIR', Path(__file__).resolve().parents[2]/'runs')).resolve()
REFERENCE = REPO/'ablations/component_ablation_20261023'
LABELS = REFERENCE/'data'
ASSETS = Path(os.getenv('GMR_ASSETS_DIR', '/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr'))
PRETRAINED = Path(os.getenv('GMR_PRETRAINED_ROOT', '/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder'))
FROZEN_RUNS = REPO/'gemr_runs'
BACKBONES = ('moment_detr', 'qd_detr', 'cg_detr', 'eatr', 'flashvtg')
sys.dont_write_bytecode = True


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=str)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if exclusive:
        os.link(temporary, path)
        temporary.unlink()
    else:
        os.replace(temporary, path)


def seed_all(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configuration():
    return json.loads((ROOT/'configs/d_gmr.json').read_text())
