"""Experiment-local paths, immutable provenance, and serialization."""
import hashlib, importlib.util, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OLD = REPO / 'ablations/global_hard_roi_20261023'
ASSETS = Path(os.getenv('GMR_ASSETS_DIR', '/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr'))
PRETRAINED = Path(os.getenv('GMR_PRETRAINED_ROOT', '/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder'))
BACKBONES = ['moment_detr', 'qd_detr', 'cg_detr', 'eatr', 'flashvtg']
sys.dont_write_bytecode = True

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n'); tmp.replace(path)

def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def seed():
    random.seed(2026); np.random.seed(2026); torch.manual_seed(2026); torch.cuda.manual_seed_all(2026)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

def source(backbone):
    if backbone == 'flashvtg':
        default = json.loads((OLD/'results/flashvtg/gmr/CONFIG.json').read_text())['initialization']['snapshot']
        return Path(os.getenv('GMR_FLASH_CHECKPOINT', default))
    return PRETRAINED/backbone/'gmr'/('best.pt' if backbone == 'cg_detr' else 'best.ckpt')
