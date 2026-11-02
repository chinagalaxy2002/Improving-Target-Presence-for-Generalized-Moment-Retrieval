"""Immutable inputs and atomic, isolated evaluation artifacts."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
HANDOFF = PROJECT / 'soccergmr_full_test_handoff_20261102'
CORE = Path('/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr')
KIT = PROJECT / 'gmr_geq_protocol_kit'
CHECKPOINTS = json.loads((HANDOFF / 'CHECKPOINTS.json').read_text())
DATA = json.loads((HANDOFF / 'DATA_AUDIT.json').read_text())
FEATURES = Path(DATA['feature_root'])
GT = Path(DATA['available_decodable_test']['path'])
TITLE = 'Standard-trained checkpoints → SoccerGMR Full-Decodable Test (2970/2986)'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n')
    tmp.replace(path)

def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]

def jsonl(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(''.join(json.dumps(d, ensure_ascii=False, allow_nan=False) + '\n' for d in data))
    tmp.replace(path)

def run_dir(entry): return HERE / 'runs' / entry['backbone'] / entry['method']

def code_files():
    roots = [KIT, PROJECT/'gmr_multiroute_kit',
             PROJECT/'generalized-moment-retrieval/training/moment_detr_gmr',
             PROJECT/'generalized-moment-retrieval/models/moment_detr_gmr',
             PROJECT/'generalized-moment-retrieval/configs/moment_detr_gmr',
             PROJECT/'generalized-moment-retrieval/eval',
             Path('/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared/official_eval'),
             CORE.parent/'eatr/models', CORE.parent/'eatr/utils',
             CORE.parent/'qd_detr/gmr_deployment', CORE.parent/'cg_detr/gmr_source']
    files = set(HERE.glob('*.py')) | set(CORE.glob('*.py'))
    files |= {CORE.parent/'eatr/deploy.py',CORE.parent/'eatr/config.py',
              CORE.parent/'qd_detr/train_gmr.py', CORE.parent/'cg_detr/train_gmr.py'}
    for root in roots:
        files.update(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.yml','.yaml'))
    return {str(p): sha(p) for p in sorted(files)}
