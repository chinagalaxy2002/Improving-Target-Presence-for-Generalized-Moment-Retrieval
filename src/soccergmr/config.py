"""Validated, serializable experiment and data configuration."""
from __future__ import annotations
import json
import os
from pathlib import Path

from .io import fingerprint

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = dict(seed=2023, batch_size=8, eval_batch_size=8, accumulation=1,
    workers=2, max_epochs=200, patience=20, validation_every=1,
    learning_rate=3e-5, weight_decay=1e-4, gradient_clip=0.1,
    geq_max_epochs=200, geq_patience=10, geq_lr_native=1e-5, geq_lr_head=3e-5,
    thresholds=[0.4, 0.5, 0.6], amp=False)


class Options(dict):
    """Native builders use both dot access and membership tests."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


def load_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    unknown = set(raw)-{'backbone','data','protocol','architecture','description','clip'}
    if unknown:
        raise ValueError('Unknown configuration keys: '+str(sorted(unknown)))
    from .backbones.registry import get_spec
    get_spec(raw['backbone'])
    unknown_protocol=set(raw.get('protocol',{}))-set(DEFAULT_PROTOCOL)
    if unknown_protocol:raise ValueError('Unknown protocol keys: '+str(sorted(unknown_protocol)))
    protocol = dict(DEFAULT_PROTOCOL, **raw.get('protocol', {}))
    if protocol['thresholds'] != [0.4, 0.5, 0.6]:
        raise ValueError('GEQ selection thresholds must be exactly 0.4/0.5/0.6')
    if protocol['geq_patience'] != 10:
        raise ValueError('The primary protocol requires GEQ patience=10')
    for name in ('batch_size','eval_batch_size','accumulation','max_epochs','geq_max_epochs','patience','validation_every'):
        if protocol[name] < 1:
            raise ValueError(name+' must be positive')
    if protocol['validation_every'] != 1:
        raise ValueError('Epoch patience requires a complete Val evaluation every epoch')
    if protocol['amp']:
        raise ValueError('AMP is disabled in the validated primary protocol')
    data = raw['data']
    if set(data.get('splits', {})) != {'train','val','test_standard','test_full'}:
        raise ValueError('Specify Standard train/val and both Test split paths')
    def resolve(value):
        expanded = os.path.expandvars(str(value))
        if '$' in expanded:
            raise ValueError('Unset environment variable in '+str(value))
        p = Path(expanded).expanduser()
        return str((path.parent/p).resolve()) if not p.is_absolute() else str(p.resolve())
    data = dict(data)
    data['splits'] = {k:resolve(v) for k,v in data['splits'].items()}
    data['features'] = {k:resolve(v) for k,v in data['features'].items()}
    if set(data['features']) != {'standard','full'}:
        raise ValueError('Separate Standard and Full feature roots are required')
    data.setdefault('max_video_tokens', 128)
    data.setdefault('max_text_tokens', 32)
    data.setdefault('clip_seconds', 2.0)
    data.setdefault('full_label', 'Full Test')
    architecture_path = raw.get('architecture', '../backbones/'+raw['backbone']+'.json')
    architecture = json.loads(Path(resolve(architecture_path)).read_text())
    # Paths in historical option snapshots cannot affect current data or output.
    for key in ('train_path','eval_path','test_path','v_feat_dirs','t_feat_dir','results_dir','resume','ckpt_filepath'):
        architecture[key] = None
    config = dict(backbone=raw['backbone'], data=data, protocol=protocol, architecture=architecture)
    if 'clip' in raw:
        clip = dict({'gmr_weight': 0., 'geq_weight': 0.}, **raw['clip'])
        if 'method' in clip:
            clip = validate_clip_v2(raw['clip'])
        if 'method' not in clip and set(clip) != {'gmr_weight', 'geq_weight'}:
            raise ValueError('Unknown clip configuration keys')
        import math
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in clip.values() if not isinstance(v, str)):
            raise ValueError('Clip weights must be finite and nonnegative')
        config['clip'] = clip
    config['identity'] = fingerprint(config)
    return config


def stage_config_identity(config, stage):
    """Version 1: Stage 1 is independent only of Stage 2 auxiliary weight.

    All source/data/native protocol/architecture and Stage 1 clip settings remain
    bound. This permits A/B and C/D to share audited, strictly loaded parents.
    """
    if stage != 'gmr' or 'clip' not in config or 'method' in config['clip']:
        return config['identity']
    projected = {k: v for k, v in config.items() if k != 'identity'}
    projected['clip'] = {'gmr_weight': config['clip']['gmr_weight']}
    return fingerprint({'schema': 'clip-stage-identity-v1', 'config': projected})


def validate_clip_v2(raw):
    defaults = dict(method='balanced', loss_weight=0.1, gamma=2., dice_weight=0.2,
                    rank_weight=0.2, margin=0.5, topk=5)
    if set(raw)-set(defaults):raise ValueError('Unknown v2 clip configuration')
    c = dict(defaults, **raw)
    if c['method'] not in ('balanced','focal','dice','ranking','soft','native'):
        raise ValueError('Unknown clip method')
    import math
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v < 0 for k,v in c.items() if k!='method'):
        raise ValueError('Invalid clip numeric setting')
    if not isinstance(c['topk'],int) or c['topk']<1:raise ValueError('topk must be a positive integer')
    return c
