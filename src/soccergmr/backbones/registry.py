"""Moment-DETR native interface for the clip-supervision experiments."""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys
from ..config import Options

@dataclass(frozen=True)
class Spec:
    name: str = 'moment_detr'
    feature_order: tuple = ('clip', 'slowfast')
    text_attention_mask: bool = False
    encoder_offset: int = 0
    encoder_passes: int = 1
    extra_encoder_input: str = ''

SPECS = {'moment_detr': Spec()}

def get_spec(name):
    if name not in SPECS:raise ValueError('This branch supports only moment_detr')
    return SPECS[name]

def build(config, device='cpu'):
    get_spec(config['backbone'])
    root=Path(__file__).parent/'vendor/moment_detr'
    for package in ('models','utils'):
        module=sys.modules.get(package)
        if module is not None and getattr(module,'__file__',None) and not Path(module.__file__).resolve().is_relative_to(root.resolve()):
            raise RuntimeError('Foreign native module: '+package)
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    options=Options(config['architecture']);options.device=str(device)
    options.max_v_l=config['data']['max_video_tokens'];options.disable_synthetic_negatives=True
    model,criterion=import_module('models.moment_detr_gmr.moment_detr').build_model(options)
    criterion.losses=[x for x in criterion.losses if x!='exist']
    from .loss import GMRLoss
    return model.to(device),GMRLoss(criterion,'moment_detr').to(device)

def encoder_root(model,name):
    get_spec(name);return model

def encoder_layers(model,name):
    return encoder_root(model,name).transformer.encoder.layers

def forward(model,name,inputs,targets=None,metas=None,epoch=None):
    get_spec(name);return model(**inputs)
