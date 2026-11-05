"""Register one native DETR family per worker process.

Native snapshots share historical top-level module names. Explicit isolation
prevents accidentally mixing two families' `models` or `utils` packages.
"""
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import sys
import torch
from torch import nn
from ..config import Options

@dataclass(frozen=True)
class Spec:
    name: str
    builder: str
    feature_order: tuple
    text_attention_mask: bool = False
    encoder_offset: int = 0
    encoder_passes: int = 1
    extra_encoder_input: str = ''

SPECS = {
    s.name:s for s in (
        Spec('moment_detr','models.moment_detr_gmr.moment_detr:build_model',('clip','slowfast')),
        Spec('eatr','models.build_model:build_model',('slowfast','clip')),
        Spec('qd_detr','qd_detr.model:build_model',('slowfast','clip'),encoder_offset=1),
        Spec('cg_detr','cg_detr.model:build_model',('slowfast','clip'),encoder_offset=1),
        Spec('sim_detr','sim_detr.soccer_gmr_csc.model_builder:build_soccer_gmr_model',('clip','slowfast'),True,encoder_passes=2),
        Spec('bm_detr','bm_detr.model:build_model',('slowfast','clip'),extra_encoder_input='neg_src_txt'),
        Spec('bam_detr','bam_detr.model:build_model',('slowfast','clip'),encoder_offset=1,encoder_passes=2),
        Spec('uvcom','uvcom.model:build_model',('slowfast','clip'),encoder_offset=1,encoder_passes=2),
    )
}
_ACTIVE = None

def get_spec(name):
    if name not in SPECS:
        raise ValueError(f'Unknown backbone {name}; choose {list(SPECS)}')
    return SPECS[name]

def activate(name):
    global _ACTIVE
    spec=get_spec(name)
    if _ACTIVE is not None and _ACTIVE!=name:
        raise RuntimeError('Use a fresh process for each backbone (native module namespaces overlap)')
    root=Path(__file__).parent/'vendor'/name
    for package in ('models','utils',name):
        module=sys.modules.get(package)
        if module is not None and getattr(module,'__file__',None):
            if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
                raise RuntimeError(f'Foreign native package already imported: {package}')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    _ACTIVE=name
    return spec

def build(config, device='cpu'):
    spec=activate(config['backbone'])
    module,function=spec.builder.split(':')
    options=Options(config['architecture'])
    options.device=str(device)
    options.max_v_l=config['data']['max_video_tokens']
    if spec.name in ('bm_detr','bam_detr','uvcom'):options.max_q_l=config['data']['max_text_tokens']
    options.disable_synthetic_negatives=True
    model,criterion=getattr(import_module(module),function)(options)
    if spec.name=='bm_detr':
        from .bm_adapter import BMModel, BMLoss
        return BMModel(model, options.hidden_dim).to(device), BMLoss(criterion).to(device)
    if spec.name=='bam_detr':
        from .bam_adapter import BAMModel, BAMLoss
        return BAMModel(model, options.hidden_dim*3).to(device), BAMLoss(criterion).to(device)
    if spec.name=='uvcom':
        from .uvcom_adapter import UVModel, UVLoss
        return UVModel(model, options.hidden_dim).to(device), UVLoss(criterion).to(device)
    if spec.name=='eatr':
        model.exist_head=import_module('models.module').MLP(options.hidden_dim,options.hidden_dim,1,2)
    if spec.name=='moment_detr':
        criterion.losses=[x for x in criterion.losses if x!='exist']
    if spec.name=='qd_detr':criterion=criterion.native
    if spec.name=='cg_detr':
        from .loss import stable_cg_criterion
        criterion=stable_cg_criterion(criterion)
    from .loss import GMRLoss
    return model.to(device),GMRLoss(criterion,spec.name).to(device)

def encoder_root(model, name):
    if name in ('bm_detr','bam_detr','uvcom'):return model.native
    return model.semantic_model.base_model if name=='sim_detr' else model

def encoder_layers(model, name):
    root=encoder_root(model,name)
    return root.CIM.GKA_Module.layers if name=='uvcom' else root.transformer.encoder.layers

def forward(model, name, inputs, targets=None, metas=None, epoch=None):
    if name=='cg_detr':
        return model(**inputs,vid=None,qid=[m['qid'] for m in metas],targets=targets)
    if name=='uvcom':return model(**inputs,epoch=epoch)
    return model(**inputs)
