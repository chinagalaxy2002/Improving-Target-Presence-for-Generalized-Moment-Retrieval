"""Two independent additions to a frozen, pretrained GMR localizer."""
from pathlib import Path
import importlib
import sys
import types

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SETTINGS = {'hard_only': (False, True), 'global_hard': (True, True)}


def remove_lora(module):
    """Keep the exact pretrained linear tensors, remove adaptation branches."""
    for name, child in list(module.named_children()):
        if hasattr(child, 'lora_A') and hasattr(child, 'base_linear'):
            setattr(module, name, child.base_linear)
        else:
            remove_lora(child)


def no_saliency(module, args, kwargs):
    args = list(args)
    if len(args) >= 5:
        args[4] = None
    kwargs.pop('saliency_scores', None)
    return tuple(args), kwargs


class NativeVariant(nn.Module):
    def __init__(self, inner, global_query):
        super().__init__()
        self.inner = inner
        self.global_query = global_query
        self.native_logit = None
        inner.base.register_forward_hook(self.capture_native)
        remove_lora(inner)
        inner.requires_grad_(False)
        head = next(getattr(inner, n) for n in
                         ['event_query', 'global_query', 'global_event_query'] if hasattr(inner, n))
        head.requires_grad_(global_query)
        inner.extractor.requires_grad_(True)
        inner.refiner.requires_grad_(True)
        inner.refiner.sal_proj.requires_grad_(False)
        inner.refiner.register_forward_pre_hook(no_saliency, with_kwargs=True)
        inner.extractor.pooling_type = 'uniform_box'

    def capture_native(self, module, args, out):
        self.native_logit = out.get('pred_exist_logits', out.get('pred_exist_logit'))

    def train(self, mode=True):
        super().train(mode)
        self.inner.base.eval()
        return self

    def forward(self, **inputs):
        out = self.inner(**inputs)
        if not self.global_query:
            assert self.native_logit is not None
            out['pred_exist_logits'] = self.native_logit.reshape(-1)
        out['pred_exist_score'] = out['pred_exist_logits'].sigmoid()
        return out


def build(backbone, job, device):
    global_query, hard = SETTINGS[job['setting']]
    # Historical constructors materialize the same random tensors before
    # removing the unwanted branches. These are construction-only arguments.
    construction = dict(job, use_lora=True, pooling_type='gaussian',
                        presence_source='global_refine', use_boundary_refine=True)
    flash = backbone == 'flashvtg'
    if not flash:
        sys.path.insert(0, str(ROOT / 'backends' / backbone))
        backend = importlib.import_module('backend')
        inner, init = backend.initialize(device, construction)
        model = NativeVariant(inner, global_query).to(device)
        extractor, refiner = inner.extractor, inner.refiner
    else:
        sys.path.insert(0, str(ROOT / 'code'))
        import flash_backend as backend
        model, init = backend.initialize(device, construction, global_query)
        extractor, refiner = model.base.exist_head.extractor, model.base.exist_head.refiner
    assert hard and extractor.pooling_type == 'uniform_box'
    assert not any(hasattr(m, 'lora_A') for m in model.modules())
    assert not any(p.requires_grad for p in refiner.sal_proj.parameters())
    init['construction_arguments'] = construction
    return backend, model, init, extractor, refiner
