"""Sim-DETR adapter using the repository's shared Global Event Query head."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / 'source'
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SOURCE))
from shared.geq_protocol.geq_model import GlobalEventQueryHead
from sim_detr.soccer_gmr_csc.model_builder import build_soccer_gmr_model
from sim_detr.soccer_gmr_csc.dataset import SoccerGMRDataset
from sim_detr.soccer_gmr_csc.evaluation import generate_submission


class SimDETRGEQ(nn.Module):
    """Same GEQ readout; read the first encoder pass before Sim's HD2MR pass."""
    def __init__(self, base, mode):
        super().__init__()
        assert mode in ('full', 'frozen')
        self.base = base.requires_grad_(mode == 'full')
        self.mode = mode
        inner = base.semantic_model.base_model
        layers = inner.transformer.encoder.layers
        self.depth = len(layers)
        self.geq = GlobalEventQueryHead(self.depth)
        self.context = None
        self.video_length = 0
        self.handles = [inner.input_txt_proj.register_forward_hook(self.capture_text)]
        self.handles.extend(layer.register_forward_hook(self.capture_video) for layer in layers)
        if mode == 'frozen':
            self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.mode == 'frozen':
            self.base.eval()
        return self

    def capture_text(self, module, args, output):
        if self.context is not None and 'text' not in self.context:
            self.context['text'] = output if self.mode == 'full' else output.detach()

    def capture_video(self, module, args, output):
        if self.context is not None and len(self.context['video']) < self.depth:
            memory = output.transpose(0, 1)[:, :self.video_length]
            self.context['video'].append(memory if self.mode == 'full' else memory.detach())

    def forward(self, **inputs):
        assert self.context is None
        vm, tm = inputs['src_vid_mask'].bool(), inputs['src_txt_mask'].bool()
        self.video_length = vm.shape[1]
        context = {'video': []}
        self.context = context
        try:
            native = self.base(**inputs)
        finally:
            self.context = None
        assert len(context['video']) == self.depth
        assert all(v.shape[:2] == vm.shape for v in context['video'])
        logit = self.geq(context['text'], tm, context['video'], vm)
        return dict(native, native_output=native, pred_exist_logits=logit,
                    pred_exist_logit=logit, pred_exist_score=logit.sigmoid())


def build_native(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    opt = saved['opt']
    # This historical native checkpoint predates the optional binding branch.
    if not hasattr(opt, 'binding_loss_coef'):
        opt.binding_loss_coef = 0.0
    opt.device = torch.device(device)
    opt.num_workers = 2
    model, criterion = build_soccer_gmr_model(opt)
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device), criterion.to(device), opt


def standard_test_dataset(opt):
    return SoccerGMRDataset(opt.test_path, opt.v_feat_dirs, opt.t_feat_dir,
        max_q_l=opt.max_q_l, max_v_l=opt.max_v_l, max_windows=opt.max_windows,
        clip_len=opt.clip_length, load_labels=True)


def objective(output, targets, criterion, mode):
    """Shared E1 rule: GEQ BCE + unchanged native GMR loss in full mode."""
    geq = F.binary_cross_entropy_with_logits(output['pred_exist_logits'].reshape(-1),
                                             targets['exist_label'].float().reshape(-1))
    if mode == 'frozen':
        return geq, dict(geq=geq)
    losses = criterion(output['native_output'], targets)
    native = sum(value*criterion.weight_dict[key] for key, value in losses.items() if key in criterion.weight_dict)
    return native+geq, dict(native=native, geq=geq)
