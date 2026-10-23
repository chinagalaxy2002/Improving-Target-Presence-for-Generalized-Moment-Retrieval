"""Native FlashVTG export with independent presence and hard RoI refinement."""
import os
import json
import types
from pathlib import Path
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
os.environ.setdefault('GMR_E0_CHECKPOINT', json.loads(
    (REPO / 'provenance/EXTERNAL_CHECKPOINTS.json').read_text())['E0']['path'])
os.environ.setdefault('GMR_ASSETS_DIR', '/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr')
from environment import make_dataset, digest, E0_CHECKPOINTS
from runtime import build
from training.flash_vtg_gmr.dataset import start_end_collate, prepare_batch_inputs
from components import remove_lora, no_saliency

opt = None


def initialize(device, construction, global_query):
    global opt
    model, opt, checks = build(construction, device)
    remove_lora(model)
    head = model.base.exist_head
    head.extractor.pooling_type = 'uniform_box'
    head.refiner.register_forward_pre_hook(no_saliency, with_kwargs=True)
    model.job.update(use_lora=False, pooling_type='uniform_box')
    head.job.update(model.job)
    model.use_lora = False
    model.requires_grad_(False)
    for name in ['token', 'text_read', 'video_reads', 'global_head']:
        item = getattr(head, name)
        item.requires_grad_(global_query)
    head.extractor.requires_grad_(True)
    head.refiner.requires_grad_(True)
    head.refiner.sal_proj.requires_grad_(False)
    # Restore the trained GMR presence MLP on its original pooled input. The
    # second head invocation (None input) only supplies refinement context.
    model.native_exist_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1)).to(device)
    state = torch.load(E0_CHECKPOINTS[2026], map_location='cpu', weights_only=False)['model']
    model.native_exist_head.load_state_dict({n.split('exist_head.', 1)[1]: v for n, v in state.items() if n.startswith(('exist_head.', 'base.exist_head.'))}, strict=True)
    model.native_exist_head.requires_grad_(False)
    model._native_logit = None

    def capture(module, args, output):
        if args[0] is not None:
            model._native_logit = model.native_exist_head(args[0]).squeeze(-1)
    head.register_forward_hook(capture)
    original_forward = model.forward

    def forward(self, **inputs):
        self._native_logit = None
        out = original_forward(**inputs)
        if not global_query:
            assert self._native_logit is not None
            out['pred_exist_logits'] = self._native_logit
            if '_out' in out:
                out['_out']['pred_exist_score'] = self._native_logit.sigmoid().detach().cpu()
        out['pred_exist_score'] = out['pred_exist_logits'].sigmoid()
        return out

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            # Native FlashVTG constructs dense training proposals only while
            # training; keep that branch, with original dropout frozen.
            for m in self.base.modules():
                if isinstance(m, nn.Dropout) and not any(p.requires_grad for p in m.parameters(recurse=False)):
                    m.eval()
        return self

    model.forward = types.MethodType(forward, model)
    model.train = types.MethodType(train, model)
    return model, dict(verified_gmr_baseline=True, sha256=digest(E0_CHECKPOINTS[2026]),
                       snapshot=str(E0_CHECKPOINTS[2026]), baseline_config=vars(opt), checks=checks)


def get_dataset(split):
    return make_dataset(opt, split)


def collate(items, device):
    metas, batch = start_end_collate(items)
    inputs, targets = prepare_batch_inputs(batch, device)
    targets.update(label=metas, fps=torch.full((len(metas),), 1 / opt.clip_length, device=device))
    inputs['targets'] = targets
    return metas, inputs
