"""Native factories/data, loaded once per process. No historical init entrypoints."""
import json
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from gemr.common import (ASSETS, GEQ_ROOT, LABELS, PRETRAINED, REFERENCE, REPO,
                         module, rows)


class ProbabilityScoreAdapter:
    """All five inspected outputs are probabilities, with different native origins.

    DETR: softmax foreground probability. FlashVTG: sigmoid of the learned
    mixture of intra/inter scale classification logits, before presence scaling.
    Log-odds are backbone-local; there is no cross-backbone score training.
    """
    @staticmethod
    def to_logit(scores):
        assert torch.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
        return torch.logit(scores.clamp(1e-6, 1-1e-6))

    @classmethod
    def restore(cls, scores, residual):
        logits = cls.to_logit(scores)
        # Difference form is bit-exact at residual=0 and retains its gradient.
        return (scores + ((logits + residual).sigmoid() - logits.sigmoid())).clamp(0, 1)


class GMRBackboneAdapter(nn.Module):
    offset = 0
    final_index = 1
    geq_name = None
    grid_policy = 'clip_length'

    def __init__(self, name, device):
        super().__init__()
        self.name, self.device = name, device
        if name == 'flashvtg':
            record = json.loads((REPO/'ablations/global_hard_roi_20261023/results/flashvtg/gmr/CONFIG.json').read_text())
            self.native_path = record['initialization']['snapshot']
        else:
            self.native_path = str(PRETRAINED/name/'gmr'/('best.pt' if name == 'cg_detr' else 'best.ckpt'))
        baseline = torch.load(self.native_path, map_location='cpu', weights_only=False)
        assert baseline.get('epoch', -1) >= 0
        cfg = baseline.get('opt', baseline.get('config'))
        if name == 'flashvtg':
            cfg = json.loads((REFERENCE/'configs/baseline_protocol.json').read_text())
        self.cfg = dict(cfg) if isinstance(cfg, dict) else vars(cfg).copy()
        self.opt = SimpleNamespace(**dict(self.cfg, device=torch.device('cpu'), num_workers=0))
        # Moment-DETR was trained with CLIP first, unlike the other four.
        # Preserve the saved feature concatenation order exactly.
        original_dirs = self.cfg.get('v_feat_dirs') or ['slowfast', 'clip']
        self.opt.v_feat_dirs = [str(ASSETS/Path(p).name) for p in original_dirs]
        self.opt.t_feat_dir = str(ASSETS/'clip_text')
        source = REFERENCE/('baseline_source' if name == 'flashvtg' else f'backends/{name}/source')
        sys.path.insert(0, str(source))
        self._construct()
        weights = baseline['model']
        if name == 'flashvtg':
            assert all(k.startswith('base.') for k in weights)
            weights = {k[5:]: v for k, v in weights.items()}
        self.base.load_state_dict(weights, strict=True)
        assert all(torch.equal(v, self.base.state_dict()[k]) for k, v in weights.items())
        self.geq_path = GEQ_ROOT/name/'row_02/epoch_004.ckpt'
        pretrained = torch.load(self.geq_path, map_location='cpu', weights_only=False)
        assert pretrained['epoch'] == 4 and pretrained['job']['row'] == 2
        assert pretrained['job']['seed'] == 2026
        effective_path = self.geq_path.parent/'EFFECTIVE_CONFIG.json'
        effective = json.loads(effective_path.read_text())
        flags = effective['effective']
        assert effective['passed'] and flags['global_event_query']
        assert not any(flags[k] for k in ['lora', 'boundary_refiner', 'saliency_6d', 'candidate_presence_fusion'])
        # Reuse the established GEQ class without materializing RoI, LoRA or GAP.
        sys.path.insert(0, str(REFERENCE/'backends/moment_detr'))
        GEQ = module('gemr_reference_geq', REFERENCE/'backends/moment_detr/model.py').GlobalEventQueryHead
        self.geq = GEQ(len(self.base.transformer.encoder.layers))
        prefix = 'base.exist_head.' if name == 'flashvtg' else 'inner.' + self.geq_name + '.'
        geq_weights = {k[len(prefix):]: v for k, v in pretrained['model'].items()
                       if k.startswith(prefix) and k[len(prefix):] in self.geq.state_dict()}
        self.geq.load_state_dict(geq_weights, strict=True)
        native_prefix = 'base.' if name == 'flashvtg' else 'inner.base.'
        equal = 0
        for key, value in pretrained['model'].items():
            if not key.startswith(native_prefix) or '.lora_' in key:
                continue
            canonical = key[len(native_prefix):].replace('.base_linear.', '.')
            if name == 'flashvtg' and canonical.startswith('exist_head.'):
                continue
            assert canonical in weights and torch.equal(value, weights[canonical]), canonical
            equal += 1
        expected = sum(not (name == 'flashvtg' and k.startswith('exist_head.')) for k in weights)
        assert equal == expected, (equal, expected)
        self.base.requires_grad_(False)
        self.geq.requires_grad_(False)
        self.context = None
        self.handles = [self.base.input_txt_proj.register_forward_hook(self._text),
                        self.base.transformer.register_forward_hook(self._transformer),
                        self.geq.global_head.register_forward_pre_hook(self._global)]
        self.handles += [layer.register_forward_hook(self._video) for layer in self.base.transformer.encoder.layers]
        self.manifest = dict(initialization_protocol='component_ablation_20261023 row_02 epoch_004, seed 2026',
                             native_checkpoint=self.native_path, geq_checkpoint=str(self.geq_path),
                             strict_native_load=True, strict_geq_load=True, native_equal_tensors=equal,
                             geq_tensors=len(geq_weights), native_config=self.cfg,
                             temporal_grid=self.grid_policy, clip_length=self.opt.clip_length,
                             video_feature_order=[Path(p).name for p in self.opt.v_feat_dirs],
                             score_semantics=('sigmoid(mixed intra/inter-scale logits)' if name == 'flashvtg'
                                              else 'softmax(foreground/background logits)[foreground=0]'),
                             disabled_modules=['GAP', 'LoRA', 'Gaussian-RoI', 'Hard-RoI', 'boundary_refinement', 'presence_score_fusion'])
        self.to(device).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def _construct(self):
        raise NotImplementedError

    def _text(self, mod, args, output):
        if self.context is not None:
            self.context['text'] = output

    def _video(self, mod, args, output):
        if self.context is not None:
            value = output.transpose(0, 1)[:, self.offset:self.offset+self.length]
            # CG historically snapshots before memory_local += memory_global.
            # Flash historically retains a view: its pyramid's in-place ReLU
            # updates the last memory before GEQ reads it. Cloning that view
            # changes the pretrained GEQ's presence probability substantially.
            self.context['videos'].append(value.clone() if self.name == 'cg_detr' else value)

    def _transformer(self, mod, args, output):
        if self.context is not None:
            final = output[self.final_index]
            if self.name == 'flashvtg':
                final = final.transpose(0, 1)
            self.context['final'] = final[:, :self.length]
            if self.name != 'flashvtg':
                self.context['proposal_features'] = output[0][-1]

    def _global(self, mod, args):
        if self.context is not None:
            self.context['global_event_feature'] = args[0]

    def extract_temporal_features(self, context):
        return context['final']

    def extract_text_features(self, context, mask):
        return (context['text']*mask[..., None]).sum(1)/mask.sum(1, keepdim=True).clamp_min(1)

    def get_native_scores(self, native):
        return native['pred_logits'].softmax(-1)[..., 0]

    def get_native_proposals(self, native, durations):
        centers, widths = native['pred_spans'].unbind(-1)
        xy = torch.stack([centers-widths/2, centers+widths/2], -1)*durations[:, None, None]
        xy = torch.minimum(xy.clamp_min(0), durations[:, None, None])
        return torch.cat([xy, self.get_native_scores(native)[..., None]], -1)

    def timestamps(self, inputs, durations):
        valid = inputs['src_vid_mask'].bool()
        index = torch.arange(valid.shape[1], device=valid.device, dtype=durations.dtype)[None]
        if self.grid_policy == 'normalized_tef':
            # QD/CG data construct TEF as [i/n, (i+1)/n], GT normalized by duration.
            grid = (index+.5)*durations[:, None]/valid.sum(1, keepdim=True)
        else:
            # Moment/EaTR GT normalization and Flash pyramid decode use clip_length.
            left = index*self.opt.clip_length
            right = torch.minimum(left+self.opt.clip_length, durations[:, None])
            grid = (left+right)/2
            valid = valid & (left < durations[:, None])
        assert valid.any(1).all()
        return grid.masked_fill(~valid, 0), valid

    @torch.no_grad()
    def forward(self, inputs):
        assert self.context is None
        self.eval()
        self.length = inputs['src_vid_mask'].shape[1]
        durations = inputs['durations']
        features = {k: v for k, v in inputs.items() if k != 'durations'}
        ctx = {'videos': []}
        self.context = ctx
        try:
            native = self.base(**features)
            logit = self.geq(ctx['text'], inputs['src_txt_mask'].bool(), ctx['videos'], inputs['src_vid_mask'].bool())
            proposals = self.get_native_proposals(native, durations)
            valid = torch.isfinite(proposals).all(-1) & (proposals[..., 1] > proposals[..., 0])
            times, video_valid = self.timestamps(inputs, durations)
            if self.name == 'flashvtg':
                # Dense proposals have no DETR query IDs. Use their temporal RoI
                # feature as the native representation; IDs remain native score order.
                proposal_features = torch.zeros((*proposals.shape[:2], 256), device=proposals.device)
            else:
                proposal_features = ctx['proposal_features']
                assert proposal_features.shape[:2] == proposals.shape[:2]
            return dict(proposals=proposals, proposal_valid=valid, native_features=proposal_features,
                        video=self.extract_temporal_features(ctx), video_valid=video_valid, times=times,
                        text=self.extract_text_features(ctx, inputs['src_txt_mask'].bool()), durations=durations,
                        global_event_feature=ctx['global_event_feature'], presence_logit=logit,
                        presence_probability=logit.sigmoid(),
                        native_presence_logit=native.get('pred_exist_logits', native.get('pred_exist_logit')))
        finally:
            self.context = None

    def dataset(self, split):
        if self.name in ('moment_detr', 'flashvtg'):
            kwargs = dict(dset_name=self.opt.dset_name, data_path=str(LABELS/f'{split}.jsonl'),
                          v_feat_dirs=self.opt.v_feat_dirs, q_feat_dir=self.opt.t_feat_dir,
                          max_q_l=self.opt.max_q_l, max_v_l=self.opt.max_v_l,
                          clip_len=self.opt.clip_length, max_windows=self.opt.max_windows,
                          span_loss_type=self.opt.span_loss_type, ctx_mode=self.opt.ctx_mode,
                          mr_only=True, keep_empty_gt=True, load_labels=True)
            if self.name == 'moment_detr':
                kwargs['domain'] = None
            native_dataset = self.data_module.StartEndDataset
            class AnnotationPreservingDataset(native_dataset):
                def get_span_labels(self, windows, ctx_l):
                    # Native span-label subsampling may shuffle the GT list in
                    # place. TEM/set losses must see ALL original annotations.
                    return super().get_span_labels(copy.deepcopy(windows), ctx_l)
            data = AnnotationPreservingDataset(**kwargs)
        else:
            data = self.data_module.Soccer(split)
        assert data.data == rows(LABELS/f'{split}.jsonl')
        assert len(data) == dict(train=4138, val=465, test=1036)[split]
        return data

    def collate(self, items):
        if self.name in ('moment_detr', 'flashvtg'):
            meta, batch = self.data_module.start_end_collate(items)
            inputs, _ = self.data_module.prepare_batch_inputs(batch, self.device)
        elif self.name == 'cg_detr':
            meta, inputs = self.data_module.collate(items, self.device)
        else:
            meta, inputs, _ = self.data_module.collate(items)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.name == 'flashvtg':
            inputs['targets'] = {}
        inputs['durations'] = torch.tensor([m['duration'] for m in meta], dtype=torch.float32, device=self.device)
        return list(meta), inputs

    def export_ranked_proposals(self, windows, meta):
        # Same duration bounding and collapse repair as established Global-only
        # export, using this backbone's configured clip length instead of 2.
        value = windows.detach().cpu().clone()
        duration = float(meta['duration'])
        value[:, :2] = (value[:, :2]/self.opt.clip_length).round()*self.opt.clip_length
        value[:, :2].clamp_(0, duration)
        for row in value:
            if row[1] <= row[0]:
                if row[1] > 0:
                    row[0] = max(0., float(row[1])-min(self.opt.clip_length, duration))
                else:
                    row[0], row[1] = 0., min(self.opt.clip_length, duration)
        value = value[value[:, 2].argsort(descending=True, stable=True)]
        return value.tolist()
