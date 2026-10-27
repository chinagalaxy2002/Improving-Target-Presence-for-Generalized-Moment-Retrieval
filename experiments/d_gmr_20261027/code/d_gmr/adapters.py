"""Five native GMR factories, data paths, criteria and export rules for D-GMR."""
import copy
import os
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from d_gmr.common import ASSETS, LABELS, PRETRAINED, REFERENCE, REPO, module, rows


def apply_pat_select(windows, sigma=1.0, iou_thresh=0.3, top_k=10):
    """
    Plurality-Aware Temporal Selection (PAT-Select):
    Applies Gaussian IoU decay to redundant proposals to promote diverse temporal moments.
    """
    if not windows:
        return []
    candidates = [list(w) for w in windows]
    selected = []

    while len(selected) < top_k and candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates.pop(0)
        selected.append(best)

        s_b, e_b = best[0], best[1]
        remaining = []
        for cand in candidates:
            s_c, e_c, score_c = cand[0], cand[1], cand[2]
            inter = max(0, min(e_b, e_c) - max(s_b, s_c))
            union = (e_b - s_b) + (e_c - s_c) - inter
            iou = inter / max(union, 1e-6)

            if iou >= iou_thresh:
                decay = float(np.exp(-(iou ** 2) / sigma))
                cand[2] = score_c * decay
            remaining.append(cand)
        candidates = remaining

    return selected


class Adapter:
    offset = 0
    final_index = 1
    grid_policy = 'clip_length'

    def __init__(self, name, device, config=None):
        self.name, self.device = name, device
        self.config = config or {}
        if name == 'flashvtg':
            record = json.loads((REPO/'ablations/global_hard_roi_20261023/results/flashvtg/gmr/CONFIG.json').read_text())
            self.native_path = Path(os.getenv('DGMR_FLASHVTG_CHECKPOINT', record['initialization']['snapshot']))
        else:
            self.native_path = PRETRAINED/name/'gmr'/('best.pt' if name == 'cg_detr' else 'best.ckpt')
        state = torch.load(self.native_path, map_location='cpu', weights_only=False)
        assert state.get('epoch', -1) >= 0
        saved = state.get('opt', state.get('config'))
        if name == 'flashvtg':
            saved = json.loads((REFERENCE/'configs/baseline_protocol.json').read_text())
        self.cfg = dict(saved) if isinstance(saved, dict) else vars(saved).copy()
        self.opt = SimpleNamespace(**dict(self.cfg, device=torch.device('cpu'), num_workers=0))
        original_dirs = self.cfg.get('v_feat_dirs') or ['slowfast', 'clip']
        self.opt.v_feat_dirs = [str(ASSETS/Path(value).name) for value in original_dirs]
        self.opt.t_feat_dir = str(ASSETS/'clip_text')
        self.opt.train_path = str(LABELS/'train.jsonl')
        self.opt.eval_path = str(LABELS/'val.jsonl')
        source = REFERENCE/('baseline_source' if name == 'flashvtg' else f'backends/{name}/source')
        sys.path.insert(0, str(source))
        self._construct()
        weights = state['model']
        if name == 'flashvtg':
            assert all(key.startswith('base.') for key in weights)
            weights = {key[5:]: value for key, value in weights.items()}
        self.base.load_state_dict(weights, strict=True)
        assert all(torch.equal(value, self.base.state_dict()[key]) for key, value in weights.items())
        self.base.requires_grad_(True).to(device)
        self.criterion.to(device)
        self.opt.device = torch.device(device)
        if name == 'flashvtg':
            self.criterion.device = self.opt.device
        self.batch_size = int(self.cfg['bsz'])
        self.manifest = dict(
            initialization_protocol='strict original native GMR; D-GMR two-tier ranking + PAT-select',
            native_checkpoint=str(self.native_path.resolve()), native_checkpoint_epoch=state['epoch'],
            strict_native_load=True, native_equal_tensors=len(weights), native_config=self.cfg,
            temporal_grid=self.grid_policy, clip_length=self.opt.clip_length,
            video_feature_order=[Path(value).name for value in self.opt.v_feat_dirs],
            score_semantics=('sigmoid(mixed intra/inter-scale logits)' if name == 'flashvtg'
                             else 'softmax(foreground/background logits)[foreground=0]'),
            geq_initialization='random seed 2026; no trained GEQ checkpoint',
            pat_select_enabled=bool(self.config.get('enable_pat_select', True)))

    def _construct(self):
        if self.name == 'moment_detr':
            from easydict import EasyDict
            from models.moment_detr_gmr.moment_detr import build_model
            from training.moment_detr_gmr import dataset
            self.base, self.criterion = build_model(EasyDict(vars(self.opt)))
            self.data_module = dataset
        elif self.name == 'qd_detr':
            from qd_detr.model import build_model
            self.base, self.criterion = build_model(self.opt)
            self.data_module = module('d_gmr_qd_data', REFERENCE/'backends/qd_detr/data.py')
            self.data_module.ASSETS, self.data_module.LABELS = ASSETS, LABELS
            self.offset, self.final_index, self.grid_policy = 1, 2, 'normalized_tef'
        elif self.name == 'cg_detr':
            from cg_detr.model import build_model
            self.base, self.criterion = build_model(self.opt)
            self.data_module = module('d_gmr_cg_training', REPO/'backbonefolder/cg_detr/train_gmr.py')
            self.data_module.GMR = ASSETS.parent
            self.criterion.__class__ = self.data_module.GMRCriterion
            self.criterion.weight_dict['loss_exist'] = self.cfg['exist_loss_coef']
            self.offset, self.final_index, self.grid_policy = 1, 2, 'normalized_tef'
        elif self.name == 'eatr':
            from models.build_model import build_model
            from models.module import MLP
            self.base, self.criterion = build_model(self.opt)
            self.base.exist_head = MLP(256, 256, 1, 2)
            self.data_module = module('d_gmr_eatr_data', REFERENCE/'backends/eatr/data.py')
            self.data_module.SOURCE = ASSETS.parent
        elif self.name == 'flashvtg':
            import nncore
            from models.flash_vtg_gmr.model import build_model1
            from training.flash_vtg_gmr import dataset
            self.opt.config = str(REFERENCE/'baseline_source/configs/flash_vtg_gmr/model.py')
            self.opt.cfg = nncore.Config.from_file(self.opt.config)
            self.base, self.criterion = build_model1(self.opt)
            self.data_module = dataset
            self.final_index = 0
        else:
            raise KeyError(self.name)

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
                    return super().get_span_labels(copy.deepcopy(windows), ctx_l)
            data = AnnotationPreservingDataset(**kwargs)
        else:
            data = self.data_module.Soccer(split)
        assert data.data == rows(LABELS/f'{split}.jsonl')
        assert len(data) == dict(train=4138, val=465, test=1036)[split]
        return data

    def collate(self, items):
        if self.name in ('moment_detr', 'flashvtg'):
            metas, batch = self.data_module.start_end_collate(items)
            inputs, targets = self.data_module.prepare_batch_inputs(batch, self.device)
            if self.name == 'flashvtg':
                targets.update(label=metas, fps=torch.full((len(metas),), .5, device=self.device))
                inputs['targets'] = targets
        elif self.name == 'cg_detr':
            metas, inputs, targets = self.data_module.collate(items)
            inputs['targets'] = targets
        else:
            metas, inputs, targets = self.data_module.collate(items)
            inputs, targets = self.data_module.move(inputs, targets, self.device)
        inputs['durations'] = torch.tensor([meta['duration'] for meta in metas],
                                           dtype=torch.float32, device=self.device)
        return list(metas), inputs, targets

    def native_loss(self, native, targets, metas, epoch):
        if self.name == 'flashvtg':
            values = self.criterion((metas, None), native, targets)
        elif self.name == 'eatr':
            positive = targets['exist_label'].bool()
            indices = positive.nonzero().flatten().tolist()
            if indices:
                selected = {}
                for key, value in native.items():
                    if key == 'pseudo_event_spans':
                        selected[key] = [value[index] for index in indices]
                    elif key == 'aux_outputs':
                        selected[key] = [{name: item[positive] for name, item in row.items()} for row in value]
                    else:
                        selected[key] = value[positive]
                values = self.criterion(selected, {'span_labels': [targets['span_labels'][index] for index in indices]}, epoch)
            else:
                values = {}
            values['loss_exist'] = F.binary_cross_entropy_with_logits(
                native['pred_exist_logits'].reshape(-1), targets['exist_label'].reshape(-1))
        else:
            values = self.criterion(native, targets)
        weights = dict(self.criterion.weight_dict)
        if self.name == 'eatr':
            weights['loss_exist'] = self.cfg['exist_loss_coef']
        total = sum(value*weights[key] for key, value in values.items() if key in weights)
        assert torch.isfinite(total)
        return total, values

    def timestamps(self, inputs, durations):
        valid = inputs['src_vid_mask'].bool()
        index = torch.arange(valid.shape[1], device=valid.device, dtype=durations.dtype)[None]
        if self.grid_policy == 'normalized_tef':
            grid = (index+.5)*durations[:, None]/valid.sum(1, keepdim=True)
        else:
            left = index*self.opt.clip_length
            right = torch.minimum(left+self.opt.clip_length, durations[:, None])
            grid = (left+right)/2
            valid = valid & (left < durations[:, None])
        assert valid.any(1).all()
        return grid.masked_fill(~valid, 0), valid

    def export_ranked_proposals(self, windows, meta, apply_pat=None):
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
        result = value.tolist()
        if self.name == 'flashvtg' and self.opt.nms_thd != -1:
            from training.flash_vtg_gmr.inference import post_processing_mr_nms
            result = post_processing_mr_nms([dict(pred_relevant_windows=result)], self.opt.nms_thd,
                                             self.opt.max_before_nms, self.opt.max_after_nms,
                                             self.opt.nms_type)[0]['pred_relevant_windows']

        use_pat = self.config.get('enable_pat_select', True) if apply_pat is None else apply_pat
        if use_pat:
            sigma = self.config.get('pat_sigma', {}).get(self.name, 1.0)
            iou_th = self.config.get('pat_iou_thresh', {}).get(self.name, 0.3)
            result = apply_pat_select(result, sigma=sigma, iou_thresh=iou_th, top_k=self.config.get('top_k', 10))
        return result


def build_adapter(name, device, config=None):
    return Adapter(name, device, config)
