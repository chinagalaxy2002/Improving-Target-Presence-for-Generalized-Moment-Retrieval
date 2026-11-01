"""Random native factories and native losses/export only; no D-GMR modules."""
import sys, copy, json, importlib.util
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from paths import KIT, FEATURES
REFERENCE = KIT / 'vendor'
ASSETS = FEATURES
LABELS = Path('/disabled_source_dataset')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

class TargetAdapter:
    offset = 0
    final_index = 1
    grid_policy = 'clip_length'

    def __init__(self, name, device, dataset, seed=2023, config=None):
        assert dataset == 'charades' and seed == 2023
        self.name = name
        self.device = torch.device(device)
        self.dataset_name = dataset
        self.config = config or {}
        self.cfg = json.loads((KIT / 'configs' / f'{name}.json').read_text())
        self.cfg['clip_length'] = 2.0
        self.cfg['max_v_l'] = 128
        self.opt = SimpleNamespace(**dict(self.cfg, device=torch.device('cpu'), num_workers=0))
        self.opt.v_feat_dirs = []
        self.opt.t_feat_dir = None
        self.opt.train_path = None
        self.opt.eval_path = None
        source = REFERENCE / ('baseline_source' if name == 'flashvtg' else f'backends/{name}/source')
        sys.path.insert(0, str(source))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self._construct()
        if name == 'flashvtg':
            self.base.args.clip_length = 2.0
        self.base.requires_grad_(True).to(self.device)
        self.criterion.to(self.device)
        self.opt.device = self.device
        if name == 'flashvtg':
            self.criterion.device = self.device
        self.manifest = dict(initialization_protocol='random native seed2023; no model checkpoint loaded', dataset=dataset, backbone=name, random_seed=seed, feature_order=['clip', 'slowfast'] if name == 'moment_detr' else ['slowfast', 'clip'], native_config=self.cfg)

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
            self.data_module = module('d_gmr_qd_data', REFERENCE / 'backends/qd_detr/data.py')
            self.data_module.ASSETS, self.data_module.LABELS = (ASSETS, LABELS)
            self.offset, self.final_index, self.grid_policy = (1, 2, 'normalized_tef')
        elif self.name == 'eatr':
            from models.build_model import build_model
            from models.module import MLP
            self.base, self.criterion = build_model(self.opt)
            self.base.exist_head = MLP(256, 256, 1, 2)
            self.data_module = module('d_gmr_eatr_data', REFERENCE / 'backends/eatr/data.py')
            self.data_module.SOURCE = ASSETS.parent
        elif self.name == 'flashvtg':
            import nncore
            from models.flash_vtg_gmr.model import build_model1
            from training.flash_vtg_gmr import dataset
            self.opt.config = str(REFERENCE / 'baseline_source/configs/flash_vtg_gmr/model.py')
            self.opt.cfg = nncore.Config.from_file(self.opt.config)
            self.base, self.criterion = build_model1(self.opt)
            self.data_module = dataset
            self.final_index = 0
        else:
            raise KeyError(self.name)

    def collate(self, items):
        if self.name in ('moment_detr', 'flashvtg'):
            metas, batch = self.data_module.start_end_collate(items)
            inputs, targets = self.data_module.prepare_batch_inputs(batch, self.device)
            if self.name == 'flashvtg':
                targets.update(label=metas, fps=torch.full((len(metas),), 0.5, device=self.device))
                inputs['targets'] = targets
        else:
            metas, inputs, targets = self.data_module.collate(items)
            inputs, targets = self.data_module.move(inputs, targets, self.device)
        inputs['durations'] = torch.tensor([meta['duration'] for meta in metas], dtype=torch.float32, device=self.device)
        return (list(metas), inputs, targets)

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
            values['loss_exist'] = F.binary_cross_entropy_with_logits(native['pred_exist_logits'].reshape(-1), targets['exist_label'].reshape(-1))
        else:
            values = self.criterion(native, targets)
        weights = dict(self.criterion.weight_dict)
        if self.name == 'eatr':
            weights['loss_exist'] = self.cfg['exist_loss_coef']
        total = sum((value * weights[key] for key, value in values.items() if key in weights))
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite native losses: '+str({k:float(v) for k,v in values.items() if isinstance(v,torch.Tensor) and not torch.isfinite(v).all()}))
        return (total, values)

    def export_ranked_proposals(self, windows, meta, apply_pat=None):
        value = windows.detach().cpu().clone()
        duration = float(meta['duration'])
        value[:, :2] = (value[:, :2] / self.opt.clip_length).round() * self.opt.clip_length
        value[:, :2].clamp_(0, duration)
        for row in value:
            if row[1] <= row[0]:
                if row[1] > 0:
                    row[0] = max(0.0, float(row[1]) - min(self.opt.clip_length, duration))
                else:
                    row[0], row[1] = (0.0, min(self.opt.clip_length, duration))
        value = value[value[:, 2].argsort(descending=True, stable=True)]
        result = value.tolist()
        if self.name == 'flashvtg' and self.opt.nms_thd != -1:
            from training.flash_vtg_gmr.inference import post_processing_mr_nms
            result = post_processing_mr_nms([dict(pred_relevant_windows=result)], self.opt.nms_thd, self.opt.max_before_nms, self.opt.max_after_nms, self.opt.nms_type)[0]['pred_relevant_windows']
        return result
