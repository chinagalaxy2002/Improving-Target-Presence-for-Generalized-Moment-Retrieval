"""One backbone per process: reuse native factories, data transforms and losses."""
import json, sys
from types import SimpleNamespace
import torch
from torch.nn import functional as F
from common import ROOT, REPO, OLD, ASSETS, module, source, sha

# Load the repository's frozen official evaluator before native trainer imports.
sys.path.insert(0, str(OLD))
from official_eval.eval_main import evaluate_gmr as _official_evaluate

class Adapter:
    def __init__(self, name, device):
        self.name, self.device = name, device
        self.path = source(name); digest = sha(self.path)
        expected = json.loads((OLD/f'results/{name}/gmr/CONFIG.json').read_text())['initialization']['sha256']
        assert digest == expected, 'Checkpoint differs from the verified original E0 source'
        state = torch.load(self.path, map_location='cpu', weights_only=False)
        assert state.get('epoch', -1) >= 0
        cfg = state.get('opt', state.get('config'))
        if name == 'flashvtg': cfg = json.loads((OLD/'configs/baseline_protocol.json').read_text())
        self.cfg = dict(cfg) if isinstance(cfg, dict) else vars(cfg).copy()
        build_cfg = dict(self.cfg, device=torch.device('cpu'))
        self.opt = SimpleNamespace(**build_cfg)
        self.opt.v_feat_dirs = [str(ASSETS/k) for k in ('slowfast','clip')]
        self.opt.t_feat_dir = str(ASSETS/'clip_text')
        self.opt.train_path = str(ROOT/'data/train.jsonl'); self.opt.eval_path = str(ROOT/'data/val.jsonl')
        self.opt.num_workers = 0
        base_path = OLD/('baseline_source' if name == 'flashvtg' else f'backends/{name}/source')
        sys.path.insert(0, str(base_path))
        if name == 'moment_detr':
            from easydict import EasyDict
            from models.moment_detr_gmr.moment_detr import build_model
            from training.moment_detr_gmr import dataset
            self.base, self.criterion = build_model(EasyDict(vars(self.opt)))
            self.data_module = dataset
        elif name == 'qd_detr':
            from qd_detr.model import build_model
            self.base, self.criterion = build_model(self.opt)
            self.data_module = module('native_qd_data', OLD/'backends/qd_detr/data.py')
            self.data_module.ASSETS = ASSETS; self.data_module.LABELS = ROOT/'data'
        elif name == 'cg_detr':
            from cg_detr.model import build_model
            self.base, self.criterion = build_model(self.opt)
            # Exact locally reconstructed GMR training criterion, including CG
            # native saliency/distillation/orthogonality and positive-only MR.
            self.native = module('native_cg_training', REPO/'backbonefolder/cg_detr/train_gmr.py')
            self.criterion.__class__ = self.native.GMRCriterion
            self.criterion.weight_dict['loss_exist'] = self.cfg['exist_loss_coef']
            self.data_module = self.native
            self.data_module.GMR = ASSETS.parent
        elif name == 'eatr':
            from models.build_model import build_model
            from models.module import MLP
            self.base, self.criterion = build_model(self.opt)
            self.base.exist_head = MLP(256, 256, 1, 2)
            self.data_module = module('native_eatr_data', OLD/'backends/eatr/data.py')
            self.data_module.SOURCE = ASSETS.parent
        else:
            import nncore
            from models.flash_vtg_gmr.model import build_model1
            from training.flash_vtg_gmr import dataset
            self.opt.config = str(OLD/'baseline_source/configs/flash_vtg_gmr/model.py')
            self.opt.cfg = nncore.Config.from_file(self.opt.config)
            self.base, self.criterion = build_model1(self.opt)
            self.data_module = dataset
        weights = state['model']
        if name == 'flashvtg':
            assert all(k.startswith('base.') for k in weights)
            weights = {k[len('base.'):]: v for k,v in weights.items()}
        self.base.load_state_dict(weights, strict=True)
        assert all(torch.equal(v, self.base.state_dict()[k]) for k,v in weights.items())
        assert sha(self.path) == digest
        self.base.requires_grad_(True)
        self.base.to(device); self.criterion.to(device)
        self.opt.device = torch.device(device)
        if name == 'flashvtg': self.criterion.device = self.opt.device
        self.bs = int(self.cfg['bsz'])
        self.manifest = dict(checkpoint=str(self.path), sha256=digest, epoch=state['epoch'],
                             strict_load=True, equal_tensors=len(weights), baseline_config=self.cfg,
                             native_weight_dict=self.criterion.weight_dict, batch_size=self.bs,
                             external_features='precomputed CLIP/SlowFast; encoders outside computation graph')
    def dataset(self, split):
        if self.name in ('moment_detr','flashvtg'):
            kw = dict(dset_name=self.opt.dset_name, data_path=str(ROOT/f'data/{split}.jsonl'),
                      v_feat_dirs=self.opt.v_feat_dirs, q_feat_dir=self.opt.t_feat_dir,
                      max_q_l=self.opt.max_q_l, max_v_l=self.opt.max_v_l,
                      clip_len=self.opt.clip_length, max_windows=self.opt.max_windows,
                      span_loss_type=self.opt.span_loss_type, ctx_mode=self.opt.ctx_mode,
                      mr_only=True, keep_empty_gt=True, load_labels=True)
            if self.name == 'moment_detr': kw['domain'] = None
            ds = self.data_module.StartEndDataset(**kw)
        else: ds = self.data_module.Soccer(split)
        assert len(ds) == {'train':4138,'val':465,'test':1036}[split]
        return ds
    def collate(self, items):
        if self.name in ('moment_detr','flashvtg'):
            meta, batch = self.data_module.start_end_collate(items)
            inputs, targets = self.data_module.prepare_batch_inputs(batch, self.device)
            if self.name == 'flashvtg':
                targets.update(label=meta, fps=torch.full((len(meta),), .5, device=self.device))
                inputs['targets'] = targets
        elif self.name == 'cg_detr':
            meta, inputs, targets = self.data_module.collate(items)
            inputs['targets'] = targets
        else:
            meta, inputs, targets = self.data_module.collate(items)
            inputs, targets = self.data_module.move(inputs, targets, self.device)
        inputs['durations'] = torch.tensor([m['duration'] for m in meta], device=self.device)
        return list(meta), inputs, targets
    def native_loss(self, native, targets, meta, epoch):
        if self.name == 'flashvtg': ld = self.criterion((meta, None), native, targets)
        elif self.name == 'eatr':
            positive = targets['exist_label'].bool(); ids = positive.nonzero().flatten().tolist()
            if ids:
                selected = {}
                for k,v in native.items():
                    if k == 'pseudo_event_spans': selected[k] = [v[i] for i in ids]
                    elif k == 'aux_outputs': selected[k] = [{a:b[positive] for a,b in x.items()} for x in v]
                    else: selected[k] = v[positive]
                ld = self.criterion(selected, {'span_labels':[targets['span_labels'][i] for i in ids]}, epoch)
            else: ld = {}
            ld['loss_exist'] = F.binary_cross_entropy_with_logits(native['pred_exist_logits'], targets['exist_label'])
        else: ld = self.criterion(native, targets)
        weights = dict(self.criterion.weight_dict)
        if self.name == 'eatr': weights['loss_exist'] = self.cfg['exist_loss_coef']
        total = sum(v*weights[k] for k,v in ld.items() if k in weights)
        return total, ld
