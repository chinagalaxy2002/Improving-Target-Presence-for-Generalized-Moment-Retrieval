"""E3: native factory initialization with all weight-restoration APIs prohibited."""
import contextlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from adapters import Adapter
from common import ROOT, REPO, OLD, ASSETS, module

@contextlib.contextmanager
def forbid_pretrained_restore():
    """A constructor that attempts implicit checkpoint/hub restoration must fail."""
    attempts = []
    def denied(*args, **kwargs):
        attempts.append(str(args[0])[:200] if args else 'keyword-only restore')
        raise RuntimeError('E3 construction forbids any checkpoint/weight restore')
    with contextlib.ExitStack() as stack:
        for owner, name in [(torch,'load'), (torch.jit,'load'), (torch.hub,'load'),
                            (torch.hub,'load_state_dict_from_url'), (nn.Module,'load_state_dict')]:
            stack.enter_context(patch.object(owner, name, denied))
        yield attempts

class UnusedNativePresence(nn.Module):
    """Stateless forward-API placeholder. No parameters, BCE, or exported score."""
    def forward(self, x):
        return x.new_zeros((x.shape[0],))


def subset(value, mask, key=None):
    ids = mask.nonzero().flatten().tolist()
    if key == 'global_rep_tokens': return value
    if key in ('span_labels','pseudo_event_spans','label'): return [value[i] for i in ids]
    if torch.is_tensor(value) and value.ndim and value.shape[0] == len(mask): return value[mask]
    if isinstance(value, dict): return {k:subset(v,mask,k) for k,v in value.items()}
    if isinstance(value, list): return [subset(v,mask) for v in value]
    if isinstance(value, tuple): return tuple(subset(v,mask) for v in value)
    return value

class ScratchAdapter(Adapter):
    def __init__(self, name, device):
        # Intentionally do not call Adapter.__init__, source(), or torch.load.
        self.name, self.device = name, device
        config_path = ROOT/f'configs/E3/{name}.json'
        record = json.loads(config_path.read_text())
        self.cfg = record['native_factory_config']
        self.opt = SimpleNamespace(**dict(self.cfg, device=torch.device('cpu')))
        self.opt.v_feat_dirs = [str(ASSETS/k) for k in ('slowfast','clip')]
        self.opt.t_feat_dir = str(ASSETS/'clip_text')
        self.opt.train_path = str(ROOT/'data/train.jsonl'); self.opt.eval_path = str(ROOT/'data/val.jsonl')
        self.opt.num_workers = 0
        sys.path.insert(0, str(OLD/('baseline_source' if name == 'flashvtg' else f'backends/{name}/source')))
        with forbid_pretrained_restore() as attempts:
            if name == 'moment_detr':
                from easydict import EasyDict
                from models.moment_detr_gmr.moment_detr import build_model
                from training.moment_detr_gmr import dataset
                self.base, self.criterion = build_model(EasyDict(vars(self.opt)))
                self.data_module = dataset
            elif name == 'qd_detr':
                from qd_detr.model import build_model
                self.base, wrapped = build_model(self.opt)
                self.criterion = wrapped.native  # Remove wrapper's native presence BCE.
                self.data_module = module('scratch_qd_data', OLD/'backends/qd_detr/data.py')
                self.data_module.ASSETS = ASSETS; self.data_module.LABELS = ROOT/'data'
            elif name == 'cg_detr':
                from cg_detr.model import build_model
                self.base, self.criterion = build_model(self.opt)
                self.native = module('scratch_cg_training', REPO/'backbonefolder/cg_detr/train_gmr.py')
                self.criterion.__class__ = self.native.SoccerCriterion
                # Explicit no-distillation protocol, including native self-distillation.
                self.criterion.losses = [x for x in self.criterion.losses if x != 'distill']
                self.criterion.weight_dict = {k:v for k,v in self.criterion.weight_dict.items() if 'distill' not in k}
                self.data_module = self.native; self.data_module.GMR = ASSETS.parent
            elif name == 'eatr':
                from models.build_model import build_model
                self.base, self.criterion = build_model(self.opt)
                self.data_module = module('scratch_eatr_data', OLD/'backends/eatr/data.py')
                self.data_module.SOURCE = ASSETS.parent
            else:
                import nncore
                from models.flash_vtg_gmr.model import build_model1
                from training.flash_vtg_gmr import dataset
                self.opt.config = str(OLD/'baseline_source/configs/flash_vtg_gmr/model.py')
                self.opt.cfg = nncore.Config.from_file(self.opt.config)
                self.base, self.criterion = build_model1(self.opt)
                self.data_module = dataset
            # Official native factories above perform their own random initialization.
            # No learned native video-level presence module remains in E3.
            self.removed_presence_parameters = [n for n,p in self.base.named_parameters() if n.startswith('exist_head.')]
            self.base.exist_head = UnusedNativePresence()
            self.criterion.losses = [x for x in self.criterion.losses if x != 'exist']
            self.criterion.weight_dict = {k:v for k,v in self.criterion.weight_dict.items() if not k.startswith('loss_exist')}
        assert attempts == []
        self.base.requires_grad_(True); self.base.to(device); self.criterion.to(device)
        self.opt.device = torch.device(device)
        if name == 'cg_detr': self.criterion.args.device = self.opt.device
        if name == 'flashvtg': self.criterion.device = self.opt.device
        self.bs = int(self.cfg['bsz'])
        self.manifest = dict(initialization='official native random initialization; GEQ random',checkpoint=None,
             pretrained_restore_attempts=attempts,weight_restore_guard=True,baseline_config=self.cfg,
             effective_options=vars(self.opt),native_weight_dict=self.criterion.weight_dict,batch_size=self.bs,
             removed_native_presence_parameters=self.removed_presence_parameters,native_presence_bce=False,
             geq_only_video_presence_bce=True,native_distillation=False,
             e1_comparison_caveat='E1 also supervises native presence BCE; E3 removes it. CG E3 additionally disables native self-distillation. Not a pure initialization ablation.',
             external_features='Existing offline CLIP/SlowFast features retained; encoders outside computation graph')

    def native_loss(self, native, targets, meta, epoch):
        # Native localization objectives and original positive-query MR policies.
        if self.name == 'flashvtg':
            ld = self.criterion((meta,None),native,targets)
        elif self.name in ('qd_detr','cg_detr','eatr'):
            positive = native['pred_spans'].new_tensor([bool(m['relevant_windows']) for m in meta],dtype=torch.bool)
            if positive.any():
                selected = subset(native, positive)
                selected_targets = subset(targets, positive)
                if self.name == 'eatr': ld = self.criterion(selected, selected_targets, epoch)
                else: ld = self.criterion(selected, selected_targets)
            else:
                zero = native['pred_spans'].sum()*0 + native['pred_logits'].sum()*0
                ld = {k:zero for k in self.criterion.weight_dict}
        else: ld = self.criterion(native,targets)
        ld = {k:v for k,v in ld.items() if 'loss' in k}
        assert not any('exist' in k or 'distill' in k for k in ld)
        total = sum(v*self.criterion.weight_dict[k] for k,v in ld.items() if k in self.criterion.weight_dict)
        return total, ld
