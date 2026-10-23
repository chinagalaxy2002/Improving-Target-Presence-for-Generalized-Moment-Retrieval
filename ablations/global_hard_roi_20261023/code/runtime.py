"""Common initialization and active parameter selection for all eight ablations."""
import hashlib
import platform

from environment import ROOT, E0_CHECKPOINTS, digest, options, read_json, seed_all
import torch
from initialization import load_e0_weights
from model import LoRALinear, MCEVModel
from models.flash_vtg_gmr.model import build_model1


def build(job, device, initialize=True):
    assert job['presence_source']=='global_refine' and job['use_boundary_refine'] and job['use_lora'] and job['lora_rank']==8
    assert job['seed'] == 2026, 'This experiment is single-seed only.'
    seed_all(job['seed'])
    # Materialize the same tensors in the same order in every variant, including
    # disabled modules. NoLoRA retains an identity wrapper with scaling=0.
    materialized = dict(job, use_boundary_refine=True, use_lora=True,
                        use_moment_conditioning=True, use_global_branch=True)
    opt = options(job, device)
    base, _ = build_model1(opt)
    model = MCEVModel(base, materialized)
    checks = {}
    if initialize:
        baseline = torch.load(E0_CHECKPOINTS[job['seed']], map_location='cpu', weights_only=False)['model']
        checks = load_e0_weights(model, baseline)
        checks['tensor_sha256'] = {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                                   for name, value in model.state_dict().items()}
    model.job = dict(job)
    model.use_boundary_refine = job['use_boundary_refine']
    model.use_lora = job['use_lora']
    head = model.base.exist_head
    head.job = dict(job)
    head.use_boundary_refine = job['use_boundary_refine']
    for module in model.modules():
        if isinstance(module, LoRALinear) and not job['use_lora']:
            module.scaling = 0.0
    source = job['presence_source']
    assert source in ('global_only', 'global_refine', 'fused', 'local')
    for name, parameter in model.named_parameters():
        active = False
        if '.lora_' in name:
            active = job['use_lora']
        elif '.exist_head.' in name:
            part = name.split('.exist_head.', 1)[1].split('.')[0]
            if part in ('token', 'text_read'):
                active = True
            elif part in ('video_reads', 'global_head'):
                active = source in ('global_only', 'global_refine', 'fused')
            elif part == 'extractor':
                active = source in ('global_refine', 'fused', 'local')
            elif part in ('moment_cross_attn', 'moment_match_mlp', 'local_head'):
                active = source in ('fused', 'local')
            elif part == 'fusion_gate':
                active = source == 'fused'
            elif part == 'plurality':
                active = job['multi_weight'] > 0
            elif part == 'refiner':
                active = job['use_boundary_refine'] and source != 'global_only'
        parameter.requires_grad_(active)
    return model.to(opt.device), opt, checks


def provenance(job):
    files = list((ROOT / 'code').glob('*.py')) + list((ROOT / 'baseline_source').rglob('*.py'))
    files += list((ROOT / 'configs').glob('*.json')) + [ROOT / 'data/train.jsonl', ROOT / 'data/val.jsonl']
    return {'init_protocol': 'correct_init_v1', 'seed': job['seed'],
            'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'checkpoint': str(E0_CHECKPOINTS[job['seed']]), 'checkpoint_sha256': digest(E0_CHECKPOINTS[job['seed']]),
            'files_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}}


def training_mode(model):
    model.train()
    # Preserve the original training protocol for nn.Dropout / BatchNorm.
    trainable = {m for m in model.modules() if any(p.requires_grad for p in m.parameters(recurse=False))}
    for module in model.base.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm)) and module not in trainable:
            module.eval()


def selection_score(report):
    spectrum = sum(report[key]['G-mIoU@1'] for key in ['gate_04', 'fixed_05', 'gate_06']) / 3
    minimum = min(report['fixed_05']['positive_recall'], report['fixed_05']['negative_recall'])
    return spectrum + .4 * report['val_G_threshold']['G-mIoU@1'] + .5 * report['raw_localizer_pre_exist']['MR-full-mAP'] + .1 * minimum
