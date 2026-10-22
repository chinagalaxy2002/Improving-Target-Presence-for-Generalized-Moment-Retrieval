"""Moment-DETR GMR checkpoint -> frozen backbone plus reference S4 modules."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from easydict import EasyDict
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from models.moment_detr_gmr.moment_detr import build_model
from training.moment_detr_gmr.dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from model import S4MomentDETR
from s4_reference import compute_mcev_loss

SOURCE = Path('/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/moment_detr/gmr/best.ckpt')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def initialize(device, job, snapshot=None):
    directory = ROOT / 'snapshots'; directory.mkdir(exist_ok=True)
    source = Path(snapshot or SOURCE).resolve()
    temporary = directory / f'.capture_{os.getpid()}.tmp'
    for _ in range(5):
        with source.open('rb') as src, temporary.open('wb') as dst:
            before = os.fstat(src.fileno()); shutil.copyfileobj(src, dst); after = os.fstat(src.fileno())
        digest = sha256(temporary)
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and sha256(source) == digest:
            break
    else:
        raise RuntimeError('Baseline changed during snapshot')
    path = directory / f'{digest}.ckpt'
    if path.exists():
        assert sha256(path) == digest
        temporary.unlink()
    else:
        temporary.replace(path); path.chmod(0o444)
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state.get('epoch', -1) >= 0 and state['opt']['use_exist_head'] is True
    assert any(n.startswith('exist_head.') for n in state['model'])
    options = EasyDict(state['opt']); options.device = torch.device('cpu'); options.num_workers = 0
    base, _ = build_model(options)
    base.load_state_dict(state['model'], strict=True)
    model = S4MomentDETR(base, job)
    preserved = 0
    actual = model.base.state_dict()
    for name, value in state['model'].items():
        mapped = name
        if name.startswith('transformer.encoder.layers.') and ('.linear1.' in name or '.linear2.' in name):
            prefix, leaf = name.rsplit('.', 1); mapped = prefix + '.base_linear.' + leaf
        assert torch.equal(value, actual[mapped]), name
        preserved += 1
    counts = {key: sum(p.numel() for n, p in model.named_parameters() if predicate(n, p)) for key, predicate in {
        'total': lambda n,p: True, 'trainable': lambda n,p: p.requires_grad,
        'frozen': lambda n,p: not p.requires_grad, 'lora': lambda n,p: '.lora_' in n}.items()}
    manifest = {'source_checkpoint': str(SOURCE), 'snapshot': str(path), 'sha256': digest,
                'verified_gmr_baseline': True, 'baseline_epoch': state['epoch'], 'baseline_config': state['opt'],
                'counts': counts, 'strict_pretrained_tensors_preserved': preserved,
                'fresh_optimizer_scheduler': True}
    manifest_path = directory / f'{digest}.json'
    if not manifest_path.exists(): manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    config_path = directory / 'baseline_config.json'
    if not config_path.exists(): config_path.write_text(json.dumps(state['opt'], indent=2, default=str) + '\n')
    return model.to(device), manifest


def get_dataset(split):
    assert split in ('train', 'val', 'test')
    options = json.loads((ROOT / 'snapshots' / 'baseline_config.json').read_text())
    path = options['train_path' if split == 'train' else 'eval_path']
    if split == 'test': path = str(Path(path).with_name('test.jsonl'))
    data = StartEndDataset(dset_name=options['dset_name'], domain=None, data_path=path,
        v_feat_dirs=options['v_feat_dirs'], q_feat_dir=options['t_feat_dir'], max_q_l=options['max_q_l'],
        max_v_l=options['max_v_l'], clip_len=options['clip_length'], max_windows=options['max_windows'],
        mr_only=True, keep_empty_gt=True, load_labels=True)
    assert len(data) == sum(bool(line.strip()) for line in Path(path).read_text().splitlines())
    return data


def collate(items, device):
    metas, tensors = start_end_collate(items)
    inputs, _ = prepare_batch_inputs(tensors, device)
    inputs['durations'] = torch.tensor([m['duration'] for m in metas], device=device)
    return metas, inputs


def loss(output, metas, job):
    targets = {'exist_label': output['pred_exist_logits'].new_tensor([bool(m['relevant_windows']) for m in metas])}
    return compute_mcev_loss(output, metas, targets, job)
