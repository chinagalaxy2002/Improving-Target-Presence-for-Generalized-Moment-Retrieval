"""Isolated native QD-DETR-GMR -> S4 minus Fusion initialization and data API."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'source'))
from qd_detr.model import build_model
from model import S4QDDETR, loss_for
from data import Soccer, collate as native_collate

SOURCE = Path('/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/qd_detr/gmr/best.ckpt')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _snapshot(source):
    directory = ROOT / 'snapshots'
    directory.mkdir(exist_ok=True)
    temp = directory / f'.snapshot-{os.getpid()}.tmp'
    for _ in range(5):
        with source.open('rb') as original, temp.open('wb') as target:
            before = os.fstat(original.fileno())
            for block in iter(lambda: original.read(1024 * 1024), b''):
                target.write(block)
            after = os.fstat(original.fileno())
        checksum = digest(temp)
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns) and digest(source) == checksum:
            break
        temp.unlink(missing_ok=True)
    else:
        raise RuntimeError('Baseline checkpoint changed during snapshot')
    final = directory / f'gmr-{checksum}.ckpt'
    if final.exists():
        assert digest(final) == checksum
        temp.unlink()
    else:
        temp.replace(final)
        final.chmod(0o444)
    return final


def initialize(device, job, snapshot=None):
    """Load all trained GMR tensors strictly before introducing S4 and LoRA."""
    checkpoint = Path(snapshot).resolve() if snapshot else _snapshot(SOURCE)
    checksum = digest(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert digest(checkpoint) == checksum, 'Checkpoint changed while loading'
    baseline_config = state['config']
    assert baseline_config['variant'] == 'gmr' and state['epoch'] >= 0
    assert all(key in state['model'] for key in ('exist_head.mlp.0.weight', 'exist_head.mlp.2.weight'))
    opt = SimpleNamespace(**baseline_config)
    opt.device = 'cpu'
    baseline, _ = build_model(opt)
    baseline.load_state_dict(state['model'], strict=True)
    before = {key: value.clone() for key, value in baseline.state_dict().items()}
    model = S4QDDETR(baseline, job)
    after = model.base.state_dict()
    for key, value in before.items():
        mapped = key
        if key.startswith('transformer.encoder.layers.') and ('.linear1.' in key or '.linear2.' in key):
            stem, leaf = key.rsplit('.', 1)
            mapped = f'{stem}.base_linear.{leaf}'
        assert torch.equal(value, after[mapped]), f'Pretrained tensor changed: {key}'
    assert all(torch.count_nonzero(value) == 0 for name, value in model.named_parameters() if name.endswith('lora_B'))
    assert all(not value.requires_grad for name, value in model.named_parameters() if name.startswith('base.') and '.lora_' not in name)
    counts = {
        'total': sum(value.numel() for value in model.parameters()),
        'trainable': sum(value.numel() for value in model.parameters() if value.requires_grad),
        'frozen': sum(value.numel() for value in model.parameters() if not value.requires_grad),
        'lora': sum(value.numel() for name, value in model.named_parameters() if '.lora_' in name),
    }
    initialization = {
        'source_checkpoint': str(SOURCE), 'snapshot': str(checkpoint), 'sha256': checksum,
        'verified_gmr_baseline': True, 'baseline_config': baseline_config, 'counts': counts,
        'strict_base_load': True, 'preserved_tensors': len(before), 'epoch': state['epoch'],
        'optimizer_policy': 'fresh optimizer and scheduler; baseline state ignored',
        'frozen_gmr_head': 'original exist_head preserved and frozen; S4 replaces output only',
    }
    return model.to(device), initialization


def get_dataset(split):
    assert split in ('train', 'val', 'test'), 'Test is enabled only for user-requested fixed-epoch plotting'
    return Soccer(split)


def collate(items, device):
    metadata, inputs, _ = native_collate(items)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    inputs['durations'] = torch.tensor([float(item['duration']) for item in metadata], device=device)
    return metadata, inputs


def loss(output, metas, job):
    return loss_for(output, metas, job)
