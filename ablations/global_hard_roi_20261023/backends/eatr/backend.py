"""Trainer API for EaTR + S4 minus fusion, initialized from trained GMR."""
import json
from pathlib import Path
import torch
from runtime import ROOT, DEFAULT_SOURCE, build, snapshot as snapshot_baseline, digest
from model import loss_for
import data


def initialize(device, job, snapshot=None):
    config = json.loads((ROOT / 'config.json').read_text())
    # The complete saved job restores S4 behavior when evaluating a checkpoint.
    if isinstance(job, dict):
        config.update({key: job[key] for key in config if key in job})
    source = job.get('source_checkpoint') if isinstance(job, dict) else None
    checkpoint = Path(snapshot) if snapshot else snapshot_baseline(source or DEFAULT_SOURCE, 'gmr')
    model, checks = build(device, checkpoint, config)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    baseline_config = json.loads(json.dumps(state['config'], default=str))
    manifest = checks['initialization']
    assert checks['verified_gmr_baseline'] and baseline_config['use_exist_head']
    initialization = {
        **checks,
        'source_checkpoint': manifest['source_checkpoint'],
        'snapshot': str(checkpoint),
        'sha256': digest(checkpoint),
        'verified_gmr_baseline': True,
        'baseline_config': baseline_config,
        's4_config': config,
        'baseline_epoch': state['epoch'],
        'native_existence_head_policy': 'loaded strictly, retained frozen; S4 independent global prediction overrides output',
        'optimizer_policy': 'fresh optimizer and scheduler',
    }
    return model, initialization


def get_dataset(split):
    return data.Soccer(split)


def collate(items, device):
    metadata, inputs, _ = data.collate(items)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    inputs['durations'] = torch.tensor([item['duration'] for item in metadata], device=device)
    return metadata, inputs


def loss(output, metas, job):
    config = json.loads((ROOT / 'config.json').read_text())
    if isinstance(job, dict):
        config.update({key: job[key] for key in config if key in job})
    return loss_for(output, metas, config)
