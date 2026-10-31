"""Install the completed native EaTR-GMR baseline into the current experiment."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# Keep the sibling queue.py from shadowing Python's standard-library queue.
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == SCRIPT_DIR:
    sys.path.pop(0)
import torch

sys.path.insert(0, SCRIPT_DIR)
from execution_plan import HERE, assert_not_running, plan


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, default=str) + '\n')


def main():
    source = HERE / 'reused_baselines' / 'eatr'
    run = HERE / 'runs' / 'eatr' / 'baseline'
    marker = run / 'reused_baseline.json'
    if marker.exists():
        print('ALREADY_REUSED', marker)
        return
    assert_not_running('eatr', 'baseline')
    raw = source / 'best.ckpt'
    source_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    expected = '3d7f6f6cb8c62cfee30db2d48fee5d1e62e5d69604cc7ab484075ab4979606c0'
    if source_sha != expected:
        raise RuntimeError(f'Source checkpoint hash mismatch: {raw}')
    state = torch.load(raw, map_location='cpu', weights_only=False)
    weights = {}
    for name, tensor in state['model'].items():
        if name.startswith('transformer.decoder.slot_atten.'):
            name = name.replace('transformer.decoder.slot_atten.', 'transformer.decoder.slot_atten.shared.', 1)
        weights['base.' + name] = tensor
    existing = run / 'best.ckpt'
    if existing.exists():
        current = torch.load(existing, map_location='cpu', weights_only=False)['model']
        if set(current) != set(weights) or any(current[k].shape != weights[k].shape for k in current):
            raise RuntimeError('Reused EaTR model does not match current baseline architecture')
    archive = HERE / 'archived_runs' / plan()['revision'] / 'eatr' / 'baseline'
    if run.exists():
        if archive.exists():
            raise RuntimeError(f'Refusing to overwrite existing archive: {archive}')
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run), str(archive))
    run.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model': weights,
        'epoch': state['epoch'],
        'best': state['best'],
        'inference_only': True,
        'source_checkpoint': str(raw),
        'source_checkpoint_sha256': source_sha,
        'source_seed': 2026,
        'note': 'Native weight tensors unchanged; only MultiRouteModel wrapper names were added.',
    }
    temporary = run / 'best.tmp'
    torch.save(checkpoint, temporary)
    os.replace(temporary, run / 'best.ckpt')
    checkpoint_sha = hashlib.sha256((run / 'best.ckpt').read_bytes()).hexdigest()
    metrics = json.loads((source / 'test_metrics.json').read_text())
    metrics['G-mIoU_0_1'] = {k: float(v) / 100 for k, v in metrics['brief'].items() if k.startswith('G-mIoU@')}
    metrics['brief_0_1'] = {k: float(v) / 100 if k.startswith('G-mIoU@') else v for k, v in metrics['brief'].items()}
    dump(run / 'test_metrics.json', metrics)
    shutil.copy2(source / 'test_predictions.jsonl', run / 'test_predictions.jsonl')
    shutil.copy2(source / 'best_val_metrics.json', run / 'best_val_metrics.json')
    source_config = json.loads((source / 'effective_config.json').read_text())
    dump(run / 'source_effective_config.json', source_config)
    dump(run / 'experiment_config.json', {
        'backbone': 'eatr', 'variant': 'baseline', 'seed': source_config['seed'],
        'learning_rate': source_config['lr'], 'microbatch': source_config['bsz'],
        'gradient_accumulation': 1, 'effective_batch': source_config['bsz'],
        'epochs': source_config['n_epoch'], 'num_workers': source_config['num_workers'],
        'selection': source_config['checkpoint_selection'], 'reused_completed_baseline': True,
        'source_config': 'source_effective_config.json',
    })
    provenance = {
        'source': str(raw), 'source_sha256': source_sha,
        'wrapper_checkpoint_sha256': checkpoint_sha,
        'seed': 2026, 'bsz': 32, 'test_mAP': metrics['brief']['mAP'],
        'archived_previous_baseline': str(archive),
        'existing_frozen_test_predictions_reused': True,
        'same_seed_as_other_route_experiments': False,
    }
    dump(marker, provenance)
    dump(run / 'frozen_checkpoint.json', {
        'path': str(run / 'best.ckpt'), 'sha256': checkpoint_sha,
        'best_val_map': state['best'], 'source_checkpoint_sha256': source_sha,
    })
    dump(run / 'status.json', {
        'status': 'COMPLETED', 'reused_completed_baseline': True,
        'best_val_map': state['best'], 'checkpoint_sha256': checkpoint_sha,
        'test': metrics['brief'], 'provenance': 'reused_baseline.json',
    })
    print('EATR_BASELINE_REUSED', run, 'mAP', metrics['brief']['mAP'], 'seed', source_config['seed'])


if __name__ == '__main__':
    main()
