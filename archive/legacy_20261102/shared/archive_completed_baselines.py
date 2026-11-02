"""Archive existing frozen GMR baselines without changing active queue runs."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


HERE = Path(os.environ.get('GMR_RUNTIME_ROOT', Path(__file__).resolve().parent))
SOURCES = Path('/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder')
DESTINATION = HERE / 'reused_baselines'
EXPECTED = {
    'moment_detr': ('best.ckpt', 'c69fc1bdc6b7187c471bb0272cca5e7bb1087c175f10c646dad30fca1295ca1f'),
    'eatr': ('best.ckpt', '3d7f6f6cb8c62cfee30db2d48fee5d1e62e5d69604cc7ab484075ab4979606c0'),
    'qd_detr': ('best.ckpt', '03011c48168532bc8a99cee669d0d0489d0919f8fb127474107ed705e88b43c0'),
    'cg_detr': ('best.pt', '200f2f200bea257b1704f3bf285aecd50cf1722eb7fa7390e0763042f5a9b1d8'),
}
SIDECARS = (
    'effective_config.json', 'frozen_selection.json', 'frozen_checkpoint.json',
    'checkpoint_frozen.json', 'best_val_metrics.json', 'test_metrics.json',
    'final_test_metrics.json', 'test_predictions.jsonl', 'final_test_predictions.jsonl',
)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    records = []
    for backbone, (filename, expected) in EXPECTED.items():
        source = SOURCES / backbone / 'gmr'
        checkpoint = source / filename
        if digest(checkpoint) != expected:
            raise RuntimeError(f'Frozen checkpoint hash mismatch: {checkpoint}')
        target = DESTINATION / backbone
        target.mkdir(parents=True, exist_ok=True)
        copied = []
        for name in (filename, *SIDECARS):
            src, dst = source / name, target / name
            if not src.is_file():
                continue
            if dst.exists():
                if digest(src) != digest(dst):
                    raise RuntimeError(f'Refusing to overwrite different archive file: {dst}')
            else:
                shutil.copy2(src, dst)
            if digest(src) != digest(dst):
                raise RuntimeError(f'Archive verification failed: {dst}')
            copied.append(name)
        config = json.loads((source / 'effective_config.json').read_text())
        metric_file = 'final_test_metrics.json' if backbone == 'cg_detr' else 'test_metrics.json'
        brief = json.loads((source / metric_file).read_text())['brief']
        records.append({
            'backbone': backbone,
            'checkpoint': str((target / filename).relative_to(HERE)),
            'source_checkpoint': str(checkpoint),
            'checkpoint_sha256': expected,
            'checkpoint_format': 'original_native_model_and_optimizer',
            'source_seed': config.get('seed'),
            'source_learning_rate': config.get('lr'),
            'source_batch_size': config.get('bsz'),
            'test_mAP': brief['mAP'],
            'state_dict_schema_matches_current_baseline': True,
            'schema_check': 'all keys and shapes match after stripping base. and EaTR slot_atten.shared. wrapper names',
            'reuse_scope': 'historical_baseline_reference',
            'same_training_protocol_as_seed2023_ablation': False,
            'copied_files': copied,
        })
        print(backbone, target / filename, 'SHA256_OK', 'mAP', brief['mAP'])
    manifest = {
        'baselines': records,
        'active_runs_replaced': False,
        'queue_changed': False,
        'pending_unstarted_baseline_jobs_at_audit': [],
        'note': 'Historical baselines use seed2026/lr3e-5; current ablations use seed2023/lr5e-5. Preserve both protocols explicitly.',
    }
    (DESTINATION / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
