"""Reuse only byte-identical, completed, protocol-matched baseline/global cells."""
import json
import shutil
from pathlib import Path
from run import sha, dump

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'component_ablation_20261023'
BACKBONES = ['moment_detr', 'qd_detr', 'cg_detr', 'eatr', 'flashvtg']


def main():
    reused = []
    for backbone in BACKBONES:
        for setting, row in [('gmr', 1), ('global_only', 2)]:
            source = OLD / 'runs' / backbone / f'row_{row:02d}'
            complete = json.loads((source / 'COMPLETE.json').read_text())
            effective = json.loads((source / 'EFFECTIVE_CONFIG.json').read_text())
            assert complete['status'] == 'complete' and complete['frozen_unchanged']
            assert complete['epochs'] == (0 if row == 1 else 4)
            assert effective['passed'] and effective['frozen_parameters_exactly_preserved'] > 0
            flags = effective['effective']
            assert flags['global_event_query'] == (row == 2)
            assert not any(flags[k] for k in ['lora', 'boundary_refiner', 'saliency_6d', 'candidate_presence_fusion'])
            config = json.loads((source / 'CONFIG.json').read_text())
            assert config['job']['seed'] == 2026
            for split in ['train', 'val', 'test']:
                assert sha(ROOT / 'data' / f'{split}.jsonl') == sha(OLD / 'data' / f'{split}.jsonl')
            target = ROOT / 'runs' / backbone / setting
            target.mkdir(parents=True, exist_ok=False)
            files = ['CONFIG.json', 'PREFLIGHT.json', 'EFFECTIVE_CONFIG.json', 'val_predictions.jsonl',
                     'test_predictions.jsonl', 'val_metrics.json', 'test_metrics.json']
            if row == 2:
                files += ['history.jsonl', 'CHECKPOINT.json']
                history = [json.loads(s) for s in (source / 'history.jsonl').read_text().splitlines()]
                assert [h['epoch'] for h in history] == [1, 2, 3, 4]
                metadata = json.loads((source / 'CHECKPOINT.json').read_text())
                assert metadata['epoch'] == 4 and sha(source / 'epoch_004.ckpt') == metadata['sha256']
            manifest = {name: sha(source / name) for name in files}
            for name in files:
                shutil.copy2(source / name, target / name)
                assert sha(target / name) == manifest[name]
            record = dict(backbone=backbone, setting=setting, original_row=row,
                          source=str(source), copied_files_sha256=manifest,
                          original_checkpoint=str(source / 'epoch_004.ckpt') if row == 2 else config['initialization']['snapshot'],
                          reason='Identical pretrained checkpoint, disabled components, seed, training budget, data and exporter')
            dump(target / 'REUSE.json', record)
            dump(target / 'COMPLETE.json', dict(complete, setting=setting, reused=True))
            reused.append(record)
    dump(ROOT / 'reports/REUSE_AUDIT.json', dict(passed=True, reused_cells=10, records=reused))


if __name__ == '__main__':
    main()
