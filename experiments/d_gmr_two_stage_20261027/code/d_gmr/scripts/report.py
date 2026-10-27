"""Generate result tables for D-GMR only from completed evaluations."""
import argparse
import csv
import fcntl
import json
from pathlib import Path

from d_gmr.common import BACKBONES, RUNS, dump

KEYS = ('mAP', 'mR@1', 'mR@5', 'mR+@5', 'AUROC', 'Rej-F1@0.4',
        'G-mIoU@1', 'G-mIoU@3', 'N-acc')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(RUNS/'RESULTS.md'))
    args = parser.parse_args()
    guard = (RUNS/'report.lock').open('a+')
    fcntl.flock(guard, fcntl.LOCK_EX)
    lines = ['# D-GMR two-stage: real experiment results', '',
        'One fixed configuration, seed 2026 and four epochs per backbone. Missing values remain pending.', '',
        '## Protocol', '',
        'Each model starts from its strict native GMR checkpoint. The E1 epoch-4 GEQ is loaded and frozen. '
        'TEM/TECR use seed-2026 initialization. Native GMR and TEM/TECR use learning rates 1e-5 and 1e-4.', '',
        'Loss = native GMR + quality + intra_ranking + 0.5 inter_ranking + 0.5 set + 0.5 evidence; GEQ BCE weight is zero. '
        'Inference applies Plurality-Aware Temporal Selection (PAT-Select) for overlap-based score decay.', '',
        '## Parameters and elapsed time', '',
        '| Backbone | Native GMR | GEQ | TEM/TECR | Total | Train seconds / epoch |',
        '|---|---:|---:|---:|---:|---|']
    checkpoint_rows, metric_rows = [], []
    for backbone in BACKBONES:
        run = RUNS/backbone
        config = read(run/'config.json')
        logs = [read(path) for path in sorted((run/'logs').glob('epoch_*.json'))]
        if config:
            times = ', '.join(f"{value['training_seconds']:.2f}" for value in logs) or 'pending'
            lines.append(f"| {backbone} | {config['native_parameter_count']} | "
                         f"{config['geq_parameter_count']} | {config['added_parameter_count']} | "
                         f"{config['total_parameter_count']} | {times} |")
        else:
            lines.append(f'| {backbone} | pending | pending | pending | pending | pending |')
        for path in sorted((run/'checkpoints').glob('epoch_*.pth')):
            checkpoint_rows.append(dict(backbone=backbone, path=str(path.resolve()),
                                        bytes=path.stat().st_size,
                                        recovery_before_validation='_preval' in path.stem))
    lines += ['', 'Counts are parameters, not FLOPs. Timing includes data IO and native forward/backward.', '',
              '## Every epoch: Val', '',
              '| Backbone | Epoch | Original GMR mAP | Joint native mAP | Joint GEQ mAP | D-GMR mAP | Δ vs GMR | mR@1 | mR@5 | mR+@5 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for backbone in BACKBONES:
        run = RUNS/backbone
        for epoch in range(1, 5):
            directory = run/'val_results'/f'epoch_{epoch:03d}'
            reports = [read(directory/name) for name in
                       ('original_gmr_metrics.json', 'joint_native_metrics.json',
                        'joint_geq_metrics.json', 'metrics.json')]
            if all(reports):
                original, native, geq, dgmr = [value['brief'] for value in reports]
                lines.append(f"| {backbone} | {epoch} | {original['mAP']:.4f} | "
                             f"{native['mAP']:.4f} | {geq['mAP']:.4f} | {dgmr['mAP']:.4f} | "
                             f"{dgmr['mAP']-original['mAP']:+.4f} | {dgmr['mR@1']:.4f} | "
                             f"{dgmr['mR@5']:.4f} | {dgmr['mR+@5']:.4f} |")
                metric_rows.append(dict(backbone=backbone, split='val', epoch=epoch,
                                        original_gmr_mAP=original['mAP'],
                                        joint_native_mAP=native['mAP'], joint_geq_mAP=geq['mAP'],
                                        delta_mAP=dgmr['mAP']-original['mAP'],
                                        **{key: dgmr[key] for key in KEYS}))
            else:
                lines.append(f'| {backbone} | {epoch} | pending | pending | pending | pending | pending | pending | pending | pending |')
    lines += ['', '## Val-selected checkpoint and formal Test', '',
              '| Backbone | Epoch | Val mAP | Original GMR Test mAP | Joint native | Joint GEQ | D-GMR | Δ vs GMR | Checkpoint |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    analyses = []
    for backbone in BACKBONES:
        run = RUNS/backbone
        best, test = read(run/'best_epoch.json'), read(run/'TEST_COMPLETE.json')
        if best:
            path = Path(best['checkpoint_path'])
            link = path.relative_to(RUNS) if path.is_relative_to(RUNS) else path
            if test:
                dgmr, original = test['metrics'], test['original_gmr_metrics']
                values = (f"{original['mAP']:.4f} | {test['joint_native_metrics']['mAP']:.4f} | "
                          f"{test['joint_geq_metrics']['mAP']:.4f} | {dgmr['mAP']:.4f} | "
                          f"{dgmr['mAP']-original['mAP']:+.4f}")
            else:
                values = 'pending | pending | pending | pending | pending'
            lines.append(f"| {backbone} | {best['best_epoch']} | {best['val_mAP']:.4f} | "
                         f"{values} | [{path.name}]({link}) |")
        else:
            lines.append(f'| {backbone} | pending | pending | pending | pending | pending | pending | pending | pending |')
        if test:
            dgmr, original = test['metrics'], test['original_gmr_metrics']
            metric_rows.append(dict(backbone=backbone, split='test', epoch=test['epoch'],
                                    original_gmr_mAP=original['mAP'],
                                    joint_native_mAP=test['joint_native_metrics']['mAP'],
                                    joint_geq_mAP=test['joint_geq_metrics']['mAP'],
                                    delta_mAP=dgmr['mAP']-original['mAP'],
                                    **{key: dgmr[key] for key in KEYS}))
            direction = 'improved' if dgmr['mAP'] > original['mAP'] else ('decreased' if dgmr['mAP'] < original['mAP'] else 'unchanged')
            directory = Path(test['output'])
            coverage = read(directory/'coverage.json')
            before, after = coverage['original_gmr']['0.5'], coverage['d_gmr']['0.5']
            analyses.append(
                f"- **{backbone}**: mAP {direction} ({dgmr['mAP']-original['mAP']:+.4f} pp); "
                f"mR@1 change {dgmr['mR@1']-original['mR@1']:+.4f} pp. "
                f"Top-10 all-GT coverage at tIoU≥0.5: {before.get('top10_covers_all_gt', 0)} → "
                f"{after.get('top10_covers_all_gt', 0)} of {before['positive_queries']} positive queries. "
                f"Duplicate best-GT hits: {before.get('duplicate_hits_same_gt', 0)} → "
                f"{after.get('duplicate_hits_same_gt', 0)}. [All metrics]({directory.relative_to(RUNS)}/RESULTS.md).")
        elif (run/'TRAIN_FAILED.json').exists():
            analyses.append(f"- **{backbone}**: failed/interrupted: {read(run/'TRAIN_FAILED.json')['error']}")
        else:
            analyses.append(f'- **{backbone}**: formal Test pending.')
    lines += ['', 'Test is evaluated only after all four Val epochs and never updates best_epoch.json.', '',
              '## Failures and limits', '', *analyses, '',
              'Joint training can change native proposals and boundaries through native losses. '
              'Two-tier ranking loss penalizes intra-GT duplicates and promotes inter-GT plurality.', '',
              '## Checkpoint inventory', '',
              'Every epoch has an independent pre-Val recovery archive and a final archive containing Val metrics. '
              'No archive is overwritten or automatically removed.', '',
              '| Backbone | File | Bytes |', '|---|---|---:|']
    for value in checkpoint_rows:
        path = Path(value['path'])
        link = path.relative_to(RUNS) if path.is_relative_to(RUNS) else path
        lines.append(f"| {value['backbone']} | [{path.name}]({link}) | {value['bytes']} |")
    if not checkpoint_rows:
        lines.append('| pending | pending | pending |')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(lines)+'\n')
    temporary.replace(output)
    dump(RUNS/'CHECKPOINTS.json', dict(checkpoints=checkpoint_rows,
                                       epoch_archives_never_overwritten=True))
    if metric_rows:
        path = RUNS/'all_metrics.csv'
        fields = ['backbone', 'split', 'epoch', 'original_gmr_mAP', 'joint_native_mAP',
                  'joint_geq_mAP', 'delta_mAP', *KEYS]
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(metric_rows)
        temporary.replace(path)
    print(output, flush=True)


if __name__ == '__main__':
    main()
