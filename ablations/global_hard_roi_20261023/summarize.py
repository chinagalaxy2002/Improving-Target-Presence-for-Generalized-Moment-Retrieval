"""CPU-only result tables and official metric recomputation for the 2x2 study."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
from decimal import Decimal

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / ('runs' if (ROOT / 'runs').exists() else 'results')
BACKBONES = {'moment_detr': 'Moment-DETR', 'qd_detr': 'QD-DETR', 'cg_detr': 'CG-DETR', 'eatr': 'EaTR', 'flashvtg': 'FlashVTG'}
SETTINGS = {'gmr': 'GMR', 'global_only': '+ Global Event Query', 'hard_only': '+ Hard-RoI', 'global_hard': '+ Global Event Query + Hard-RoI'}
METRICS = ['N-acc', 'AUROC', 'Rej-F1', 'mAP', 'mR@1', 'mR@5', 'mR+@5', 'G-mIoU@1', 'G-mIoU@3']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def values(report, tau):
    return {k: report['brief']['Rej-F1@' + tau if k == 'Rej-F1' else k] for k in METRICS}


def write_csv(path, records):
    if not records:
        return
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def delta(a, b):
    # Match the visible table: subtract values after displaying two decimals.
    return {k: float(Decimal(f'{a[k]:.2f}') - Decimal(f'{b[k]:.2f}')) for k in METRICS}


def verify(directory, split):
    from official_eval.eval_main import evaluate_gmr
    from official_eval.metrics import prepare_submission_for_gmiou, compute_G_mIoU
    gt = read_rows(ROOT / 'data' / f'{split}.jsonl')
    pred = read_rows(directory / f'{split}_predictions.jsonl')
    assert len(pred) == len(gt) == (465 if split == 'val' else 1036)
    assert len({p['qid'] for p in pred}) == len(gt) and {p['qid'] for p in pred} == {g['qid'] for g in gt}
    actual = evaluate_gmr(pred, gt, cls_thresholds=(.4, .5, .6), gmiou_cls_threshold=.4, map_num_workers=0, verbose=False)
    saved = json.loads((directory / f'{split}_metrics.json').read_text())
    negatives = {g['qid'] for g in gt if not g['relevant_windows']}
    for tau in ['0.4', '0.5', '0.6']:
        gated, _ = prepare_submission_for_gmiou(pred, cls_threshold=float(tau), max_pred_windows=10)
        current = {k: actual['brief'][k] for k in ['AUROC', 'mAP', 'mR@1', 'mR@5', 'mR+@5']}
        current.update(compute_G_mIoU(gated, gt, k_list=(1, 3)))
        current['Rej-F1'] = actual['GMR-CLS']['per_threshold'][tau]['Rej-F1']
        current['N-acc'] = 100 * sum(p['pred_exist_score'] <= float(tau) for p in pred if p['qid'] in negatives) / len(negatives)
        expected = values(saved[tau], tau)
        assert all(abs(current[k] - expected[k]) < 1e-6 for k in METRICS), (directory, split, tau, current, expected)
    return dict(backbone=directory.parent.name, setting=directory.name, split=split, queries=len(gt), prediction_sha256=sha(directory / f'{split}_predictions.jsonl'))


def table(records):
    lines = ['| Backbone | Configuration | ' + ' | '.join(METRICS) + ' |',
             '|---|---|' + '---:|' * len(METRICS)]
    for row in records:
        lines.append('| ' + ' | '.join([BACKBONES[row['backbone']], SETTINGS[row['setting']]] + [f'{row[k]:.2f}' for k in METRICS]) + ' |')
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'reports'
    out.mkdir(exist_ok=True)
    records, checks = [], []
    for backbone in BACKBONES:
        for setting in SETTINGS:
            directory = RESULTS / backbone / setting
            if not (directory / 'COMPLETE.json').exists():
                continue
            for split in ['val', 'test']:
                if args.verify:
                    checks.append(verify(directory, split))
                for tau, report in json.loads((directory / f'{split}_metrics.json').read_text()).items():
                    records.append(dict(backbone=backbone, setting=setting, split=split, tau=tau, **values(report, tau)))
    count = len(records) // 6
    lookup = {(r['backbone'], r['setting'], r['split'], r['tau']): r for r in records}
    gains, effects = [], []
    for row in records:
        if row['setting'] != 'gmr':
            baseline = lookup[(row['backbone'], 'gmr', row['split'], row['tau'])]
            gains.append(dict(backbone=row['backbone'], setting=row['setting'], split=row['split'], tau=row['tau'], **delta(row, baseline)))
    for backbone in BACKBONES:
        for split in ['val', 'test']:
            for tau in ['0.4', '0.5', '0.6']:
                if (backbone, 'global_hard', split, tau) not in lookup:
                    continue
                get = lambda s: lookup[(backbone, s, split, tau)]
                for a, b, label in [('global_only', 'gmr', 'Global effect without Hard-RoI'),
                                     ('hard_only', 'gmr', 'Hard-RoI effect without Global'),
                                     ('global_hard', 'global_only', 'Hard-RoI effect with Global'),
                                     ('global_hard', 'hard_only', 'Global effect with Hard-RoI')]:
                    effects.append(dict(backbone=backbone, split=split, tau=tau, effect=label, **delta(get(a), get(b))))
                interaction = {k: round((get('global_hard')[k] - get('global_only')[k]) - (get('hard_only')[k] - get('gmr')[k]), 6) for k in METRICS}
                effects.append(dict(backbone=backbone, split=split, tau=tau, effect='Interaction (raw difference-in-differences)', **interaction))
    write_csv(out / 'all_metrics.csv', records)
    write_csv(out / 'gains_vs_gmr_pp.csv', gains)
    write_csv(out / 'factorial_effects_pp.csv', effects)
    lines = ['# Global Event Query + Hard-RoI: fixed epoch 4', '',
             f'{count}/20 cells available (5 backbones × 4 configurations). Frozen GMR uses 0 new epochs; all adapted cells use exactly 4 epochs, seed 2026.', '',
             'LoRA, Gaussian pooling, 6D saliency, candidate presence fusion and synthetic samples are disabled. Metrics are percentages. Primary tau=0.4; tau=0.5/0.6 are fixed sensitivity checks.', '']
    for split in ['test', 'val']:
        for tau in ['0.4', '0.5', '0.6']:
            selected = [r for r in records if r['split'] == split and r['tau'] == tau]
            body = table(selected)
            (out / f'{split}_tau_{tau}.md').write_text('\n'.join(body) + '\n')
            if tau == '0.4':
                lines.extend([f'## {split.title()}, tau={tau}', '', *body, ''])
    lines.extend(['The baseline and Global-only cells reuse verified matching historical runs; the 10 Hard-only / Global+Hard cells are newly trained. All cells start independently from the same pretrained GMR checkpoint within each backbone.', '',
                  'Hard-only retains native GMR presence. Original GMR tensors stay frozen. Hard-only all-negative batches have no boundary target and perform no optimizer update. No test-based checkpoint or threshold selection; these are exploratory single-seed results on previously observed Standard labels.', '',
                  'FlashVTG retains its original soft presence confidence calibration, rounding and NMS. Its final candidate confidence ordering can depend on presence even with identical raw localizer proposals.', '',
                  '[All metrics](all_metrics.csv) · [GMR-relative gains, pp](gains_vs_gmr_pp.csv) · [Conditional effects and interaction](factorial_effects_pp.csv)'])
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    if args.verify:
        assert count == 20 and len(checks) == 40
        dump(out / 'METRIC_VERIFICATION.json', dict(passed=True, cells=20, newly_trained=10, reused=10,
             split_prediction_sets=40, threshold_evaluations=120, metrics=METRICS, checks=checks))
    print(f'{count}/20 cells summarized', flush=True)


if __name__ == '__main__':
    main()
