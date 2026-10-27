#!/usr/bin/env python3
"""Recompute saved Val/Test metrics, result tables and five threshold figures."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'evaluation'))
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
from official_eval.eval_main import evaluate_gmr

BACKBONES = ('moment_detr', 'qd_detr', 'cg_detr', 'eatr', 'flashvtg')
NAMES = ('Moment-DETR', 'QD-DETR', 'CG-DETR', 'EaTR', 'FlashVTG')
VIEWS = {
    'original_gmr': ('original_gmr_predictions.jsonl', 'original_gmr_metrics.json'),
    'joint_native': ('joint_native_predictions.jsonl', 'joint_native_metrics.json'),
    'joint_geq': ('joint_geq_predictions.jsonl', 'joint_geq_metrics.json'),
    'd_gmr': ('predictions.jsonl', 'metrics.json'),
}
METRICS = ('mAP', 'mR@1', 'mR@5', 'mR+@5', 'AUROC', 'Rej-F1@0.4',
           'G-mIoU@1', 'G-mIoU@3', 'N-acc')


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-figures', action='store_true')
    args = parser.parse_args()
    manifest = read(ROOT/'MANIFEST.json')
    for name, digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    reports = ROOT/'reports'
    reports.mkdir(exist_ok=True)
    metric_rows, table, checked = [], [], 0
    for backbone, label in zip(BACKBONES, NAMES):
        result = ROOT/'results'/backbone
        selected = read(result/'best_epoch.json')
        complete = read(result/'TEST_COMPLETE.json')
        history = []
        for epoch in range(1, 5):
            saved = read(result/'val_history'/f'epoch_{epoch:03d}'/'metrics.json')['brief']
            history.append((saved['mAP'], saved['mR@1'], -epoch))
        assert -max(history)[2] == selected['best_epoch'] == complete['epoch']
        for split in ('val', 'test'):
            ground_truth = rows(ROOT/'data'/f'{split}.jsonl')
            gt_by_id = {r['qid']: r for r in ground_truth}
            negatives = {r['qid'] for r in ground_truth if not r['relevant_windows']}
            for view, (prediction_name, metric_name) in VIEWS.items():
                prediction = rows(result/split/prediction_name)
                assert len({r['qid'] for r in prediction}) == len(prediction) == len(ground_truth)
                assert {r['qid'] for r in prediction} == gt_by_id.keys()
                reproduced = evaluate_gmr(prediction, ground_truth, cls_thresholds=(.4,),
                    gmiou_cls_threshold=.4, map_num_workers=0, verbose=False)['brief']
                reproduced['N-acc'] = 100*sum(r['pred_exist_score'] <= .4 for r in prediction
                    if r['qid'] in negatives)/len(negatives)
                saved = read(result/split/metric_name)['brief']
                for key, value in reproduced.items():
                    assert abs(value-saved[key]) < 1e-8, (backbone, split, view, key, value, saved[key])
                    checked += 1
                metric_rows.append(dict(backbone=backbone, split=split, view=view,
                    epoch=selected['best_epoch'], **{k: reproduced[k] for k in METRICS}))
        original = complete['original_gmr_metrics']
        final = complete['metrics']
        table.append(f"| {label} | {selected['best_epoch']} | {selected['val_mAP']:.2f} | "
                     f"{original['mAP']:.2f} | {final['mAP']:.2f} | "
                     f"{final['mAP']-original['mAP']:+.2f} | {final['mR@1']:.2f} | "
                     f"{final['mR@5']:.2f} | {final['mR+@5']:.2f} | {final['AUROC']:.2f} | "
                     f"{final['Rej-F1@0.4']:.2f} | {final['G-mIoU@1']:.2f} | "
                     f"{final['G-mIoU@3']:.2f} | {final['N-acc']:.2f} |")
        print(f'{label}: selected epoch {selected["best_epoch"]}, Val/Test metrics reproduced', flush=True)
    with (reports/'reproduced_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(metric_rows)
    text = ['# D-GMR results', '',
        'Seed 2026; four training epochs; checkpoint selected by Val mAP, then Val mR@1, then earlier epoch.',
        'Standard Test: 1,036 queries (544 positive / 492 negative); presence threshold 0.4.',
        'Positive-query localization has no presence hard gate. Values are percentages; deltas are percentage points.', '',
        '| Backbone | Epoch | Val mAP | Original mAP | D-GMR mAP | ΔmAP | mR@1 | mR@5 | mR+@5 | AUROC | Rej-F1 | G-mIoU@1 | G-mIoU@3 | N-acc |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|', *table, '',
        '[All epochs: Val/Test CSV](../results/all_metrics.csv) · [Recomputed metrics](reproduced_metrics.csv)', '']
    (reports/'RESULTS.md').write_text('\n'.join(text))
    audit = dict(passed=True, method='D-GMR', seed=2026, backbones=list(BACKBONES),
                 val_selected=True, metric_values_checked=checked,
                 input_files_verified=len(manifest['sha256']), model_inference_performed=False)
    (reports/'REPRODUCTION_AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
    if not args.skip_figures:
        subprocess.run([sys.executable, str(ROOT/'plot_threshold_sweeps.py'), '--recompute'], check=True)
    print(f'Checked {checked} metrics; reports: {reports}', flush=True)


if __name__ == '__main__':
    main()
