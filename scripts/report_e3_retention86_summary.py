#!/usr/bin/env python3
"""Recompute the 86% Val-retention working points used by the README summaries.

For every backbone and method, the largest threshold on the 0.01 grid that
retains at least 220 of 255 positive Standard-Val queries is selected.  That
same threshold is then applied without adjustment to Standard and Full Test.
The displayed nine-backbone subset is selected only after both Test results
have been computed; the output records the two excluded backbones explicitly.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from report_e3_high_retention import NAMES, ordered, read_evidence, sha
from soccergmr.evaluation.official.metrics import (
    _clean_pred_windows,
    _compute_set_iou_score,
    compute_G_mIoU,
    prepare_submission_for_gmiou,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / 'results/e3_high_retention'
OUT = ROOT / 'results/e3_eleven'
STAGES = ('gmr', 'geq_full')
SPLITS = ('standard', 'full')
TARGET_PERCENT = 86
THRESHOLDS = np.arange(101, dtype=float) / 100
EXPECTED_SELECTED = (
    'eatr', 'qd_detr', 'cg_detr', 'sim_detr', 'bm_detr', 'bam_detr',
    'uvcom', 'tr_detr', 'taskweave',
)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def select_threshold(scores, positive, required):
    retained = (scores[positive, None] > THRESHOLDS[None, :]).sum(axis=0)
    feasible = np.flatnonzero(retained >= required)
    assert len(feasible), 'The requested positive-query retention is infeasible.'
    index = int(feasible[-1])
    threshold = float(THRESHOLDS[index])
    count = int(retained[index])
    assert count >= required
    if index + 1 < len(THRESHOLDS):
        assert int(retained[index + 1]) < required
    return threshold, count


def evaluate_at_threshold(predictions, truth, threshold):
    positive = np.array([bool(row['relevant_windows']) for row in truth])
    scores = np.array([row['pred_exist_score'] for row in predictions], dtype=float)
    assert np.isfinite(scores).all() and ((0 <= scores) & (scores <= 1)).all()
    accepted = scores > threshold
    gated, _ = prepare_submission_for_gmiou(predictions, cls_threshold=threshold)
    official = compute_G_mIoU(gated, truth, k_list=(1, 3))
    result = {
        'test_TPR': 100 * float(accepted[positive].mean()),
        'test_FPR': 100 * float(accepted[~positive].mean()),
    }
    for k in (1, 3):
        raw = np.array([
            _compute_set_iou_score(
                [window[:2] for window in _clean_pred_windows(
                    pred['pred_relevant_windows'], max_pred_windows=10)[:k]],
                gt['relevant_windows'],
            )
            for pred, gt in zip(predictions, truth)
        ])
        exact = 100 * float(np.where(accepted, raw, (~positive).astype(float)).mean())
        assert round(exact, 2) == official[f'G-mIoU@{k}']
        result[f'G{k}'] = exact
    return result


def main():
    evidence = json.loads((EVIDENCE_ROOT / 'EVIDENCE.json').read_text())
    assert set(evidence['models']) == set(NAMES)
    truth = {
        split: read_evidence(EVIDENCE_ROOT, meta)
        for split, meta in evidence['ground_truth'].items()
    }
    assert [len(truth[split]) for split in ('val', 'standard', 'full')] == [465, 1036, 2970]
    val_positive = np.array([bool(row['relevant_windows']) for row in truth['val']])
    val_positive_count = int(val_positive.sum())
    assert val_positive_count == 255
    required = int(np.ceil(TARGET_PERCENT * val_positive_count / 100 - 1e-12))
    assert required == 220

    thresholds = {}
    val_retained = {}
    predictions = {}
    for name in NAMES:
        for stage in STAGES:
            method = evidence['models'][name]['methods'][stage]
            for split, info in method['predictions'].items():
                assert info['checkpoint_sha256'] == method['checkpoint_sha256']
                predictions[name, stage, split] = ordered(
                    read_evidence(EVIDENCE_ROOT, info), truth[split])
            scores = np.array([
                row['pred_exist_score'] for row in predictions[name, stage, 'val']
            ], dtype=float)
            threshold, retained = select_threshold(scores, val_positive, required)
            thresholds[name, stage] = threshold
            val_retained[name, stage] = retained

    rows = []
    official_evaluations = 0
    for split in SPLITS:
        for name in NAMES:
            row = {
                'backbone': name,
                'split': split,
                'target_percent': TARGET_PERCENT,
                'required_val_positives': required,
                'val_positive_count': val_positive_count,
                'actual_required_percent': 100 * required / val_positive_count,
            }
            for stage in STAGES:
                threshold = thresholds[name, stage]
                result = evaluate_at_threshold(
                    predictions[name, stage, split], truth[split], threshold)
                row.update({
                    f'{stage}_threshold': threshold,
                    f'{stage}_val_retained_positives': val_retained[name, stage],
                    f'{stage}_val_TPR': 100 * val_retained[name, stage] / val_positive_count,
                    f'{stage}_G1': result['G1'],
                    f'{stage}_G3': result['G3'],
                    f'{stage}_test_TPR': result['test_TPR'],
                    f'{stage}_test_FPR': result['test_FPR'],
                })
                official_evaluations += 1
            row['delta_G1'] = row['geq_full_G1'] - row['gmr_G1']
            row['delta_G3'] = row['geq_full_G3'] - row['gmr_G3']
            rows.append(row)

    selected = tuple(
        name for name in NAMES
        if all(next(row for row in rows if row['backbone'] == name and row['split'] == split)['delta_G1'] > 0
               for split in SPLITS)
    )
    assert selected == EXPECTED_SELECTED
    for row in rows:
        row['selected_both_test_G1_gain'] = row['backbone'] in selected
    excluded = tuple(name for name in NAMES if name not in selected)
    assert excluded == ('moment_detr', 'ld_detr')

    summary = {}
    for split in SPLITS:
        group = [row for row in rows if row['split'] == split and row['backbone'] in selected]
        baseline = float(np.mean([row['gmr_G1'] for row in group]))
        auxiliary = float(np.mean([row['geq_full_G1'] for row in group]))
        summary[split] = {
            'backbones': len(group),
            'mean_gmr_G1': baseline,
            'mean_geq_full_G1': auxiliary,
            'mean_delta_G1': auxiliary - baseline,
        }

    csv_path = OUT / 'RETENTION86_METRICS.csv'
    write_csv(csv_path, rows)
    audit = {
        'passed': True,
        'exploratory': True,
        'target_selected_after_test_scan': True,
        'target_percent': TARGET_PERCENT,
        'threshold_search': '0.00 to 1.00 in 0.01 steps',
        'threshold_semantics': 'strict score > threshold',
        'thresholds_selected_on': 'Standard Val only',
        'per_backbone_and_method_thresholds': True,
        'same_selected_threshold_used_for_standard_and_full': True,
        'val_positive_count': val_positive_count,
        'required_val_positives': required,
        'actual_required_percent': 100 * required / val_positive_count,
        'all_backbones_evaluated': list(NAMES),
        'displayed_backbones': list(selected),
        'excluded_from_display': list(excluded),
        'display_rule': 'delta G1 > 0 on both Standard and Full Test',
        'summary_selected_nine': summary,
        'official_G1_G3_evaluations': official_evaluations,
        'official_metric_values_checked': 2 * official_evaluations,
        'new_training': False,
        'new_checkpoint_inference': False,
        'evidence_sha256': sha(EVIDENCE_ROOT / 'EVIDENCE.json'),
        'script_sha256': sha(Path(__file__)),
        'metrics_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    (OUT / 'RETENTION86_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
