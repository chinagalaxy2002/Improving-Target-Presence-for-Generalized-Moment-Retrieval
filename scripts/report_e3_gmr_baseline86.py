#!/usr/bin/env python3
"""Compute the true no-auxiliary GMR baseline G1 and G3 at the 86% Val-retention working point.

The nine Val prediction files of the original GMR parent models are archived in
`results/e3_eleven/gmr_baseline_reference/val/`. Test predictions are read from
`results/test_thresholds` (for the four historical backbones) and
`results/test_thresholds_11_backbones` (for the five expansion backbones).
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from report_e3_high_retention import ordered, read_evidence, sha
from soccergmr.evaluation.official.metrics import (
    _clean_pred_windows,
    _compute_set_iou_score,
    compute_G_mIoU,
    prepare_submission_for_gmiou,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/e3_eleven'
VAL_ROOT = OUT / 'gmr_baseline_reference/val'
EVIDENCE_ROOT = ROOT / 'results/e3_high_retention'
REFERENCE_COMMIT = '682db34ab0511ce0ebfd69e9b966150cac6473a5'
REFERENCE_BRANCH = 'soccergmr-standard-geq-full'
NAMES = (
    'eatr', 'qd_detr', 'cg_detr', 'sim_detr', 'bm_detr', 'bam_detr',
    'uvcom', 'tr_detr', 'taskweave',
)
HISTORICAL = {'eatr', 'qd_detr', 'cg_detr', 'sim_detr'}
CHECKPOINTS = {
    'eatr': '3d7f6f6cb8c62cfee30db2d48fee5d1e62e5d69604cc7ab484075ab4979606c0',
    'qd_detr': '03011c48168532bc8a99cee669d0d0489d0919f8fb127474107ed705e88b43c0',
    'cg_detr': '200f2f200bea257b1704f3bf285aecd50cf1722eb7fa7390e0763042f5a9b1d8',
    'sim_detr': 'c237601599d5e257f02481751956bfeabeb51253e6e3fc26da40277c3a95bd1c',
    'bm_detr': '1382ac4645793db224894a9c91eed754ef305a65d7f376462ca14fc48f283070',
    'bam_detr': 'd2865dd460dafab8cf1a65a08583562d30105a612ebad2ee5027382f5ff11bc5',
    'uvcom': 'a07df6dd16ba90d40f27de87deee86604aab4ec6cbd19f95f4654d6f21943663',
    'tr_detr': 'ed31af7256a00c308fec62aecdcbdfac1866081d9dd1fd014169c9913a389230',
    'taskweave': '9d74a134329e869965515119688dad08285f4f258d71c8aa0056104dbca4ddbc',
}
TARGET_PERCENT = 86
REQUIRED = 220
GRID = np.arange(101, dtype=float) / 100


def read_gzip(path):
    raw = gzip.decompress(path.read_bytes())
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return rows, raw


def select_threshold(predictions, positive):
    scores = np.array([row['pred_exist_score'] for row in predictions], dtype=float)
    counts = (scores[positive, None] > GRID[None, :]).sum(0)
    feasible = np.flatnonzero(counts >= REQUIRED)
    assert len(feasible)
    index = int(feasible[-1])
    if index + 1 < len(GRID):
        assert int(counts[index + 1]) < REQUIRED
    return float(GRID[index]), int(counts[index])


def evaluate(predictions, truth, threshold):
    positive = np.array([bool(row['relevant_windows']) for row in truth])
    scores = np.array([row['pred_exist_score'] for row in predictions], dtype=float)
    accepted = scores > threshold
    gated, _ = prepare_submission_for_gmiou(predictions, cls_threshold=threshold)
    official = compute_G_mIoU(gated, truth, k_list=(1, 3))
    result = {
        'test_TPR': 100 * float(accepted[positive].mean()),
        'test_FPR': 100 * float(accepted[~positive].mean()),
    }
    for k in (1, 3):
        raw_iou = np.array([
            _compute_set_iou_score(
                [window[:2] for window in _clean_pred_windows(
                    pred['pred_relevant_windows'], max_pred_windows=10)[:k]],
                gt['relevant_windows'],
            )
            for pred, gt in zip(predictions, truth)
        ])
        exact = 100 * float(np.where(accepted, raw_iou, (~positive).astype(float)).mean())
        assert round(exact, 2) == official[f'G-mIoU@{k}']
        result[f'G{k}'] = exact
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    evidence = json.loads((EVIDENCE_ROOT / 'EVIDENCE.json').read_text())
    truth = {
        split: read_evidence(EVIDENCE_ROOT, evidence['ground_truth'][split])
        for split in ('val', 'standard', 'full')
    }
    val_positive = np.array([bool(row['relevant_windows']) for row in truth['val']])
    assert int(val_positive.sum()) == 255 and REQUIRED == int(np.ceil(.86 * 255))

    thresholds = {}
    val_counts = {}
    val_hashes = {}
    for name in NAMES:
        path = VAL_ROOT / name / 'predictions.jsonl.gz'
        predictions, _ = read_gzip(path)
        predictions = ordered(predictions, truth['val'])
        threshold, count = select_threshold(predictions, val_positive)
        thresholds[name] = threshold
        val_counts[name] = count
        val_hashes[name] = sha(path)

    rows = []
    source_files = []
    for name in NAMES:
        result_root = ('results/test_thresholds' if name in HISTORICAL
                       else 'results/test_thresholds_11_backbones')
        profile = ('historical: patience6 / 75 video tokens' if name in HISTORICAL
                   else 'expansion: patience10 / 128 video tokens')
        for split in ('standard', 'full'):
            relative = Path(result_root) / 'runs' / name / split / 'gmr'
            prediction_path = ROOT / relative / 'predictions.jsonl.gz'
            marker_path = ROOT / relative / 'complete.json'
            predictions, raw = read_gzip(prediction_path)
            marker = json.loads(marker_path.read_text())
            assert hashlib.sha256(raw).hexdigest() == marker['predictions_sha256']
            if marker.get('checkpoint_sha256'):
                assert marker['checkpoint_sha256'] == CHECKPOINTS[name]
            predictions = ordered(predictions, truth[split])
            metric = evaluate(predictions, truth[split], thresholds[name])
            rows.append({
                'backbone': name,
                'split': split,
                'profile': profile,
                'target_percent': TARGET_PERCENT,
                'required_val_positives': REQUIRED,
                'val_positive_count': 255,
                'threshold': thresholds[name],
                'val_retained_positives': val_counts[name],
                'val_TPR': 100 * val_counts[name] / 255,
                'G1': metric['G1'],
                'G3': metric['G3'],
                'test_TPR': metric['test_TPR'],
                'test_FPR': metric['test_FPR'],
                'checkpoint_sha256': CHECKPOINTS[name],
                'val_prediction_sha256': val_hashes[name],
                'test_prediction_sha256': sha(prediction_path),
                'reference_path': str(relative / 'predictions.jsonl.gz'),
            })
            source_files.append({
                'backbone': name,
                'split': split,
                'path': str(relative / 'predictions.jsonl.gz'),
                'sha256': sha(prediction_path),
            })

    metrics_path = OUT / 'GMR_BASELINE_RETENTION86_METRICS.csv'
    write_csv(metrics_path, rows)
    means = {
        split: float(np.mean([row['G1'] for row in rows if row['split'] == split]))
        for split in ('standard', 'full')
    }
    audit = {
        'passed': True,
        'method': 'native GMR baseline without aux (no GEQ, no aux)',
        'comparison_role': 'true no-auxiliary GMR baseline parent from soccergmr-standard-geq-full',
        'reference_repository': 'chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval',
        'reference_branch': REFERENCE_BRANCH,
        'reference_commit': REFERENCE_COMMIT,
        'target_percent': TARGET_PERCENT,
        'required_val_positives': REQUIRED,
        'threshold_rule': 'largest 0.01-grid Standard-Val threshold retaining >=220/255 positives; strict score > threshold',
        'same_threshold_used_for_standard_and_full': True,
        'backbones': list(NAMES),
        'historical_protocol_backbones': sorted(HISTORICAL),
        'expansion_protocol_backbones': [name for name in NAMES if name not in HISTORICAL],
        'unmatched_parent_and_protocol': True,
        'mean_G1': means,
        'official_G1_G3_evaluations': len(rows),
        'official_metric_values_checked': 2 * len(rows),
        'archived_val_prediction_files': len(NAMES),
        'test_source_files': source_files,
        'metrics_sha256': sha(metrics_path),
        'script_sha256': sha(Path(__file__)),
        'new_training': False,
        'new_checkpoint_inference': False,
    }
    (OUT / 'GMR_BASELINE_RETENTION86_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Completed GMR baseline 86% audit and metrics recomputation.')
    print(f"Standard mean G1: {means['standard']:.2f}")
    print(f"Full mean G1: {means['full']:.2f}")


if __name__ == '__main__':
    main()
