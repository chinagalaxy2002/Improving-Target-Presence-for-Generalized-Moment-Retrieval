#!/usr/bin/env python3
"""Recompute reference-branch GEQ-only G1 at the 86% Val-retention point.

The nine Val prediction files are archived in this branch. Test predictions
are read from a checkout of ``soccergmr-standard-geq-full`` at commit 682db34.
This is an outcome-level reference: the GEQ-only models do not share the
auxiliary-trained Stage-1 parent used by the current GMR/GEQ+aux pair.
"""
import argparse
import csv
import gzip
import hashlib
import json
import subprocess
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
VAL_ROOT = OUT / 'geq_only_reference/val'
EVIDENCE_ROOT = ROOT / 'results/e3_high_retention'
REFERENCE_COMMIT = '682db34ab0511ce0ebfd69e9b966150cac6473a5'
REFERENCE_BRANCH = 'soccergmr-standard-geq-full'
NAMES = (
    'eatr', 'qd_detr', 'cg_detr', 'sim_detr', 'bm_detr', 'bam_detr',
    'uvcom', 'tr_detr', 'taskweave',
)
HISTORICAL = {'eatr', 'qd_detr', 'cg_detr', 'sim_detr'}
CHECKPOINTS = {
    'eatr': 'b1a6cd36dac8406c670af9ad0ebf3edb8acd0757ef27f678f8ff2893eea97347',
    'qd_detr': '144d36e9c0e84d1afbe45bd55ee17a141189cd3bc255d059a062eb0962ade8fb',
    'cg_detr': 'd06de66b24b5b161e0dfc0910520b772784c44f6a0058deb1155b86be7e6e159',
    'sim_detr': '931bc5d075b0835da324a1559498511ffb87ab083d927f004b9d00f45321f643',
    'bm_detr': '0dac37336426b7b06128db49fe6686628e58572db586e748ce7dd8b8b0d17e4a',
    'bam_detr': '2b90dc19a423ca0e63f5bad564840f8bf0ec73ce5cda0a1e3f258547a3ed55c6',
    'uvcom': 'e474e04f716aa25615421da9714884a82ed276f83b59033f791d80e301af4b4b',
    'tr_detr': '031303aaf806f43e91599688d9fd0ad375b0b7e0386c5bf80129cea0d04a0917',
    'taskweave': 'dab8e5465699508cb361a5e9686e4f3751f32d9a358b1162d718b23242557fa1',
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reference-root', type=Path, required=True,
        help='Checkout root for soccergmr-standard-geq-full at commit 682db34.',
    )
    args = parser.parse_args()
    reference_root = args.reference_root.resolve()
    commit = subprocess.check_output(
        ['git', '-C', str(reference_root), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == REFERENCE_COMMIT, (commit, REFERENCE_COMMIT)

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
        profile = 'historical: patience6 / 75 video tokens' if name in HISTORICAL else 'expansion: patience10 / 128 video tokens'
        for split in ('standard', 'full'):
            relative = Path(result_root) / 'runs' / name / split / 'geq_full'
            prediction_path = reference_root / relative / 'predictions.jsonl.gz'
            marker_path = reference_root / relative / 'complete.json'
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

    metrics_path = OUT / 'GEQ_ONLY_RETENTION86_METRICS.csv'
    write_csv(metrics_path, rows)
    means = {
        split: float(np.mean([row['G1'] for row in rows if row['split'] == split]))
        for split in ('standard', 'full')
    }
    audit = {
        'passed': True,
        'comparison_role': 'external outcome-level reference; not a matched aux ablation',
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
    (OUT / 'GEQ_ONLY_RETENTION86_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
