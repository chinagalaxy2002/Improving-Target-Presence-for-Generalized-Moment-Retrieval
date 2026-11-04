"""Verify all 44 fixed Tests and 101-point curves without weights or features."""
import csv
import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from soccergmr.checkpoint import source_identity
from soccergmr.evaluation import evaluate
from soccergmr.evaluation.official.metrics import _clean_pred_windows, _compute_set_iou_score

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gzip(path):
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines() if line.strip()]


def main():
    manifest = json.loads((ROOT/'PUBLICATION_MANIFEST.json').read_text())
    for name, value in manifest['files'].items():
        assert sha(ROOT/name) == value, name
    source = source_identity()
    assert source == manifest['training_source']
    out = ROOT/'results/e3_eleven'
    assert '24 passed' in (out/'provenance/unit_tests.log').read_text()
    acceptance = list((out/'provenance/acceptance').glob('*/complete.json'))
    assert len(acceptance) == 11
    for path in acceptance:
        record = json.loads(path.read_text())
        assert record['passed'] and record['source'] == source
    with (out/'METRICS.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with (out/'CURVES.csv').open() as stream:
        curves = list(csv.DictReader(stream))
    assert len(rows) == 44 and len(curves) == 4444
    evidence = json.loads((ROOT/'results/e3_high_retention/EVIDENCE.json').read_text())
    truth = {split: read_gzip(ROOT/'results/e3_high_retention'/evidence['ground_truth'][split]['path'])
             for split in ('standard', 'full')}
    official_checks = curve_points = 0
    for row in rows:
        name, stage, split = [row[k] for k in ('backbone', 'stage', 'split')]
        directory = out/'predictions'/name/stage/split
        raw = gzip.decompress((directory/'predictions.jsonl.gz').read_bytes())
        marker = json.loads((directory/'complete.json').read_text())
        assert hashlib.sha256(raw).hexdigest() == marker['predictions_sha256']
        assert sha(directory/'metrics.json') == marker['metrics_sha256']
        pred = [json.loads(line) for line in raw.splitlines() if line.strip()]
        gt = truth[split.removeprefix('test_')]
        assert len(pred) == marker['rows'] == len(gt)
        byid = {p['qid']: p for p in pred}
        assert len(byid) == len(pred) and set(byid) == {g['qid'] for g in gt}
        pred = [byid[g['qid']] for g in gt]
        assert all(p['vid'] == g['vid'] for p, g in zip(pred, gt))
        saved = json.loads((directory/'metrics.json').read_text())
        recomputed = evaluate(pred, gt)
        for key in ('brief', 'gmiou', 'mAP_detail'):
            assert recomputed[key] == saved[key], (name, stage, split, key)
        assert recomputed['brief']['mAP'] == float(row['mAP'])
        official_checks += 1
        group = [r for r in curves if (r['backbone'], r['stage'], r['split']) == (name, stage, split)]
        thresholds = np.array([i/100 for i in range(101)])
        assert [float(r['threshold']) for r in group] == list(thresholds)
        scores = np.array([p['pred_exist_score'] for p in pred])
        null = np.array([float(not g['relevant_windows']) for g in gt])
        for k in (1, 3):
            raw_iou = np.array([_compute_set_iou_score(
                [w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'], max_pred_windows=10)[:k]],
                g['relevant_windows']) for p, g in zip(pred, gt)])
            computed = 100*np.where(scores[:, None] > thresholds[None, :], raw_iou[:, None], null[:, None]).mean(0)
            assert all(round(float(a), 2) == float(b[f'G{k}']) for a, b in zip(computed, group))
            assert abs(sum(float(group[i][f'G{k}']) for i in (40, 50, 60))/3-float(row[f'G{k}_mean'])) < 1e-10
            curve_points += 101
        print('VERIFIED', name, stage, split, flush=True)
    links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', (ROOT/'README.md').read_text())
    local = [p.split('#')[0] for p in links if '://' not in p and not p.startswith('#')]
    assert all((ROOT/p).exists() for p in local), [p for p in local if not (ROOT/p).exists()]
    retention = json.loads((ROOT/'results/e3_high_retention/AUDIT.json').read_text())
    assert retention['passed'] and retention['all_published_backbones_included']
    assert len(retention['backbones']) == 11 and retention['prediction_files'] == 66
    checks = dict(passed=True, publication_files_checked=len(manifest['files']),
                  published_source_matches_training=True, source=source, full_test_metric_recomputations=official_checks,
                  G1_G3_curve_points_recomputed=curve_points, README_local_links_checked=len(local),
                  unit_tests_passed=24, backbone_acceptance_records_checked=11,
                  high_retention_official_checks=retention['official_G1_G3_checks'],
                  high_retention_mAP_checks=retention['mAP_checks'],
                  method='CPU recomputation from hashed predictions and GT; no new training or model inference',
                  manifest_sha256=sha(ROOT/'PUBLICATION_MANIFEST.json'))
    (ROOT/'PUBLICATION_CHECKS.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
