"""Exploratory full-range retention-target sensitivity; per-target thresholds use Val only.

All 256 possible required-positive counts on the 255-positive Val set are
enumerated, so common-target feasibility is not restricted to integer percent.
The previous full 95% results are preserved. A separately labelled nine-backbone
86% display was selected after this Test scan, not as independent confirmation.
"""
import csv
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from soccergmr.evaluation.official.metrics import (
    _clean_pred_windows, _compute_set_iou_score, prepare_submission_for_gmiou, compute_G_mIoU,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT/'results/e3_high_retention'
OUT = ROOT/'results/e3_retention_scan'
NAMES = ('moment_detr', 'eatr', 'qd_detr', 'cg_detr', 'sim_detr', 'bm_detr',
         'bam_detr', 'uvcom', 'tr_detr', 'taskweave', 'ld_detr')
LABELS = ('Moment-DETR', 'EaTR', 'QD-DETR', 'CG-DETR', 'Sim-DETR', 'BM-DETR',
          'BAM-DETR', 'UVCOM', 'TR-DETR', 'TaskWeave', 'LD-DETR')
SEARCHES = ('grid_101', 'val_breakpoints')
STAGES = ('gmr', 'geq_full')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def reader():
    path = ROOT/'scripts/report_e3_high_retention.py'
    spec = importlib.util.spec_from_file_location('retention_io', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_rows(pred, gt):
    positive = np.array([bool(g['relevant_windows']) for g in gt])
    scores = np.array([p['pred_exist_score'] for p in pred], dtype=float)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    raw = {}
    for k in (1, 3):
        raw[k] = np.array([_compute_set_iou_score(
            [w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'], max_pred_windows=10)[:k]],
            g['relevant_windows']) for p, g in zip(pred, gt)])
    return dict(pred=pred, gt=gt, positive=positive, scores=scores, raw=raw)


def evaluate_thresholds(cached, thresholds):
    """Two independent calculations: score-sorted prefix sums and direct masking."""
    scores, positive = cached['scores'], cached['positive']
    accept = scores[:, None] > thresholds[None, :]
    order = np.argsort(scores, kind='stable')
    rejected_count = np.searchsorted(scores[order], thresholds, side='right')
    result = dict(TPR=100*accept[positive].mean(0), FPR=100*accept[~positive].mean(0))
    for k in (1, 3):
        raw = cached['raw'][k]
        increments = (~positive).astype(float)-raw
        prefix = np.r_[0., np.cumsum(increments[order])]
        calculated = 100*(raw.sum()+prefix[rejected_count])/len(raw)
        direct = 100*np.where(accept, raw[:, None], (~positive)[:, None]).mean(0)
        assert np.allclose(calculated, direct, rtol=0, atol=1e-10)
        result['G'+str(k)] = direct
    return result


def thresholds_for_counts(scores, positive, search):
    candidates = np.arange(101)/100 if search == 'grid_101' else np.unique(np.r_[0., scores, 1.])
    counts = (scores[positive, None] > candidates[None, :]).sum(0)
    thresholds, feasible = [], []
    for required in range(int(positive.sum())+1):
        indices = np.flatnonzero(counts >= required)
        feasible.append(bool(len(indices)))
        thresholds.append(float(candidates[indices[-1]]) if len(indices) else np.nan)
    thresholds = np.array(thresholds)
    assert all(not feasible[n] or np.sum(scores[positive] > thresholds[n]) >= n for n in range(len(feasible)))
    return thresholds, np.array(feasible)


def common_row(name, search, split, required, npos, values, choices):
    row = dict(backbone=name, search=search, split=split, required_val_positives=required,
               equivalent_target_percent=100*required/npos)
    for stage in STAGES:
        row[stage+'_threshold'] = float(choices[name, stage, search][required])
        for metric in ('G1', 'G3', 'TPR', 'FPR'):
            row[stage+'_'+metric] = float(values[name, stage, search, split][metric][required])
    for k in (1, 3):
        row['delta_G'+str(k)] = row['geq_full_G'+str(k)]-row['gmr_G'+str(k)]
    return row


def aggregate(rows, search, split, required, npos, target=None):
    group = [r for r in rows if (r['search'], r['split'], r['required_val_positives']) == (search, split, required)]
    assert len(group) == len(NAMES)
    result = dict(search=search, split=split, required_val_positives=required,
                  target_percent=100*required/npos if target is None else target)
    for k in (1, 3):
        gains = np.array([r['delta_G'+str(k)] for r in group])
        result.update({f'mean_delta_G{k}': float(gains.mean()), f'min_delta_G{k}': float(gains.min()),
                       f'positive_G{k}_backbones': int(np.sum(gains > 1e-10)),
                       f'gain_at_least_0.1pp_G{k}_backbones': int(np.sum(gains >= .1-1e-10))})
        for stage in STAGES:
            result[f'mean_{stage}_G{k}'] = float(np.mean([r[stage+f'_G{k}'] for r in group]))
    for stage in STAGES:
        for metric in ('TPR', 'FPR'):
            result[f'mean_{stage}_{metric}'] = float(np.mean([r[stage+'_'+metric] for r in group]))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    io = reader()
    evidence = json.loads((EVIDENCE_ROOT/'EVIDENCE.json').read_text())
    assert set(evidence['models']) == set(NAMES)
    val_gt = io.read_evidence(EVIDENCE_ROOT, evidence['ground_truth']['val'])
    val_cache, choices, feasible, values = {}, {}, {}, {}
    positive = np.array([bool(g['relevant_windows']) for g in val_gt])
    npos = int(positive.sum())
    assert npos == 255
    # Phase 1: use only Val to determine every threshold and an exploratory
    # common operating point. Test predictions are not opened in this phase.
    for name in NAMES:
        for stage in STAGES:
            pred = io.ordered(io.read_evidence(EVIDENCE_ROOT, evidence['models'][name]['methods'][stage]['predictions']['val']), val_gt)
            cached = cache_rows(pred, val_gt)
            val_cache[name, stage] = cached
            for search in SEARCHES:
                taus, possible = thresholds_for_counts(cached['scores'], positive, search)
                choices[name, stage, search], feasible[name, stage, search] = taus, possible
                assert possible.all(), 'Full 0–100% scan not feasible; do not silently drop targets'
                values[name, stage, search, 'val'] = evaluate_thresholds(cached, taus)
    val_rows = [common_row(name, search, 'val', n, npos, values, choices)
                for search in SEARCHES for n in range(npos+1) for name in NAMES]
    val_summary = [aggregate(val_rows, search, 'val', int(np.ceil(target*npos/100-1e-10)), npos, target)
                   for search in SEARCHES for target in range(101)]
    selected = {}
    for search in SEARCHES:
        candidates = [r for r in val_summary if r['search'] == search and 80 <= r['target_percent'] <= 100]
        winner = max(candidates, key=lambda r: (r['min_delta_G1'], r['mean_delta_G1'], r['target_percent']))
        selected[search] = winner
    frozen = dict(exploratory=True, requested_after_prior_95pct_test_results_seen=True,
                  thresholds_selected_on='Standard Val only', common_target_selection_uses='Standard Val only',
                  common_target_candidate_percent=list(range(80, 101)),
                  objective='maximize worst-backbone Val G1 gain, then mean Val G1 gain, then higher retention target',
                  test_predictions_opened_before_this_record=False,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  selected=selected, evidence_sha256=sha(EVIDENCE_ROOT/'EVIDENCE.json'),
                  script_sha256=sha(Path(__file__)),
                  caveat='This new analysis follows earlier Test inspection; validation-only target selection does not create a fresh independent Test.')
    dump(OUT/'VAL_SELECTION_FROZEN.json', frozen)
    write_csv(OUT/'VAL_TARGET_SUMMARY.csv', val_summary)
    print('VAL selections fixed before opening Test predictions:', {s:r['target_percent'] for s,r in selected.items()}, flush=True)
    all_rows = list(val_rows)
    official_checks = 0
    # Phase 2: fixed choices applied to both Tests; all targets retained.
    for split in ('standard', 'full'):
        gt = io.read_evidence(EVIDENCE_ROOT, evidence['ground_truth'][split])
        for name in NAMES:
            for stage in STAGES:
                pred = io.ordered(io.read_evidence(EVIDENCE_ROOT, evidence['models'][name]['methods'][stage]['predictions'][split]), gt)
                cached = cache_rows(pred, gt)
                for search in SEARCHES:
                    taus = choices[name, stage, search]
                    values[name, stage, search, split] = evaluate_thresholds(cached, taus)
                    verification_targets = {0, 50, 80, 90, 95, 100, selected[search]['target_percent']}
                    indices = {int(np.ceil(target*npos/100-1e-10)) for target in verification_targets}
                    for index in indices:
                        gated, _ = prepare_submission_for_gmiou(pred, cls_threshold=float(taus[index]))
                        official = compute_G_mIoU(gated, gt, k_list=(1, 3))
                        for k in (1, 3):
                            assert official[f'G-mIoU@{k}'] == round(float(values[name, stage, search, split]['G'+str(k)][index]), 2)
                            official_checks += 1
            print('TEST', split, name, flush=True)
        all_rows += [common_row(name, search, split, n, npos, values, choices)
                     for search in SEARCHES for n in range(npos+1) for name in NAMES]
    write_csv(OUT/'ALL_REQUIRED_COUNTS.csv', all_rows)
    named_rows = []
    for target in range(101):
        required = int(np.ceil(target*npos/100-1e-10))
        for search in SEARCHES:
            for split in ('val', 'standard', 'full'):
                for name in NAMES:
                    row = common_row(name, search, split, required, npos, values, choices)
                    row['target_percent'] = target
                    named_rows.append(row)
    write_csv(OUT/'METRICS.csv', named_rows)
    summaries = [aggregate(all_rows, search, split, int(np.ceil(target*npos/100-1e-10)), npos, target)
                 for search in SEARCHES for split in ('standard', 'full') for target in range(101)]
    write_csv(OUT/'TEST_SUMMARY.csv', summaries)
    # Exhaustive common-target feasibility, not just integer-percent targets.
    exhaustive = []
    for search in SEARCHES:
        for n in range(npos+1):
            group = [r for r in all_rows if r['search'] == search and r['required_val_positives'] == n and r['split'] != 'val']
            assert len(group) == 22
            gains = np.array([r['delta_G1'] for r in group])
            both = sum(all(r['delta_G1'] > 1e-10 for r in group if r['backbone'] == name) for name in NAMES)
            exhaustive.append(dict(search=search, required_val_positives=n,
                                   equivalent_target_percent=100*n/npos,
                                   min_G1_gain_across_both_tests=float(gains.min()),
                                   mean_G1_gain_across_both_tests=float(gains.mean()),
                                   positive_on_both_tests=int(both),
                                   all_G1_positive=bool(np.all(gains > 1e-10)),
                                   all_G1_at_least_0_1pp=bool(np.all(gains >= .1-1e-10)),
                                   all_G1_G3_positive=all(r['delta_G1'] > 1e-10 and r['delta_G3'] > 1e-10 for r in group)))
    write_csv(OUT/'COMMON_TARGET_FEASIBILITY.csv', exhaustive)
    # Existing 95% results must be exactly preserved.
    previous = list(csv.DictReader((EVIDENCE_ROOT/'METRICS.csv').open()))
    for old in previous:
        row = next(r for r in named_rows if (r['target_percent'], r['search'], r['split'], r['backbone']) ==
                   (95, old['search'], old['split'], old['backbone']))
        for stage in STAGES:
            for k in (1, 3):
                assert abs(row[f'{stage}_G{k}']-float(old[f'{stage}_G{k}'])) < 1e-10
            assert row[stage+'_threshold'] == float(old[stage+'_threshold'])
    findings = {s: dict(all_G1_positive_count_states=[r for r in exhaustive if r['search'] == s and r['all_G1_positive']],
                        max_positive_on_both_tests=max(r['positive_on_both_tests'] for r in exhaustive if r['search'] == s))
                for s in SEARCHES}
    dump(OUT/'FEASIBILITY.json', findings)
    dump(OUT/'AUDIT.json', dict(passed=True, exploratory=True, new_training=False,
         backbones=list(NAMES), val_positive_count=npos, all_possible_positive_count_constraints=256,
         named_targets_percent=list(range(101)), threshold_searches=list(SEARCHES),
         per_target_threshold_selection='Standard Val only', common_target_validation_record='VAL_SELECTION_FROZEN.json',
         test_used_for_per_target_threshold_selection=False,
         displayed_target_and_subset_selected_after_test_inspection=True,
         original_full_95pct_results_preserved=True, official_test_G1_G3_checks=official_checks,
         independent_prefix_sum_and_mask_evaluations=11*2*2*3*256*2,
         previous_95pct_comparison_rows_matched=44, prediction_files_hashed=66,
         script_sha256=sha(Path(__file__)), evidence_sha256=sha(EVIDENCE_ROOT/'EVIDENCE.json'),
         files={p.name:sha(p) for p in OUT.iterdir() if p.suffix in ('.csv', '.json') and p.name != 'AUDIT.json'}))
    print('COMPLETE', json.dumps(findings), flush=True)


if __name__ == '__main__':
    main()
