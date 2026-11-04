#!/usr/bin/env python3
"""Presence Salvage Study: Comprehensive Benchmark and Calibration Analysis.

Evaluates 11 backbones across 4 methods on Standard and Full Test splits.
Demonstrates how GEQ only and Aux salvage the existence classification collapse.
"""
import os, gzip, json, csv
from pathlib import Path
import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from soccergmr.evaluation.official.metrics import (
    _clean_pred_windows, _compute_set_iou_score, compute_G_mIoU, prepare_submission_for_gmiou
)

BACKBONES = [
    'moment_detr', 'eatr', 'qd_detr', 'cg_detr', 'sim_detr',
    'bm_detr', 'bam_detr', 'uvcom', 'tr_detr', 'taskweave', 'ld_detr'
]

DISPLAY_NAMES = {
    'moment_detr': 'Moment-DETR', 'eatr': 'EaTR', 'qd_detr': 'QD-DETR',
    'cg_detr': 'CG-DETR', 'sim_detr': 'Sim-DETR', 'bm_detr': 'BM-DETR',
    'bam_detr': 'BAM-DETR', 'uvcom': 'UVCOM', 'tr_detr': 'TR-DETR',
    'taskweave': 'TaskWeave', 'ld_detr': 'LD-DETR'
}

METHODS = ['GMR baseline', 'GMR(aux)', 'GEQ only', 'GEQ+aux']
GRID = np.arange(101, dtype=float) / 100.0

def load_gz(p):
    return [json.loads(line) for line in gzip.decompress(open(p, 'rb').read()).splitlines() if line.strip()]

def order_predictions(preds, truth):
    idx = {p['qid']: p for p in preds}
    return [idx[t['qid']] for t in truth]

def get_paths(b):
    p1_val = f'{ROOT}/results/e3_eleven/gmr_baseline_reference/val/{b}/predictions.jsonl.gz'
    if not os.path.exists(p1_val):
        p1_val = f'/home/guoxiangyu/SoccerGMR-GEQ/results/presence_thresholds/val_predictions_11/{b}/gmr/predictions.jsonl.gz'
    p2_val = f'{ROOT}/results/e3_high_retention/evidence/{b}/gmr/val/predictions.jsonl.gz'
    p3_val = f'/home/guoxiangyu/SoccerGMR-GEQ/results/presence_thresholds/val_predictions_11/{b}/geq_full/predictions.jsonl.gz'
    p4_val = f'{ROOT}/results/e3_high_retention/evidence/{b}/geq_full/val/predictions.jsonl.gz'

    def get_test_path(m, split):
        if m == 1:
            d = f'{ROOT}/results/test_thresholds/runs' if b in ['moment_detr', 'eatr', 'qd_detr', 'cg_detr', 'sim_detr'] else f'{ROOT}/results/test_thresholds_11_backbones/runs'
            return f'{d}/{b}/{split}/gmr/predictions.jsonl.gz'
        elif m == 2:
            return f'{ROOT}/results/e3_eleven/predictions/{b}/gmr/test_{split}/predictions.jsonl.gz'
        elif m == 3:
            d = f'{ROOT}/results/test_thresholds/runs' if b in ['moment_detr', 'eatr', 'qd_detr', 'cg_detr', 'sim_detr'] else f'{ROOT}/results/test_thresholds_11_backbones/runs'
            return f'{d}/{b}/{split}/geq_full/predictions.jsonl.gz'
        elif m == 4:
            return f'{ROOT}/results/e3_eleven/predictions/{b}/geq_full/test_{split}/predictions.jsonl.gz'

    return {
        'val': [p1_val, p2_val, p3_val, p4_val],
        'standard': [get_test_path(m, 'standard') for m in (1, 2, 3, 4)],
        'full': [get_test_path(m, 'full') for m in (1, 2, 3, 4)]
    }

def eval_curves(preds, truth):
    pos = np.array([bool(t['relevant_windows']) for t in truth])
    scores = np.array([p['pred_exist_score'] for p in preds], dtype=float)
    
    raw_iou1 = np.array([
        _compute_set_iou_score(
            [w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'], max_pred_windows=10)[:1]],
            t['relevant_windows']
        ) if bool(t['relevant_windows']) else 0.0
        for p, t in zip(preds, truth)
    ])
    
    raw_iou3 = np.array([
        _compute_set_iou_score(
            [w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'], max_pred_windows=10)[:3]],
            t['relevant_windows']
        ) if bool(t['relevant_windows']) else 0.0
        for p, t in zip(preds, truth)
    ])
    
    accepted = scores[:, None] > GRID[None, :]
    g1 = 100.0 * np.where(pos[:, None], np.where(accepted, raw_iou1[:, None], 0.0), np.where(accepted, 0.0, 1.0)).mean(axis=0)
    g3 = 100.0 * np.where(pos[:, None], np.where(accepted, raw_iou3[:, None], 0.0), np.where(accepted, 0.0, 1.0)).mean(axis=0)
    tpr = 100.0 * accepted[pos].mean(axis=0)
    fpr = 100.0 * accepted[~pos].mean(axis=0)
    
    return g1, g3, tpr, fpr, scores, pos

def run_study():
    ev_root = ROOT / 'results/e3_high_retention'
    ev = json.load(open(ev_root / 'EVIDENCE.json'))
    gt = {s: load_gz(ev_root / ev['ground_truth'][s]['path']) for s in ('val', 'standard', 'full')}
    
    data = {}
    for b in BACKBONES:
        data[b] = {}
        paths = get_paths(b)
        for m_idx, m_name in enumerate(METHODS):
            data[b][m_name] = {}
            for s in ('val', 'standard', 'full'):
                preds = load_gz(paths[s][m_idx])
                preds = order_predictions(preds, gt[s])
                g1, g3, tpr, fpr, scores, pos = eval_curves(preds, gt[s])
                data[b][m_name][s] = {
                    'g1': g1, 'g3': g3, 'tpr': tpr, 'fpr': fpr,
                    'scores': scores, 'pos': pos
                }

    out_dir = Path(__file__).parent
    
    # Define the 4 Core Practical Calibration Strategies ("说法")
    # 1. Paradigm A: High-Recall Safety Constraint (Val Retention >= 86% / >=220 positives)
    # 2. Paradigm B: Balanced Operational Boundary (Fixed Tau = 0.50)
    # 3. Paradigm C: High-Fidelity Presence Gating (Fixed Tau = 0.40)
    # 4. Paradigm D: Empirical Val-Optimal Tuning (Argmax Val G1)
    
    strategies = [
        ('Recall_86', 'High-Recall Safety Constraint (Val Retention >= 86%)',
         lambda b, m: int(np.flatnonzero(data[b][m]['val']['tpr'] >= 86.274)[-1])),
        ('Fixed_0.50', 'Zero-Prior Natural Decision Boundary (Tau = 0.50)',
         lambda b, m: 50),
        ('Fixed_0.40', 'Standard GMR Gating Boundary (Tau = 0.40)',
         lambda b, m: 40),
        ('Val_Optimal', 'Empirical Risk Minimization (Argmax Val G-mIoU@1)',
         lambda b, m: int(np.argmax(data[b][m]['val']['g1']))),
    ]

    all_results = []
    
    for strat_key, strat_desc, select_th in strategies:
        for split in ('standard', 'full'):
            for b in BACKBONES:
                row = {
                    'strategy_key': strat_key,
                    'strategy_desc': strat_desc,
                    'split': split,
                    'backbone': b,
                    'display_name': DISPLAY_NAMES[b]
                }
                for m in METHODS:
                    th_idx = select_th(b, m)
                    th_val = GRID[th_idx]
                    g1 = data[b][m][split]['g1'][th_idx]
                    g3 = data[b][m][split]['g3'][th_idx]
                    tpr = data[b][m][split]['tpr'][th_idx]
                    fpr = data[b][m][split]['fpr'][th_idx]
                    row[f'{m}_th'] = th_val
                    row[f'{m}_G1'] = g1
                    row[f'{m}_G3'] = g3
                    row[f'{m}_TPR'] = tpr
                    row[f'{m}_FPR'] = fpr
                
                # Gains over baseline
                row['Delta_Aux_G1'] = row['GMR(aux)_G1'] - row['GMR baseline_G1']
                row['Delta_GEQ_G1'] = row['GEQ only_G1'] - row['GMR baseline_G1']
                row['Delta_Both_G1'] = row['GEQ+aux_G1'] - row['GMR baseline_G1']
                
                all_results.append(row)

    # Save to CSV
    csv_path = out_dir / 'STUDY_RESULTS.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f'Saved results to {csv_path}')
    
    # Compute and save summary JSON
    summary = {}
    for strat_key, strat_desc, _ in strategies:
        summary[strat_key] = {'description': strat_desc}
        for split in ('standard', 'full'):
            sub = [r for r in all_results if r['strategy_key'] == strat_key and r['split'] == split]
            summary[strat_key][split] = {
                'GMR baseline_G1_mean': float(np.mean([r['GMR baseline_G1'] for r in sub])),
                'GMR(aux)_G1_mean': float(np.mean([r['GMR(aux)_G1'] for r in sub])),
                'GEQ only_G1_mean': float(np.mean([r['GEQ only_G1'] for r in sub])),
                'GEQ+aux_G1_mean': float(np.mean([r['GEQ+aux_G1'] for r in sub])),
                'Delta_Aux_G1_mean': float(np.mean([r['Delta_Aux_G1'] for r in sub])),
                'Delta_GEQ_G1_mean': float(np.mean([r['Delta_GEQ_G1'] for r in sub])),
                'Delta_Both_G1_mean': float(np.mean([r['Delta_Both_G1'] for r in sub])),
            }
            # Severe collapse subset (where baseline G1 < 30 on tau=0.40 or taskweave/tr_detr/moment_detr/sim_detr/cg_detr)
            collapse_sub = [r for r in sub if r['backbone'] in ['taskweave', 'tr_detr', 'moment_detr', 'sim_detr', 'cg_detr']]
            summary[strat_key][f'{split}_severe_collapse_5_subset'] = {
                'GMR baseline_G1_mean': float(np.mean([r['GMR baseline_G1'] for r in collapse_sub])),
                'GMR(aux)_G1_mean': float(np.mean([r['GMR(aux)_G1'] for r in collapse_sub])),
                'GEQ only_G1_mean': float(np.mean([r['GEQ only_G1'] for r in collapse_sub])),
                'GEQ+aux_G1_mean': float(np.mean([r['GEQ+aux_G1'] for r in collapse_sub])),
                'Delta_Aux_G1_mean': float(np.mean([r['Delta_Aux_G1'] for r in collapse_sub])),
                'Delta_GEQ_G1_mean': float(np.mean([r['Delta_GEQ_G1'] for r in collapse_sub])),
                'Delta_Both_G1_mean': float(np.mean([r['Delta_Both_G1'] for r in collapse_sub])),
            }
    
    json_path = out_dir / 'STUDY_SUMMARY.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved summary to {json_path}')

if __name__ == '__main__':
    run_study()
