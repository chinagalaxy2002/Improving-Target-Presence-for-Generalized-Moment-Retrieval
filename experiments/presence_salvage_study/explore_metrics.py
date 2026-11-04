import os, gzip, json, argparse
import numpy as np
from pathlib import Path

# Add repo to pythonpath
import sys
ROOT = Path('/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from soccergmr.evaluation.official.metrics import (
    _clean_pred_windows, _compute_set_iou_score, compute_G_mIoU, prepare_submission_for_gmiou
)

BACKBONES = [
    'moment_detr', 'eatr', 'qd_detr', 'cg_detr', 'sim_detr',
    'bm_detr', 'bam_detr', 'uvcom', 'tr_detr', 'taskweave', 'ld_detr'
]

SPLITS = ['standard', 'full']
GRID = np.arange(101, dtype=float) / 100.0

def load_gz(p):
    return [json.loads(line) for line in gzip.decompress(open(p, 'rb').read()).splitlines() if line.strip()]

def order_predictions(preds, truth):
    idx = {p['qid']: p for p in preds}
    return [idx[t['qid']] for t in truth]

def get_paths(b):
    # Val paths
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

def fast_eval_curves(preds, truth):
    # compute G-mIoU@1 for all 101 thresholds efficiently
    pos = np.array([bool(t['relevant_windows']) for t in truth])
    scores = np.array([p['pred_exist_score'] for p in preds], dtype=float)
    
    # Precompute raw IoU for top 1
    raw_iou = np.array([
        _compute_set_iou_score(
            [w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'], max_pred_windows=10)[:1]],
            t['relevant_windows']
        ) if bool(t['relevant_windows']) else 0.0
        for p, t in zip(preds, truth)
    ])
    
    # scores: [N], GRID: [101]
    accepted = scores[:, None] > GRID[None, :] # [N, 101]
    
    # positive sample: if accepted -> raw_iou, else -> 0.0
    # negative sample: if accepted -> 0.0, else -> 1.0 (correct rejection)
    sample_scores = np.where(pos[:, None],
                             np.where(accepted, raw_iou[:, None], 0.0),
                             np.where(accepted, 0.0, 1.0))
    gmiou1 = 100.0 * sample_scores.mean(axis=0)
    tpr = 100.0 * accepted[pos].mean(axis=0)
    fpr = 100.0 * accepted[~pos].mean(axis=0)
    
    return gmiou1, tpr, fpr, scores, pos

def main():
    ev_root = ROOT / 'results/e3_high_retention'
    ev = json.load(open(ev_root / 'EVIDENCE.json'))
    gt = {s: load_gz(ev_root / ev['ground_truth'][s]['path']) for s in ('val', 'standard', 'full')}
    
    method_names = ['GMR baseline', 'GMR(aux)', 'GEQ only', 'GEQ+aux']
    
    # Collect all curves
    data = {}
    for b in BACKBONES:
        data[b] = {}
        paths = get_paths(b)
        for m_idx, m_name in enumerate(method_names):
            data[b][m_name] = {}
            for s in ('val', 'standard', 'full'):
                preds = load_gz(paths[s][m_idx])
                preds = order_predictions(preds, gt[s])
                g1, tpr, fpr, scores, pos = fast_eval_curves(preds, gt[s])
                data[b][m_name][s] = {
                    'g1': g1, 'tpr': tpr, 'fpr': fpr,
                    'scores': scores, 'pos': pos
                }
    
    print('Curve computation completed for all 11 backbones, 4 methods, 3 splits!')
    
    # Let us explore calibration strategies:
    # Strategy 1: Val Max G-mIoU (ERM calibration)
    # Strategy 2: Val Retention >= 86% (User experience preserving)
    # Strategy 3: Val Retention >= 90% (High recall)
    # Strategy 4: Val Retention >= 95% (Extreme recall)
    # Strategy 5: Val Youden Index (max TPR - FPR)
    # Strategy 6: Default fixed threshold 0.5
    
    strategies = {
        'Val_Max_G1': lambda v_g1, v_tpr, v_fpr: int(np.argmax(v_g1)),
        'Val_Ret_86': lambda v_g1, v_tpr, v_fpr: int(np.flatnonzero(v_tpr >= 86.274)[-1]),
        'Val_Ret_90': lambda v_g1, v_tpr, v_fpr: int(np.flatnonzero(v_tpr >= 90.0)[-1]),
        'Val_Ret_95': lambda v_g1, v_tpr, v_fpr: int(np.flatnonzero(v_tpr >= 95.0)[-1]),
        'Val_Youden': lambda v_g1, v_tpr, v_fpr: int(np.argmax(v_tpr - v_fpr)),
        'Fixed_0.5': lambda v_g1, v_tpr, v_fpr: 50,
    }
    
    for strat_name, strat_fn in strategies.items():
        print(f'\n==================== STRATEGY: {strat_name} ====================')
        for split in ('standard', 'full'):
            print(f'--- Split: {split} ---')
            means = {m: [] for m in method_names}
            for b in BACKBONES:
                row_str = f'{b:<12} | '
                for m in method_names:
                    v_g1 = data[b][m]['val']['g1']
                    v_tpr = data[b][m]['val']['tpr']
                    v_fpr = data[b][m]['val']['fpr']
                    th_idx = strat_fn(v_g1, v_tpr, v_fpr)
                    test_g1 = data[b][m][split]['g1'][th_idx]
                    means[m].append(test_g1)
                    row_str += f'{m}: {test_g1:5.2f} (th={GRID[th_idx]:.2f}) | '
                # print(row_str)
            print('  11-BACKBONE MEAN:')
            for m in method_names:
                print(f'    {m:<20}: {np.mean(means[m]):.2f}')
            d_aux = np.mean(means['GMR(aux)']) - np.mean(means['GMR baseline'])
            d_geq = np.mean(means['GEQ only']) - np.mean(means['GMR baseline'])
            d_both = np.mean(means['GEQ+aux']) - np.mean(means['GMR baseline'])
            print(f'  --> Gain over Baseline: Aux = {d_aux:+.2f} pp, GEQ only = {d_geq:+.2f} pp, GEQ+aux = {d_both:+.2f} pp')

if __name__ == '__main__':
    main()
