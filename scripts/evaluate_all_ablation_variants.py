#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab/revisions/rev_0001')
sys.path.insert(0, str(ROOT / 'src'))

from soccergmr.config import load_config
from soccergmr.backbones.registry import build, get_spec
from soccergmr.data import SoccerDataset
from soccergmr.evaluation import evaluate
from soccergmr.predict import predict
from soccergmr import checkpoint as ck
from soccergmr.evaluation.official.metrics import _clean_pred_windows, _compute_set_iou_score, get_existence_score

trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

def compute_orc_gap(predictions, gt):
    by_qid = {p['qid']: p for p in predictions}
    ordered = [by_qid[g['qid']] for g in gt if g['qid'] in by_qid]
    rejected, accepted, scores = [], [], []
    for p, g in zip(ordered, gt):
        s, _ = get_existence_score(p)
        scores.append(float(s))
        rejected.append(_compute_set_iou_score([], g['relevant_windows']))
        clean = _clean_pred_windows(p['pred_relevant_windows'], max_pred_windows=10)
        accepted.append(_compute_set_iou_score([w[:2] for w in clean[:1]], g['relevant_windows']))
    thresholds = [round(t * 0.01, 2) for t in range(101)]
    scores_arr = np.array(scores)[:, None]
    mask = (scores_arr > np.array(thresholds))
    acc_arr = np.array(accepted)[:, None]
    rej_arr = np.array(rejected)[:, None]
    ious = np.where(mask, acc_arr, rej_arr)
    curve = ious.mean(axis=0).tolist()
    auc = float(trapz_fn(curve, thresholds))

    truth = np.array([bool(g['relevant_windows']) for g in gt])
    oracle_val = float(np.where(truth, np.array(accepted), np.array(rejected)).mean())

    auc_pct = round(auc * 100, 2)
    oracle_pct = round(oracle_val * 100, 2)
    gap_pct = round(oracle_pct - auc_pct, 2)
    return auc_pct, oracle_pct, gap_pct

ALL_RUNS = [
    ('moment_detr', 'E1', 'Binding-only (identity across layers)'),
    ('moment_detr', 'E2', 'Lite Binding + residual'),
    ('moment_detr', 'E3', 'E2 + Slot-Private Innovation (B)'),
    ('moment_detr', 'E4', 'E2 + Uniform Refresh (C control)'),
    ('moment_detr', 'E5', 'E2 + Dominance Refresh (C)'),
    ('moment_detr', 'E6', 'E2 + Innovation (B) + Dominance (C)'),
    ('moment_detr', 'E7', 'E6 minus Binding (bind_coef=0)'),
    ('bm_detr', 'E2', 'Lite Binding + residual'),
    ('bm_detr', 'E3', 'E2 + Slot-Private Innovation (B)'),
    ('bm_detr', 'E4', 'E2 + Uniform Refresh (C control)'),
    ('bm_detr', 'E5', 'E2 + Dominance Refresh (C)'),
    ('bm_detr', 'E6', 'E2 + Innovation (B) + Dominance (C)'),
    ('bm_detr', 'E7', 'E6 minus Binding (bind_coef=0)'),
    ('cg_detr', 'E2', 'Lite Binding + residual'),
    ('cg_detr', 'E5', 'E2 + Dominance Refresh (C)'),
    ('uvcom', 'E1', 'Binding-only baseline'),
    ('uvcom', 'E2', 'Lite Binding + residual'),
    ('uvcom', 'E5', 'E2 + Dominance Refresh (C)'),
    ('qd_detr', 'E6', 'E2 + Innovation (B) + Dominance (C)'),
]

results_file = Path('/home/guoxiangyu/SoccerGMR-DQ-CGPv3/docs/tables/ablation_variants_detailed_results.json')
existing_results = json.loads(results_file.read_text()) if results_file.exists() else {}

device = 'cuda:0'
current_backbone = None
model = None

for bb, var, desc in ALL_RUNS:
    key = f"{bb}_{var}"
    if key in existing_results and 'standard' in existing_results[key] and 'full' in existing_results[key]:
        print(f"Skipping cached {key}")
        continue

    run_dir = Path(f'/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab/runs/rev_0001/{bb}/{var}')
    ckpt_path = run_dir / 'gmr' / 'best.pt'
    cfg_path = ROOT / f'configs/experiments/interlayer/{bb}/{var}.json'
    if not ckpt_path.exists():
        print(f"Checkpoint not found for {key}, skipping")
        continue

    print(f"\n--- Evaluating {bb} {var} ({desc}) ---")
    cfg = load_config(cfg_path)
    
    # We must run each backbone in separate subprocess if native packages conflict
    # But for same backbone, we can reuse
    if current_backbone != bb:
        del model
        torch.cuda.empty_cache()
        model, _ = build(cfg, device=device)
        current_backbone = bb
    else:
        # Re-build for same backbone if variant architecture differs
        del model
        torch.cuda.empty_cache()
        model, _ = build(cfg, device=device)

    saved = ck.load(ckpt_path)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()

    spec = get_spec(bb)
    entry = {'backbone': bb, 'variant': var, 'description': desc}
    for split_key in ['standard', 'full']:
        split_name = 'test_standard' if split_key == 'standard' else 'test_full'
        ds = SoccerDataset(cfg, split_name, spec)
        preds = predict(model, ds, cfg, device=device)
        res = evaluate(preds, ds.data)
        auc, orc, gap = compute_orc_gap(preds, ds.data)
        brief = res['brief']
        entry[split_key] = {
            'mAP': brief['mAP'],
            'g1_auc_pct': auc,
            'oracle_g1_pct': orc,
            'oracle_gap_pct': gap,
            'rej_f1_50': brief['Rej-F1@0.5'],
            'auroc': brief['AUROC'],
        }
        print(f"  {split_key}: mAP={brief['mAP']}, Gap={gap:.2f}, Rej-F1={brief['Rej-F1@0.5']}, AUROC={brief['AUROC']}")

    existing_results[key] = entry
    results_file.write_text(json.dumps(existing_results, indent=2))

print("\nAll done evaluating ablation variants!")
