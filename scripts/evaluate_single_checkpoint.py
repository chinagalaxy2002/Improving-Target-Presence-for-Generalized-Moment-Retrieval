#!/usr/bin/env python3
"""Evaluate a single trained checkpoint on Standard Test and Full Test.

Records the results into ablation_variants_detailed_results.json,
regenerates SI_BIND_ABLATION_TABLE_E1_E7 figures, and pushes to git.
"""

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

LAB_ROOT = Path("/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab").resolve()
REV_DIR = LAB_ROOT / "revisions" / "rev_0001"
DQC_ROOT = Path("/home/guoxiangyu/SoccerGMR-DQ-CGPv3").resolve()
RESULTS_JSON = DQC_ROOT / "docs" / "tables" / "ablation_variants_detailed_results.json"
TOKEN_PATH = Path("/home/guoxiangyu/token2git.txt")
PYTHON_BIN = "/home/guoxiangyu/miniconda3/envs/gmr/bin/python"

sys.path.insert(0, str(REV_DIR / "src"))
from soccergmr.config import load_config
from soccergmr.backbones.registry import build, get_spec
from soccergmr.data import SoccerDataset
from soccergmr.evaluation import evaluate
from soccergmr.predict import predict
from soccergmr import checkpoint as ck
from soccergmr.evaluation.official.metrics import _clean_pred_windows, _compute_set_iou_score, get_existence_score

trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


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


def evaluate_checkpoint(backbone: str, variant: str, gpu_id: int):
    ckpt_path = LAB_ROOT / "runs" / "rev_0001" / backbone / variant / "gmr" / "best.pt"
    cfg_path = REV_DIR / "configs" / "experiments" / "interlayer" / backbone / f"{variant}.json"
    device = f"cuda:{gpu_id}"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log(f"Evaluating {backbone}/{variant} best.pt on GPU {gpu_id}...")
    cfg = load_config(cfg_path)
    spec = get_spec(backbone)
    model, _ = build(cfg, device=device)
    saved = ck.load(ckpt_path)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()

    res_out = {}
    for split_key in ['standard', 'full']:
        split_name = 'test_standard' if split_key == 'standard' else 'test_full'
        ds = SoccerDataset(cfg, split_name, spec)
        preds = predict(model, ds, cfg, device=device)
        res = evaluate(preds, ds.data)
        auc, orc, gap = compute_orc_gap(preds, ds.data)
        brief = res['brief']
        res_out[split_key] = {
            'mAP': brief['mAP'],
            'g1_auc_pct': auc,
            'oracle_g1_pct': orc,
            'oracle_gap_pct': gap,
            'rej_f1_50': brief['Rej-F1@0.5'],
            'auroc': brief['AUROC'],
        }
        log(f"  {backbone}_{variant} {split_key}: mAP={brief['mAP']:.2f}%, Gap={gap:.2f}%, Rej-F1={brief['Rej-F1@0.5']}%, AUROC={brief['AUROC']}%")

    with open(RESULTS_JSON, "r") as f:
        existing = json.load(f)

    var_desc_map = {
        "E1": "1. E1: Binding-only (Interlayer Anchor)",
        "E2": "2. E2: Lite Binding + Residual Stream",
        "E3": "3. E3: E2 + Slot-Private Innovation (B)",
        "E4": "4. E4: E2 + Uniform Random Refresh (C Control)",
        "E5": "5. E5: E2 + Dominance-Aware Refresh (C)",
        "E6": "6. E6: E2 + Innovation (B) + Dominance (C)",
        "E7": "7. E7: E6 - Binding (bind_coef=0 Ablation)",
        "E8": "8. E8: Strict Serial A -> B (No Refresh C)",
        "E9": "9. E9: Serial C -> A -> B (Refresh -> Re-bound -> Nov)",
    }

    key = f"{backbone}_{variant}"
    existing[key] = {
        'backbone': backbone,
        'variant': variant,
        'description': var_desc_map.get(variant, f"{variant} variant"),
        'standard': res_out['standard'],
        'full': res_out['full'],
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(existing, f, indent=2)
    log(f"Successfully recorded {key} in {RESULTS_JSON}!")

    # Regenerate ablation figure
    try:
        gen_script = DQC_ROOT / "scripts" / "generate_si_bind_variants_ablation_figure.py"
        subprocess.run([PYTHON_BIN, str(gen_script)], check=True)
        log("Regenerated SI_BIND_ABLATION_TABLE_E1_E7 successfully.")
    except Exception as e:
        log(f"Warning: Failed to regenerate ablation figure: {e}")

    # Git push updates
    if TOKEN_PATH.exists():
        try:
            token = TOKEN_PATH.read_text().strip()
            subprocess.run(["git", "add", str(RESULTS_JSON)], cwd=str(DQC_ROOT), check=True)
            subprocess.run(["git", "add", "docs/images/SI_BIND_ABLATION_TABLE_E1_E7.png", "docs/tables/SI_BIND_ABLATION_TABLE_E1_E7.*"], cwd=str(DQC_ROOT), check=True)
            commit_msg = f"feat(ablation): update results and figures with evaluated {backbone}_{variant}"
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=str(DQC_ROOT), check=True)
            push_url = f"https://{token}@github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval.git"
            subprocess.run(["git", "push", push_url, "DQ_CGP_SoccerGMR_SI_Bind_IsolatedLab"], cwd=str(DQC_ROOT), check=True)
            log(f"Pushed {key} results to GitHub remote branch DQ_CGP_SoccerGMR_SI_Bind_IsolatedLab.")
        except Exception as e:
            log(f"Warning during git commit/push: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, help="Backbone name, e.g. bam_detr")
    parser.add_argument("--variant", required=True, help="Variant name, e.g. E2")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (0 or 1)")
    args = parser.parse_args()

    evaluate_checkpoint(args.backbone, args.variant, args.gpu)


if __name__ == "__main__":
    main()
