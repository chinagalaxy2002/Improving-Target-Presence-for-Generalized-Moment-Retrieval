#!/usr/bin/env python3
"""Ablation Study Reporter for SoccerGMR with GEQ mAP Selection:
Compares:
1. Baseline GMR
2. GMR + DQ-CGPv3
3. GMR + DQ-CGPv3 + GEQ (Val Mean G1 Selection - Official)
4. GMR + DQ-CGPv3 + GEQ (Val mAP Selection - Proposed)

Metrics reported across Standard Test (1036 queries) and Full Test (2970 queries):
- mAP (%)
- G-mIoU@1 (0-1 Integral / AUC)
- Rej-F1@0.5 (%)
- AUROC (%)
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from soccergmr.io import read_jsonl
from soccergmr.evaluation.official.metrics import _clean_pred_windows, _compute_set_iou_score, get_existence_score

HISTORICAL_PLOT_DATA = Path("/home/guoxiangyu/SoccerGMR-GEQ-additional/results/test_thresholds/PLOT_DATA.json")
HISTORICAL_EXPANSION = Path("/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal")

GT_PATHS = {
    "standard": "/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/test.jsonl",
    "full": "/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/experiments/gmr-Global-Eventquery/data/full_decodable/Full/test.jsonl",
}

BACKBONES = [
    ("moment_detr", "Moment-DETR"),
    ("eatr", "EaTR"),
    ("qd_detr", "QD-DETR"),
    ("cg_detr", "CG-DETR"),
    ("bm_detr", "BM-DETR"),
    ("bam_detr", "BAM-DETR"),
    ("sim_detr", "Sim-DETR"),
    ("uvcom", "UVCOM"),
]


def compute_g1_integral(predictions, gt):
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    count = len(gt)
    by_qid = {p["qid"]: p for p in predictions}
    ordered = [by_qid[g["qid"]] for g in gt if g["qid"] in by_qid]
    rejected, accepted, scores = [], [], []
    for p, g in zip(ordered, gt):
        s, _ = get_existence_score(p)
        scores.append(float(s))
        rejected.append(_compute_set_iou_score([], g["relevant_windows"]))
        clean = _clean_pred_windows(p["pred_relevant_windows"], max_pred_windows=10)
        accepted.append(_compute_set_iou_score([w[:2] for w in clean[:1]], g["relevant_windows"]))
    thresholds = [round(t * 0.01, 2) for t in range(101)]
    scores_arr = np.array(scores)[:, None]
    mask = (scores_arr > np.array(thresholds))
    acc_arr = np.array(accepted)[:, None]
    rej_arr = np.array(rejected)[:, None]
    ious = np.where(mask, acc_arr, rej_arr)
    curve = ious.mean(axis=0).tolist()
    auc = float(trapz_fn(curve, thresholds))
    return round(auc * 100, 2), round(auc, 4)


def get_evaluated_metrics(root, name, stage_name, split_key, gt):
    split_folder = "test_standard" if split_key == "standard" else "test_full"
    m_file = root / name / stage_name / split_folder / "metrics.json"
    p_file = root / name / stage_name / split_folder / "predictions.jsonl"
    if not (m_file.is_file() and p_file.is_file()):
        return None
    try:
        m = json.loads(m_file.read_text())
        preds = read_jsonl(p_file)
        auc_pct, auc_raw = compute_g1_integral(preds, gt)
        brief = m.get("brief", {})
        return {
            "mAP": brief.get("mAP"),
            "g1_auc_pct": auc_pct,
            "rej_f1_50": brief.get("Rej-F1@0.5"),
            "auroc": brief.get("AUROC"),
        }
    except Exception:
        return None


def main():
    root = ROOT / "runs" / "dq_cgp_v3"
    gts = {k: read_jsonl(p) for k, p in GT_PATHS.items()}
    out_json = root / "results" / "ablation_study_geq_map_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Load existing 3-stage ablation results
    base_results_file = root / "results" / "ablation_study_results.json"
    base_data = json.loads(base_results_file.read_text()) if base_results_file.is_file() else {}

    results = {}
    for name, display_name in BACKBONES:
        results[name] = {"display_name": display_name, "splits": {}}
        for split_key in ["standard", "full"]:
            gt = gts[split_key]
            # Inherit Method 1, 2, 3
            m1 = base_data.get(name, {}).get("splits", {}).get(split_key, {}).get("gmr")
            m2 = base_data.get(name, {}).get("splits", {}).get(split_key, {}).get("gmr_dqcgpv3")
            m3 = base_data.get(name, {}).get("splits", {}).get(split_key, {}).get("gmr_dqcgpv3_geq")
            # Method 4: geq with mAP selection
            m4 = get_evaluated_metrics(root, name, "geq_map_test", split_key, gt)

            results[name]["splits"][split_key] = {
                "gmr": m1,
                "gmr_dqcgpv3": m2,
                "gmr_dqcgpv3_geq_g1": m3,
                "gmr_dqcgpv3_geq_map": m4,
            }

    out_json.write_text(json.dumps(results, indent=2))
    print(f"Saved geq_map ablation results to {out_json}")


if __name__ == "__main__":
    main()
