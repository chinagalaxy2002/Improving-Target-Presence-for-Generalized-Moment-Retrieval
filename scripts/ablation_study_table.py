#!/usr/bin/env python3
"""Ablation Study Reporter for SoccerGMR:
Compares:
1. Baseline GMR
2. GMR + DQ-CGPv3
3. GMR + DQ-CGPv3 + GEQ Full

Metrics reported across Standard Test (1036 queries) and Full Test (2970 queries):
- mAP (%)
- G-mIoU@1 (0-1 Integral / AUC)
- Rej-F1 (%) (at canonical tau=0.5, plus tau=0.4, 0.6)
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
HISTORICAL_RUNS_DIR = Path("/home/guoxiangyu/SoccerGMR-GEQ-additional/results/test_thresholds/runs")
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
    count = len(gt)
    by_qid = {p["qid"]: p for p in predictions}
    ordered = [by_qid[g["qid"]] for g in gt if g["qid"] in by_qid]
    if len(ordered) != count:
        return None
    scores = []
    rejected = []
    accepted = []
    for p, g in zip(ordered, gt):
        s, _ = get_existence_score(p)
        scores.append(float(s))
        rejected.append(_compute_set_iou_score([], g["relevant_windows"]))
        clean = _clean_pred_windows(p["pred_relevant_windows"], max_pred_windows=10)
        accepted.append(_compute_set_iou_score([w[:2] for w in clean[:1]], g["relevant_windows"]))

    ts = [round(i / 100.0, 2) for i in range(101)]
    ys = []
    for tau in ts:
        val = sum(a if s > tau else r for a, s, r in zip(accepted, scores, rejected)) / count
        ys.append(val)
    integral = float(np.trapezoid(ys, ts))
    return round(integral * 100, 2), round(integral, 4)


def get_baseline_metrics(name, split_key, gt, pdata):
    # Method 1: Baseline GMR
    split_folder = "test_standard" if split_key == "standard" else "test_full"
    
    # 1. Check expansion formal runs (bm_detr, bam_detr, uvcom)
    m_file = HISTORICAL_EXPANSION / name / "gmr_test" / split_folder / "metrics.json"
    p_file = HISTORICAL_EXPANSION / name / "gmr_test" / split_folder / "predictions.jsonl"
    if m_file.is_file() and p_file.is_file():
        m = json.loads(m_file.read_text())
        preds = read_jsonl(p_file)
        auc_pct, auc_raw = compute_g1_integral(preds, gt)
        brief = m.get("brief", {})
        return {
            "mAP": brief.get("mAP"),
            "g1_auc_pct": auc_pct,
            "g1_auc_raw": auc_raw,
            "rej_f1_50": brief.get("Rej-F1@0.5"),
            "rej_f1_40": brief.get("Rej-F1@0.4"),
            "rej_f1_60": brief.get("Rej-F1@0.6"),
            "auroc": brief.get("AUROC"),
        }

    # 2. Check test_thresholds/runs/
    m_file2 = HISTORICAL_RUNS_DIR / name / split_key / "gmr" / "metrics.json"
    if m_file2.is_file() and pdata:
        m = json.loads(m_file2.read_text())
        brief = m.get("brief", {})
        # G1 integral from PLOT_DATA
        row = pdata.get("splits", {}).get(split_key, {}).get("backbones", {}).get(name, {}).get("gmr", {})
        ys = row.get("gmiou", {}).get("1", [])
        ts = [round(i / 100.0, 2) for i in range(101)]
        integral = float(np.trapezoid(ys, ts)) if ys else 0.0
        return {
            "mAP": brief.get("mAP", row.get("mAP")),
            "g1_auc_pct": round(integral * 100, 2),
            "g1_auc_raw": round(integral, 4),
            "rej_f1_50": brief.get("Rej-F1@0.5"),
            "rej_f1_40": brief.get("Rej-F1@0.4"),
            "rej_f1_60": brief.get("Rej-F1@0.6"),
            "auroc": brief.get("AUROC"),
        }

    return None


def get_evaluated_metrics(root, name, stage, split_key, gt):
    split_folder = "test_standard" if split_key == "standard" else "test_full"
    m_file = root / name / stage / split_folder / "metrics.json"
    p_file = root / name / stage / split_folder / "predictions.jsonl"
    if not (m_file.is_file() and p_file.is_file()):
        return None
    m = json.loads(m_file.read_text())
    preds = read_jsonl(p_file)
    auc_pct, auc_raw = compute_g1_integral(preds, gt)
    brief = m.get("brief", {})
    return {
        "mAP": brief.get("mAP"),
        "g1_auc_pct": auc_pct,
        "g1_auc_raw": auc_raw,
        "rej_f1_50": brief.get("Rej-F1@0.5"),
        "rej_f1_40": brief.get("Rej-F1@0.4"),
        "rej_f1_60": brief.get("Rej-F1@0.6"),
        "auroc": brief.get("AUROC"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "runs" / "dq_cgp_v3"))
    args = parser.parse_args()

    root = Path(args.root).resolve()

    gts = {k: read_jsonl(p) for k, p in GT_PATHS.items()}
    pdata = json.loads(HISTORICAL_PLOT_DATA.read_text()) if HISTORICAL_PLOT_DATA.is_file() else {}

    results = {}

    for name, display_name in BACKBONES:
        results[name] = {"display_name": display_name, "splits": {}}
        for split_key in ["standard", "full"]:
            gt = gts[split_key]
            
            # Method 1: baseline gmr
            base = get_baseline_metrics(name, split_key, gt, pdata)
            # Method 2: gmr + dqcgpv3
            dq_gmr = get_evaluated_metrics(root, name, "gmr_test", split_key, gt)
            # Method 3: gmr + dqcgpv3 + geq
            dq_geq = get_evaluated_metrics(root, name, "geq_full_test", split_key, gt)

            results[name]["splits"][split_key] = {
                "gmr": base,
                "gmr_dqcgpv3": dq_gmr,
                "gmr_dqcgpv3_geq": dq_geq,
            }

    # Print Formatted Markdown Tables
    print("\n" + "=" * 95)
    print("SOCCERGMR ABLATION STUDY: GMR vs GMR+DQ-CGPv3 vs GMR+DQ-CGPv3+GEQ")
    print("=" * 95)

    for split_key, split_label in [("standard", "Standard Test (1036 queries)"), ("full", "Full-Decodable Test (2970 queries)")]:
        print(f"\n### {split_label}\n")
        print("| 骨干模型 (Backbone) | 方法配置 (Method) | mAP (%) | G-mIoU@1 (0-1积分 / AUC) | Rej-F1@0.5 (%) | Rej-F1@0.4 / 0.6 | AUROC (%) |")
        print("|---|---|---|---|---|---|---|")

        for name, display_name in BACKBONES:
            s_data = results[name]["splits"][split_key]
            
            # Print GMR
            m1 = s_data["gmr"]
            if m1:
                print(f"| **{display_name}** | 1. GMR Baseline | {m1['mAP']:.2f}% | **{m1['g1_auc_pct']:.2f}%** ({m1['g1_auc_raw']:.4f}) | {m1['rej_f1_50']:.2f}% | {m1['rej_f1_40']:.2f}% / {m1['rej_f1_60']:.2f}% | {m1['auroc']:.2f}% |")
            else:
                print(f"| **{display_name}** | 1. GMR Baseline | - | - | - | - | - |")

            # Print GMR + DQ-CGPv3
            m2 = s_data["gmr_dqcgpv3"]
            if m2:
                d_map2 = f"({m2['mAP'] - m1['mAP']:+.2f}%)" if m1 and m1.get('mAP') is not None else ""
                d_auc2 = f"({m2['g1_auc_pct'] - m1['g1_auc_pct']:+.2f}%)" if m1 and m1.get('g1_auc_pct') is not None else ""
                print(f"| | 2. GMR + DQ-CGPv3 | {m2['mAP']:.2f}% {d_map2} | **{m2['g1_auc_pct']:.2f}%** ({m2['g1_auc_raw']:.4f}) {d_auc2} | {m2['rej_f1_50']:.2f}% | {m2['rej_f1_40']:.2f}% / {m2['rej_f1_60']:.2f}% | {m2['auroc']:.2f}% |")
            else:
                print(f"| | 2. GMR + DQ-CGPv3 | 评测中 / 待完成 | - | - | - | - |")

            # Print GMR + DQ-CGPv3 + GEQ
            m3 = s_data["gmr_dqcgpv3_geq"]
            if m3:
                d_map3 = f"({m3['mAP'] - m1['mAP']:+.2f}%)" if m1 and m1.get('mAP') is not None else ""
                d_auc3 = f"({m3['g1_auc_pct'] - m1['g1_auc_pct']:+.2f}%)" if m1 and m1.get('g1_auc_pct') is not None else ""
                print(f"| | 3. GMR + DQ-CGPv3 + GEQ | **{m3['mAP']:.2f}%** {d_map3} | **{m3['g1_auc_pct']:.2f}%** ({m3['g1_auc_raw']:.4f}) {d_auc3} | **{m3['rej_f1_50']:.2f}%** | {m3['rej_f1_40']:.2f}% / {m3['rej_f1_60']:.2f}% | **{m3['auroc']:.2f}%** |")
            else:
                print(f"| | 3. GMR + DQ-CGPv3 + GEQ | 评测中 / 待完成 | - | - | - | - |")
            print("|" + "---|" * 7)

    # Save to JSON
    out_json = root / "results" / "ablation_study_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved structured ablation results to {out_json}")


if __name__ == "__main__":
    main()
