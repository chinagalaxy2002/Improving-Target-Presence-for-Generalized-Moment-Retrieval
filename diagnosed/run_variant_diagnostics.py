#!/usr/bin/env python3
"""Run strict variant diagnostics on arbitrary trained checkpoins for Val / Test splits."""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soccergmr.config import load_config
from soccergmr.backbones.registry import build
from soccergmr.checkpoint import load as ck_load
from diagnosed.src.model_runner import run_inference_and_extract
from diagnosed.src.analyzers import analyze_records


def parse_args():
    parser = argparse.ArgumentParser(description="Strict Variant Diagnostics Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config JSON")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test_standard", "test_full"])
    parser.add_argument("--iou-threshold", type=float, default=0.4, help="tIoU threshold for hits")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save diagnosis artifacts")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading config from {cfg_path}...")
    cfg = load_config(cfg_path)
    backbone = cfg["backbone"]

    print(f"Building model for backbone '{backbone}' on {args.device}...")
    model, _ = build(cfg, device=args.device)

    print(f"Strict loading checkpoint from {ckpt_path}...")
    checkpoint_data = ck_load(ckpt_path)
    state_dict = checkpoint_data["model"] if "model" in checkpoint_data else checkpoint_data
    # Strict loading assertion
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"Extracting inference records on split '{args.split}'...")
    records = run_inference_and_extract(
        model=model,
        cfg=cfg,
        backbone_name=backbone,
        split_name=args.split,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"Analyzing {len(records)} records (tIoU >= {args.iou_threshold})...")
    diagnostics = analyze_records(records, iou_threshold=args.iou_threshold)

    # Calculate false positive count on null queries (M=0)
    neg_records = [r for r in records if r["num_gt"] == 0]
    false_positives = 0
    for r in neg_records:
        exist_score = r.get("pred_exist_score", None)
        if exist_score is not None:
            if exist_score >= 0.5:
                false_positives += 1
        else:
            # Check maximum foreground score in final layer
            final_scores = r["layers"][-1]["native_scores"]
            if max(final_scores) >= 0.5:
                false_positives += 1

    # Extract required fields according to Protocol §5.3
    all_pos_duplication = diagnostics["exp3_duplication"].get("all_pos", {})
    all_pos_ranking = diagnostics["exp4_generation_vs_ranking"].get("all_pos", {})
    multigt_all = diagnostics["exp4_multigt_coverage"].get("all", {})
    multigt_top5 = diagnostics["exp4_multigt_coverage"].get("top5", {})
    slot_stats = diagnostics.get("exp2_slot_stats", {})

    slot_top1_counts = slot_stats.get("slot_top1_count", [])
    slot_hit_counts = slot_stats.get("slot_independent_hit_count", [])

    # Top-2 concentration
    def calc_top2_conc(counts):
        if not counts or sum(counts) == 0:
            return 0.0
        s = sorted(counts, reverse=True)
        return float(sum(s[:2]) / sum(counts) * 100.0)

    top2_top1_conc = calc_top2_conc(slot_top1_counts)
    top2_hit_conc = calc_top2_conc(slot_hit_counts)

    # Layer-wise H and transitions
    layer_H = [layer["independent_hits_sum"] for layer in diagnostics.get("exp5_layers", [])]
    layer_transitions = diagnostics.get("exp5_transitions", [])

    best_val_score = checkpoint_data.get("selector", {}).get("best", None)
    summary = {
        "split": args.split,
        "backbone": backbone,
        "checkpoint_path": str(ckpt_path),
        "config_path": str(cfg_path),
        "best_epoch": checkpoint_data.get("epoch", None),
        "best_score": best_val_score,
        "val_mAP": best_val_score,
        "total_queries": len(records),
        "pos_queries": diagnostics["pos_queries"],
        "neg_queries": diagnostics["neg_queries"],
        "val_GT_total": all_pos_ranking.get("total_gt", 0),
        "val_H_all": all_pos_ranking.get("H_all_sum", 0),
        "val_H_top5": all_pos_ranking.get("H_at_k_sum", {}).get("5", 0),
        "val_missing": all_pos_ranking.get("N_missing_sum", 0),
        "val_omission_top5": all_pos_ranking.get("N_ranking_at_k_sum", {}).get("5", 0),
        "val_multigt_total": multigt_all.get("total_queries", 0),
        "val_multigt_full_all": multigt_all.get("counts", {}).get("full", 0),
        "val_multigt_full_top5": multigt_top5.get("counts", {}).get("full", 0),
        "val_multigt_single_all": multigt_all.get("counts", {}).get("single", 0),
        "val_multigt_zero_all": multigt_all.get("counts", {}).get("zero", 0),
        "val_T": all_pos_duplication.get("raw_hit_candidates_sum", 0),
        "val_D": all_pos_duplication.get("redundant_candidates_sum", 0),
        "val_U": all_pos_duplication.get("unhit_candidates_sum", 0),
        "val_top1_slot_counts": slot_top1_counts,
        "val_independent_hit_slot_counts": slot_hit_counts,
        "val_top2_top1_concentration": top2_top1_conc,
        "val_top2_hit_concentration": top2_hit_conc,
        "val_layer_H": layer_H,
        "val_layer_gain_loss_keep": layer_transitions,
        "val_false_positive_count": false_positives,
    }

    # Save summary and diagnostics
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "diagnostics_detailed.json", "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)

    # Save lightweight predictions jsonl
    with open(out_dir / "records.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Generate Markdown Report
    md_lines = [
        f"# Diagnostic Report: {backbone} ({args.split})",
        f"- **Checkpoint**: `{ckpt_path.name}` (epoch {summary['best_epoch']})",
        f"- **Val mAP**: {summary['best_score']}%",
        f"- **Evaluated Queries**: {len(records)} ({summary['pos_queries']} positive, {summary['neg_queries']} null)",
        f"- **GT Total**: {summary['val_GT_total']}",
        "",
        "## 1. Candidate Generation & Ranking (Decomposition)",
        f"- **H_all (Independent Hits)**: {summary['val_H_all']} / {summary['val_GT_total']} ({summary['val_H_all'] / max(1, summary['val_GT_total']) * 100:.1f}%)",
        f"- **H@5 (Top-5 Independent Hits)**: {summary['val_H_top5']} / {summary['val_GT_total']} ({summary['val_H_top5'] / max(1, summary['val_GT_total']) * 100:.1f}%)",
        f"- **Missing (No Candidate Hits GT)**: {summary['val_missing']} ({summary['val_missing'] / max(1, summary['val_GT_total']) * 100:.1f}%)",
        f"- **Omission@5 (Hit by K but outside Top-5)**: {summary['val_omission_top5']} ({summary['val_omission_top5'] / max(1, summary['val_GT_total']) * 100:.1f}%)",
        "",
        "## 2. Candidate Composition (K = H + D + U)",
        f"- **Total Candidate Hits (T)**: {summary['val_T']}",
        f"- **Independent Hits (H)**: {summary['val_H_all']}",
        f"- **Redundant Candidates (D = T - H)**: {summary['val_D']}",
        f"- **Unhit Candidates (U)**: {summary['val_U']}",
        "",
        "## 3. Multi-GT Complete Event Coverage (M >= 2)",
        f"- **Multi-GT Total Queries**: {summary['val_multigt_total']}",
        f"- **Full Coverage (All K)**: {summary['val_multigt_full_all']} ({summary['val_multigt_full_all'] / max(1, summary['val_multigt_total']) * 100:.1f}%)",
        f"- **Full Coverage (Top-5)**: {summary['val_multigt_full_top5']} ({summary['val_multigt_full_top5'] / max(1, summary['val_multigt_total']) * 100:.1f}%)",
        f"- **Single Event Hit Only**: {summary['val_multigt_single_all']} ({summary['val_multigt_single_all'] / max(1, summary['val_multigt_total']) * 100:.1f}%)",
        f"- **Zero Event Hit**: {summary['val_multigt_zero_all']} ({summary['val_multigt_zero_all'] / max(1, summary['val_multigt_total']) * 100:.1f}%)",
        "",
        "## 4. Slot Index Concentration",
        f"- **Top-2 Slot Top-1 Concentration**: {top2_top1_conc:.1f}%",
        f"- **Top-2 Slot Independent Hit Concentration**: {top2_hit_conc:.1f}%",
        f"- **Slot Top-1 Counts**: `{slot_top1_counts}`",
        f"- **Slot Hit Counts**: `{slot_hit_counts}`",
        "",
        "## 5. Layer-by-Layer Independent Hits (H^(l))",
        f"- **Layer H**: `{layer_H}`",
        f"- **False Positive Null Queries**: {summary['val_false_positive_count']} / {summary['neg_queries']}",
    ]
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Diagnostics complete. Summary saved to {out_dir / 'summary.json'}.")
    print(f"Val H_all: {summary['val_H_all']}, H@5: {summary['val_H_top5']}, Missing: {summary['val_missing']}, MultiGT Full@5: {summary['val_multigt_full_top5']}")


if __name__ == "__main__":
    main()
