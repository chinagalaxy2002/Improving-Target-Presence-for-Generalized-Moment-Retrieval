"""CLI Runner to execute full diagnostic evaluation on a single backbone.

Runs in an isolated process to satisfy native module namespace independence.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

DIAG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DIAG_ROOT.parent

if str(DIAG_ROOT) not in sys.path:
    sys.path.insert(0, str(DIAG_ROOT))
REV_SRC = "/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab/revisions/rev_0003_e9_geq/src"
if REV_SRC not in sys.path:
    sys.path.insert(0, REV_SRC)

import torch
from src.model_runner import load_model, run_inference_and_extract
from src.analyzers import analyze_records
from src.presence_analyzer import compute_presence_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run SoccerGMR Diagnostic on a Backbone")
    parser.add_argument("--backbone", type=str, required=True,
                        choices=["moment_detr", "eatr", "qd_detr", "cg_detr", "sim_detr", "bm_detr", "bam_detr", "uvcom"])
    parser.add_argument("--method", type=str, default="m1", choices=["m1", "m2", "m3"],
                        help="m1: Baseline, m2: SI-Bind E9, m3: SI-Bind E9 + GEQ")
    parser.add_argument("--split", type=str, default="test_standard",
                        choices=["test_standard", "test_full"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=str(DIAG_ROOT / "outputs"))
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_dir).resolve()
    raw_dir = out_root / "raw_predictions"
    analysis_dir = out_root / "analysis"
    raw_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.backbone}_{args.method}_{args.split}"
    print(f"[{tag}] Loading model on {args.device}...")
    t0 = time.time()
    model, cfg = load_model(args.backbone, method=args.method, device=args.device)
    print(f"[{tag}] Model loaded in {time.time()-t0:.2f}s. Running inference...")

    t_inf = time.time()
    records = run_inference_and_extract(
        model=model,
        cfg=cfg,
        backbone_name=args.backbone,
        split_name=args.split,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=2
    )
    inf_duration = time.time() - t_inf
    print(f"[{tag}] Extracted {len(records)} samples in {inf_duration:.2f}s ({len(records)/inf_duration:.1f} samples/sec).")

    # Save raw records as JSONL
    raw_file = raw_dir / f"{tag}.jsonl"
    with open(raw_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[{tag}] Saved raw predictions to {raw_file}")

    # Run Analyzers
    print(f"[{tag}] Computing diagnostic metrics (Exp 2, 3, 4, 5 & Presence)...")
    analysis = analyze_records(records, iou_threshold=0.4)
    presence_analysis = compute_presence_metrics(records)
    analysis["presence_diagnostics"] = presence_analysis
    analysis["metadata"] = {
        "backbone": args.backbone,
        "method": args.method,
        "split": args.split,
        "device": args.device,
        "sample_count": len(records),
        "inference_time_sec": inf_duration
    }

    analysis_file = analysis_dir / f"{tag}.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"[{tag}] Saved analysis metrics to {analysis_file}")


if __name__ == "__main__":
    main()
