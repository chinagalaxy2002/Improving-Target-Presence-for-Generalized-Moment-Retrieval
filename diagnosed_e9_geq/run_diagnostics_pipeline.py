"""Master Parallel Orchestrator for All 8 Backbones across 3 Methods & 2 Splits.

Methods:
- m1: GMR Baseline (native frozen checkpoints)
- m2: GMR + SI-Bind E9 (serial C -> A -> B triad)
- m3: GMR + SI-Bind E9 + GEQ (two-stage presence gating)

Splits:
- test_standard (1,036 queries)
- test_full (2,970 queries)

Dispatches jobs concurrently to cuda:0 and cuda:1.
"""

from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import time
from pathlib import Path

DIAG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DIAG_ROOT.parent
PYTHON = "/home/guoxiangyu/miniconda3/envs/gmr/bin/python"

ALL_BACKBONES = [
    "moment_detr",
    "eatr",
    "qd_detr",
    "cg_detr",
    "sim_detr",
    "bm_detr",
    "bam_detr",
    "uvcom",
]

ALL_METHODS = ["m1", "m2", "m3"]
ALL_SPLITS = ["test_standard", "test_full"]


def run_exp1_audit():
    """Runs Experiment 1: Multi-GT data distribution audit."""
    print("=================================================================")
    print("STEP 1: Running Experiment 1 (Multi-GT Data Distribution Audit)...")
    print("=================================================================")
    from src.data_auditor import audit_split, format_exp1_markdown_table

    splits = {
        "Standard Test": "/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/test.jsonl",
        "Full-Decodable Test": "/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/experiments/gmr-Global-Eventquery/data/full_decodable/Full/test.jsonl",
        "Standard Train": "/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/train.jsonl",
        "Standard Val": "/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/val.jsonl",
    }

    stats_list = []
    exp1_out_dir = DIAG_ROOT / "outputs" / "exp1_multigt"
    exp1_out_dir.mkdir(parents=True, exist_ok=True)

    for name, path_str in splits.items():
        st = audit_split(name, path_str)
        stats_list.append(st)
        slug = name.lower().replace(" ", "_").replace("-", "_")
        (exp1_out_dir / f"{slug}.json").write_text(json.dumps(st, indent=2), encoding="utf-8")

    md_report = format_exp1_markdown_table(stats_list)
    (DIAG_ROOT / "tables" / "exp1_multigt_summary.md").write_text(md_report, encoding="utf-8")
    print("Exp 1 audit complete! Saved to tables/exp1_multigt_summary.md")


def run_single_job(backbone: str, method: str, split: str, device: str) -> tuple[str, bool, str]:
    tag = f"{backbone}_{method}_{split}"
    # Check if already computed
    analysis_file = DIAG_ROOT / "outputs" / "analysis" / f"{tag}.json"
    raw_file = DIAG_ROOT / "outputs" / "raw_predictions" / f"{tag}.jsonl"
    if analysis_file.is_file() and raw_file.is_file() and analysis_file.stat().st_size > 500:
        return tag, True, "Already computed (cached)"

    cmd = [
        PYTHON,
        str(DIAG_ROOT / "run_backbone_diagnostics.py"),
        "--backbone", backbone,
        "--method", method,
        "--split", split,
        "--device", device,
        "--batch_size", "16",
    ]
    env = dict(os.environ, PYTHONPATH=f"{DIAG_ROOT}:/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab/revisions/rev_0003_e9_geq/src")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - t0
    if res.returncode == 0:
        return tag, True, f"Completed in {elapsed:.1f}s"
    else:
        err = res.stderr.strip().split("\n")[-1] if res.stderr else "Unknown error"
        return tag, False, f"Failed: {err}\nSTDOUT: {res.stdout[-300:]}\nSTDERR: {res.stderr[-500:]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=ALL_BACKBONES)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--splits", nargs="+", default=ALL_SPLITS)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing jobs")
    args = parser.parse_args()

    # Step 1: Run Data Audit
    run_exp1_audit()

    # Step 2: Build and dispatch jobs
    print("\n=================================================================")
    print(f"STEP 2: Dispatching Model Diagnostics ({len(args.backbones)} backbones × {len(args.methods)} methods × {len(args.splits)} splits)...")
    print("=================================================================")

    jobs = []
    for method in args.methods:
        for split in args.splits:
            for bb in args.backbones:
                if args.force:
                    # Clear cache
                    for p in [DIAG_ROOT / "outputs" / "analysis" / f"{bb}_{method}_{split}.json",
                              DIAG_ROOT / "outputs" / "raw_predictions" / f"{bb}_{method}_{split}.jsonl"]:
                        if p.exists():
                            p.unlink()
                jobs.append((bb, method, split))

    print(f"Total jobs to execute: {len(jobs)} on devices {args.devices}")

    q = queue.Queue()
    for dev in args.devices:
        q.put(dev)

    results = {}

    def worker_task(job):
        bb, method, split = job
        dev = q.get()
        try:
            print(f"[DISPATCH] {bb} ({method}, {split}) -> {dev}", flush=True)
            tag, success, msg = run_single_job(bb, method, split, dev)
            status_str = "SUCCESS" if success else "FAILED"
            print(f"[{status_str}] {tag} on {dev}: {msg}", flush=True)
            return tag, success, msg
        finally:
            q.put(dev)
            q.task_done()

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
        futures = {executor.submit(worker_task, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            tag, success, msg = future.result()
            results[tag] = (success, msg)

    elapsed_all = time.time() - t_start
    success_cnt = sum(1 for s, _ in results.values() if s)
    print(f"\nAll jobs finished in {elapsed_all:.1f}s: {success_cnt}/{len(jobs)} successful!")

    # Step 3: Run master supplementary analysis and figure rendering
    print("\n=================================================================")
    print("STEP 3: Aggregating Multi-Method Comparisons & Generating Publication Figures...")
    print("=================================================================")
    cmd_agg = [PYTHON, str(DIAG_ROOT / "run_supplementary_analysis.py")]
    subprocess.run(cmd_agg, check=True)
    print("\nDiagnostic pipeline fully completed! Check outputs, tables, and figures under diagnosed_e9_geq/")


if __name__ == "__main__":
    main()
