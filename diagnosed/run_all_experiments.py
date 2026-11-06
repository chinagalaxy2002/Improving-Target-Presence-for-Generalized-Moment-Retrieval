"""Master Parallel Orchestrator for all 8 Backbones across Splits & Methods.

Uses 2 GPU worker queues (cuda:0 and cuda:1) to dispatch tasks concurrently.
Collects and aggregates all outputs into Markdown and JSON artifacts.
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

ROOT = Path(__file__).resolve().parents[1]
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


def run_single_job(backbone: str, method: str, split: str, device: str) -> tuple[str, bool, str]:
    tag = f"{backbone}_{method}_{split}"
    cmd = [
        PYTHON,
        str(ROOT / "diagnosed" / "run_backbone_diagnostics.py"),
        "--backbone", backbone,
        "--method", method,
        "--split", split,
        "--device", device,
        "--batch_size", "16",
    ]
    env = dict(os.environ, PYTHONPATH=f"{ROOT}:{ROOT}/src")
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
    parser.add_argument("--methods", nargs="+", default=["m1", "m2"], choices=["m1", "m2", "m3"])
    parser.add_argument("--splits", nargs="+", default=["test_standard", "test_full"], choices=["test_standard", "test_full", "val"])
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--backbones", nargs="+", default=ALL_BACKBONES)
    args = parser.parse_args()

    # Step 1: Ensure Exp 1 is run
    print("=================================================================")
    print("STEP 1: Running Experiment 1 (Multi-GT Data Audit)...")
    print("=================================================================")
    subprocess.run([PYTHON, str(ROOT / "diagnosed" / "run_exp1_multigt.py")], check=True)

    # Step 2: Build job list for backbones
    print("\n=================================================================")
    print(f"STEP 2: Dispatching Model Diagnostics ({len(args.backbones)} backbones × {len(args.methods)} methods × {len(args.splits)} splits)...")
    print("=================================================================")

    jobs = []
    for method in args.methods:
        for split in args.splits:
            for bb in args.backbones:
                jobs.append((bb, method, split))

    print(f"Total jobs to run: {len(jobs)} on devices {args.devices}")

    # Thread-safe GPU Queue
    q = queue.Queue()
    for dev in args.devices:
        q.put(dev)

    results = {}

    def worker_task(job):
        bb, method, split = job
        dev = q.get()
        try:
            print(f"[DISPATCH] {bb} ({method}, {split}) -> {dev}")
            tag, success, msg = run_single_job(bb, method, split, dev)
            print(f"[FINISHED] {tag} on {dev}: {msg}")
            return tag, success, msg
        finally:
            q.put(dev)
            q.task_done()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
        futures = {executor.submit(worker_task, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            tag, success, msg = future.result()
            results[tag] = (success, msg)

    print("\n=================================================================")
    print("STEP 3: Aggregating Results and Generating Final Tables & Reports...")
    print("=================================================================")
    from diagnosed.src.report_generator import (
        generate_table_a,
        generate_table_b,
        generate_table_c,
        generate_table_d,
    )

    analysis_dir = ROOT / "diagnosed" / "outputs" / "analysis"
    tables_dir = ROOT / "diagnosed" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for method in args.methods:
        for split in args.splits:
            split_label = "Standard Test (1,036 queries)" if split == "test_standard" else ("Full-Decodable Test (2,970 queries)" if split == "test_full" else "Standard Val")
            method_label = "Method 1: GMR Baseline" if method == "m1" else ("Method 2: GMR + DQ-CGPv3" if method == "m2" else "Method 3: DQ-CGPv3 + GEQ")

            analysis_by_bb = {}
            for bb in args.backbones:
                json_path = analysis_dir / f"{bb}_{method}_{split}.json"
                if json_path.is_file():
                    with open(json_path) as f:
                        analysis_by_bb[bb] = json.load(f)

            if not analysis_by_bb:
                continue

            full_label = f"{method_label} | {split_label}"
            t_a = generate_table_a(analysis_by_bb, full_label)
            t_b = generate_table_b(analysis_by_bb, full_label)
            t_c = generate_table_c(analysis_by_bb, full_label)
            t_d = generate_table_d(analysis_by_bb, full_label)

            prefix = f"{method}_{split}"
            (tables_dir / f"table_a_{prefix}.md").write_text(t_a, encoding="utf-8")
            (tables_dir / f"table_b_{prefix}.md").write_text(t_b, encoding="utf-8")
            (tables_dir / f"table_c_{prefix}.md").write_text(t_c, encoding="utf-8")
            (tables_dir / f"table_d_{prefix}.md").write_text(t_d, encoding="utf-8")
            print(f"Generated Tables A, B, C, D for {prefix}")

    print("\nAll tables generated successfully in diagnosed/tables/!")


if __name__ == "__main__":
    main()
