#!/usr/bin/env python3
"""Autonomous E9 Dispatcher, Training Runner, and Continuous Evaluation Loop.

Strict Constraints:
- At most 3 tasks per GPU at all times (max_per_gpu = 3).
- uvcom_E6 is strictly preserved and never modified or rerun.
- Automatically monitors convergence (20 stale epochs early stopping).
- On completion: evaluates Standard Test & Full Test, records in ablation_variants_detailed_results.json,
  regenerates vector figures, and pushes to GitHub branch DQ_CGP_SoccerGMR_SI_Bind_IsolatedLab.
- Launches next queued job as soon as a GPU slot becomes free.
"""

from __future__ import annotations
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, Any

LAB_ROOT = Path("/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab").resolve()
REV_DIR = LAB_ROOT / "revisions" / "rev_0001"
LAUNCH_SCRIPT = LAB_ROOT / "scripts" / "launch_job.sh"
EVAL_SCRIPT = LAB_ROOT / "scripts" / "evaluate_single_checkpoint.py"
PYTHON_BIN = "/home/guoxiangyu/miniconda3/envs/gmr/bin/python"
LOG_FILE = LAB_ROOT / "logs" / "dispatch_e9.log"
RESULTS_JSON = Path("/home/guoxiangyu/SoccerGMR-DQ-CGPv3/docs/tables/ablation_variants_detailed_results.json")


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_already_running(bb: str, var: str) -> bool:
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
        for line in out.splitlines():
            if "soccergmr train" in line and f"/{bb}/{var}/gmr" in line:
                return True
    except Exception:
        pass
    return False


class JobState:
    def __init__(self, backbone: str, variant: str, gpu: int):
        self.backbone = backbone
        self.variant = variant
        self.gpu = gpu
        self.key = f"{backbone}_{variant}"
        self.proc: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None


class E9Dispatcher:
    def __init__(self, max_per_gpu: int = 3):
        self.max_per_gpu = max_per_gpu

        # Target queue: all 8 backbones for E9
        # Ordered for balanced compute distribution across GPUs
        self.target_queue = [
            ("bam_detr", "E9"),
            ("sim_detr", "E9"),
            ("eatr", "E9"),
            ("qd_detr", "E9"),
            ("cg_detr", "E9"),
            ("uvcom", "E9"),
            ("bm_detr", "E9"),
            ("moment_detr", "E9"),
        ]

        self.queue: List[tuple[str, str]] = []
        for bb, var in self.target_queue:
            comp_file = LAB_ROOT / "runs" / "rev_0001" / bb / var / "gmr" / "complete.json"
            if comp_file.exists():
                log(f"{bb}_{var} already COMPLETE. Skipping from launch queue.")
            elif is_already_running(bb, var):
                log(f"{bb}_{var} is ALREADY RUNNING. Will monitor.")
            else:
                self.queue.append((bb, var))

        log(f"Initialized E9 Queue ({len(self.queue)} tasks): {self.queue}")

        self.managed_jobs: Dict[str, JobState] = {}
        self.evaluated_keys = set()
        self._load_evaluated_keys()

    def _load_evaluated_keys(self):
        if RESULTS_JSON.exists():
            try:
                with open(RESULTS_JSON) as f:
                    data = json.load(f)
                self.evaluated_keys = set(data.keys())
            except Exception:
                pass

    def get_gpu_load(self, gpu_id: int) -> int:
        """Count active processes on gpu_id using both internal tracking and nvidia-smi."""
        # 1. Internal tracking
        internal_count = 0
        for k, job in list(self.managed_jobs.items()):
            if job.gpu == gpu_id:
                if job.proc is not None and job.proc.poll() is None:
                    internal_count += 1
                elif is_already_running(job.backbone, job.variant):
                    internal_count += 1

        # 2. External nvidia-smi tracking
        smi_count = 0
        try:
            cmd = ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader"]
            out = subprocess.check_output(cmd, text=True).strip()
            for line in out.splitlines():
                if not line.strip():
                    continue
                bus_id, pid = [x.strip() for x in line.split(",")]
                g = 0 if "17:00.0" in bus_id else (1 if "CA:00.0" in bus_id else 0)
                if g == gpu_id:
                    try:
                        proc_info = subprocess.check_output(["ps", "-p", pid, "-o", "command="], text=True)
                        if "soccergmr train" in proc_info:
                            smi_count += 1
                    except Exception:
                        pass
        except Exception:
            pass

        return max(internal_count, smi_count)

    def launch_next_job(self, gpu_id: int):
        if not self.queue:
            return

        bb, var = self.queue.pop(0)
        key = f"{bb}_{var}"

        log(f"--> LAUNCHING {key} on GPU {gpu_id} (current load: {self.get_gpu_load(gpu_id)}/{self.max_per_gpu})...")
        cmd = [
            "/usr/bin/bash",
            str(LAUNCH_SCRIPT),
            str(gpu_id),
            bb,
            var,
            str(REV_DIR),
        ]

        env = os.environ.copy()
        job = JobState(bb, var, gpu_id)
        job.start_time = time.time()
        job.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.managed_jobs[key] = job
        log(f"    Started {key} with PID {job.proc.pid} on GPU {gpu_id}.")

    def check_for_completed_runs(self):
        """Check all target runs to see if any completed training and need evaluation."""
        self._load_evaluated_keys()

        for bb, var in self.target_queue:
            key = f"{bb}_{var}"
            if key in self.evaluated_keys:
                continue

            run_dir = LAB_ROOT / "runs" / "rev_0001" / bb / var / "gmr"
            comp_file = run_dir / "complete.json"
            best_pt = run_dir / "best.pt"

            if comp_file.exists() and best_pt.exists():
                if not is_already_running(bb, var):
                    log(f"*** Detected COMPLETED run {key}! Triggering Standard & Full Test evaluation...")
                    # Pick whichever GPU has less active load
                    gpu = 1 if self.get_gpu_load(1) <= self.get_gpu_load(0) else 0
                    try:
                        eval_cmd = [
                            PYTHON_BIN,
                            str(EVAL_SCRIPT),
                            "--backbone", bb,
                            "--variant", var,
                            "--gpu", str(gpu),
                        ]
                        subprocess.run(eval_cmd, check=True)
                        self.evaluated_keys.add(key)
                        log(f"*** Evaluation and git sync for {key} COMPLETED successfully!")
                    except Exception as e:
                        log(f"Error evaluating {key}: {e}")

    def run_loop(self):
        log("=" * 70)
        log("Starting Autonomous E9 Dispatcher Loop (STRICT <= 3 tasks per GPU)")
        log(f"Target Queue: {self.queue}")
        log("=" * 70)

        while True:
            # 1. Clean up managed finished processes
            finished_keys = []
            for k, j in self.managed_jobs.items():
                if j.proc is not None:
                    ret = j.proc.poll()
                    if ret is not None and not is_already_running(j.backbone, j.variant):
                        log(f"Managed job {k} process exited with code {ret}.")
                        finished_keys.append(k)
            for k in finished_keys:
                del self.managed_jobs[k]

            # 2. Check for completed runs needing evaluation
            self.check_for_completed_runs()

            # 3. Check GPU capacity and launch queued jobs (at most 1 per check per GPU)
            load_0 = self.get_gpu_load(0)
            if load_0 < self.max_per_gpu and self.queue:
                self.launch_next_job(0)
                time.sleep(2)

            load_1 = self.get_gpu_load(1)
            if load_1 < self.max_per_gpu and self.queue:
                self.launch_next_job(1)
                time.sleep(2)

            # 4. Status snapshot every 45 seconds
            time.sleep(45)

            status_strs = []
            for bb, var in self.target_queue:
                hist_p = LAB_ROOT / "runs" / "rev_0001" / bb / var / "gmr" / "history.jsonl"
                if hist_p.exists() and is_already_running(bb, var):
                    try:
                        with open(hist_p) as f:
                            lines = [json.loads(l) for l in f if l.strip()]
                        if lines:
                            last = lines[-1]
                            best = max(lines, key=lambda x: x.get("selection_score", 0.0))
                            status_strs.append(
                                f"{bb}_{var}: Ep {last['epoch']} (stale {last.get('stale', 0)}/20, best {best.get('selection_score', 0):.2f}%)"
                            )
                    except Exception:
                        pass

            c0 = self.get_gpu_load(0)
            c1 = self.get_gpu_load(1)
            log(
                f"Status [GPU0: {c0}/{self.max_per_gpu}, GPU1: {c1}/{self.max_per_gpu} | Queue left: {len(self.queue)}]: "
                + (" | ".join(status_strs) if status_strs else "Starting...")
            )

            # Check if all 8 E9 jobs are evaluated
            all_e9_done = True
            for bb, var in self.target_queue:
                if f"{bb}_{var}" not in self.evaluated_keys:
                    all_e9_done = False
                    break

            if all_e9_done and not self.queue and not self.managed_jobs:
                log("=" * 70)
                log("ALL 8 E9 BACKBONE EXPERIMENTS HAVE COMPLETED AND BEEN EVALUATED AND PUSHED!")
                log("=" * 70)
                break


if __name__ == "__main__":
    dispatcher = E9Dispatcher(max_per_gpu=3)
    dispatcher.run_loop()
