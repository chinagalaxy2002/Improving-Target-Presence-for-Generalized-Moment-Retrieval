#!/usr/bin/env python3
"""Queue Supervisor for Stage 2 (GEQ with Val mAP Model Selection).
Runs across Dual RTX 3090 (GPU 0 & GPU 1).
Manages tasks:
  (backbone, 'geq_map') -> (backbone, 'geq_map_test')
Saves state to runs/dq_cgp_v3/queue_state_geq_map.json.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/guoxiangyu/miniconda3/envs/gmr/bin/python"
STEP_SCRIPT = str(ROOT / "scripts" / "run_geq_map_step.py")

ALL_BACKBONES = [
    "moment_detr",
    "eatr",
    "qd_detr",
    "cg_detr",
    "bm_detr",
    "bam_detr",
    "sim_detr",
    "uvcom",
]


def load_state(path):
    if not path.is_file():
        return {"completed": [], "running": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"completed": [], "running": {}}


def save_state(path, state):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(path)


def is_pid_alive(pid):
    try:
        status_file = Path(f"/proc/{pid}/status")
        if status_file.is_file():
            for line in status_file.read_text().splitlines():
                if line.startswith("State:"):
                    state_code = line.split()[1]
                    if state_code in ("Z", "T"):  # Zombie or Stopped
                        return False
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def count_external_gpu_jobs():
    """Count running jobs from the original queue or other users."""
    counts = {0: 0, 1: 0}
    try:
        orig_state_p = ROOT / "runs" / "dq_cgp_v3" / "queue_state.json"
        if orig_state_p.is_file():
            data = json.loads(orig_state_p.read_text())
            for job_key, info in data.get("running", {}).items():
                pid = info.get("pid")
                gpu = info.get("gpu")
                if pid and is_pid_alive(pid) and gpu in counts:
                    counts[gpu] += 1
    except Exception:
        pass
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "runs" / "dq_cgp_v3"))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max_per_gpu", type=int, default=3)
    parser.add_argument("--poll_interval", type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_file = root / "queue_state_geq_map.json"

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    max_per_gpu = args.max_per_gpu

    state = load_state(state_file)
    completed = set(state.get("completed", []))
    running = state.get("running", {})

    print(f"GEQ-mAP Queue Supervisor started: GPUs={gpu_ids}, max_per_gpu={max_per_gpu}")
    print(f"Root: {root}")

    active_procs = {}

    while True:
        # 1. Check active processes
        ext_counts = count_external_gpu_jobs()
        finished = []
        for job_key, info in list(running.items()):
            pid = info.get("pid")
            gpu = info.get("gpu")

            proc_dead = False
            rc = 0
            if job_key in active_procs:
                proc, fh, _ = active_procs[job_key]
                poll_rc = proc.poll()
                if poll_rc is not None:
                    proc_dead = True
                    rc = poll_rc
                    fh.close()
                    active_procs.pop(job_key, None)
            elif not is_pid_alive(pid):
                proc_dead = True

            if proc_dead:
                backbone, stage = job_key.split("/")
                if stage == "geq_map":
                    done = (root / backbone / "geq_map" / "complete.json").is_file()
                else:
                    done = (root / backbone / "geq_map_test" / "test_standard" / "complete.json").is_file()

                if done:
                    print(f"[{time.strftime('%X')}] COMPLETED: {job_key} on GPU {gpu}")
                    completed.add(job_key)
                else:
                    print(f"[{time.strftime('%X')}] FAILED or premature exit: {job_key} (rc={rc})")
                finished.append(job_key)

        for job_key in finished:
            running.pop(job_key, None)

        # 2. Count current allocations per GPU
        gpu_load = {g: 0 for g in gpu_ids}
        for g in gpu_ids:
            gpu_load[g] += ext_counts.get(g, 0)
        for job_key, info in running.items():
            g = info.get("gpu")
            if g in gpu_load:
                gpu_load[g] += 1

        # 3. Determine next dispatchable jobs
        candidates = []
        for b in ALL_BACKBONES:
            gmr_complete = root / b / "gmr" / "complete.json"
            if not gmr_complete.is_file():
                continue
            
            k1 = f"{b}/geq_map"
            done_k1 = (root / b / "geq_map" / "complete.json").is_file()
            if done_k1:
                completed.add(k1)
            elif k1 not in completed and k1 not in running:
                candidates.append((b, "geq_map"))
                continue
            
            k2 = f"{b}/geq_map_test"
            done_k2 = (root / b / "geq_map_test" / "test_standard" / "complete.json").is_file()
            if done_k2:
                completed.add(k2)
            elif k1 in completed and k2 not in completed and k2 not in running:
                candidates.append((b, "geq_map_test"))
                continue

        # 4. Dispatch candidates to available GPUs
        for b, stage in candidates:
            avail_gpus = [g for g in gpu_ids if gpu_load[g] < max_per_gpu]
            if not avail_gpus:
                break
            avail_gpus.sort(key=lambda g: gpu_load[g])
            chosen_gpu = avail_gpus[0]

            job_key = f"{b}/{stage}"
            log_path = logs_dir / f"{b}_{stage}.log"
            fh = open(log_path, "a", encoding="utf-8")
            cmd = [PYTHON, STEP_SCRIPT, "--backbone", b, "--stage", stage, "--device", f"cuda:{chosen_gpu}"]

            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            active_procs[job_key] = (proc, fh, chosen_gpu)
            running[job_key] = {"gpu": chosen_gpu, "log": str(log_path), "pid": proc.pid}
            gpu_load[chosen_gpu] += 1

            print(f"[{time.strftime('%X')}] Dispatching {job_key} to GPU {chosen_gpu} (PID {proc.pid}, load {gpu_load[chosen_gpu]}/{max_per_gpu})")

        state["completed"] = sorted(list(completed))
        state["running"] = running
        save_state(state_file, state)

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
