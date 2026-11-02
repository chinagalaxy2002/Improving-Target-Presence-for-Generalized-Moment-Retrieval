#!/usr/bin/env python3
"""Two-GPU recoverable queue: smoke -> train -> frozen-best test for 18 runs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
from execution_plan import job_profile

RUNNER = HERE / "run.py"
VARIANTS_BY_LANE = {
    0: ("baseline", "sepffn", "multi"),
    1: ("shared", "instruction", "adaptive"),
}
BACKBONES = ("eatr", "qd_detr", "cg_detr")
LOCK = threading.Lock()
MAX_ATTEMPTS_PER_STAGE = 3


def tasks(lane):
    # Get one result from every backbone early, then continue the lane's ablations.
    return [(backbone, variant) for variant in VARIANTS_BY_LANE[lane] for backbone in BACKBONES]


def write_status(state):
    with LOCK:
        temporary = HERE / "QUEUE.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(temporary, HERE / "QUEUE.json")


def worker(lane, slot, gpu, state, pending, pending_lock):
    log_dir = HERE / "logs"
    log_dir.mkdir(exist_ok=True)
    while True:
        with pending_lock:
            if not pending:
                break
            backbone, variant = pending.pop(0)
        key = f"{backbone}/{variant}"
        run_dir = HERE / "runs" / backbone / variant
        completed = run_dir / "status.json"
        if completed.exists() and json.loads(completed.read_text()).get("status") == "COMPLETED":
            state["jobs"][key] = {"status": "SKIPPED_COMPLETED", "gpu": gpu}
            write_status(state)
            continue
        state["jobs"][key] = {"status": "RUNNING", "gpu": gpu, "started": time.strftime("%F %T")}
        write_status(state)
        log_path = log_dir / f"{backbone}_{variant}.log"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        failed = None
        with log_path.open("a", buffering=1) as stream:
            for mode in ("smoke", "train", "test"):
                if mode == "smoke":
                    smoke = run_dir / "smoke.json"
                    if smoke.exists() and json.loads(smoke.read_text()).get("status") == "PASSED":
                        stream.write("QUEUE: existing smoke gate passed; skipping smoke\n")
                        continue
                for attempt in range(1, MAX_ATTEMPTS_PER_STAGE + 1):
                    stream.write(f"QUEUE: starting {mode} attempt {attempt}/{MAX_ATTEMPTS_PER_STAGE} on physical GPU {gpu}\n")
                    stream.flush()
                    command = [sys.executable, "-u", str(RUNNER), "--backbone", backbone, "--variant", variant, "--mode", mode]
                    process = subprocess.run(command, cwd=HERE.parent.parent, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                    if process.returncode == 0:
                        failed = None
                        break
                    failed = {"stage": mode, "attempt": attempt, "returncode": process.returncode}
                    state["jobs"][key].update(status="RETRYING" if attempt < MAX_ATTEMPTS_PER_STAGE else "FAILED", **failed)
                    write_status(state)
                    if attempt < MAX_ATTEMPTS_PER_STAGE:
                        time.sleep(5)
                if failed is not None:
                    break
        if failed is None:
            state["jobs"][key].update(status="COMPLETED", finished=time.strftime("%F %T"))
        else:
            state["jobs"][key].update(status="FAILED", finished=time.strftime("%F %T"), log=str(log_path), **failed)
        write_status(state)
    state["workers"][f"gpu{lane}_slot{slot}"] = {"status": "COMPLETED", "gpu": gpu, "finished": time.strftime("%F %T")}
    write_status(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu0", default="0")
    parser.add_argument("--gpu1", default="1")
    args = parser.parse_args()
    missing = []
    for lane in (0, 1):
        for backbone, variant in tasks(lane):
            profile = job_profile(backbone, variant)
            if profile.get('action') == 'reuse_completed_baseline':
                marker = HERE / 'runs' / backbone / variant / 'reused_baseline.json'
                if marker.is_file():
                    continue
            if profile.get('action') == 'restart':
                # Changed batch sizes must pass a fresh full-batch smoke in the
                # queue worker, rather than requiring an out-of-queue GPU job.
                continue
            smoke = HERE / "runs" / backbone / variant / "smoke.json"
            report = json.loads(smoke.read_text()) if smoke.exists() else {}
            if report.get("status") != "PASSED" or report.get("num_workers") != 2:
                missing.append(f"{backbone}/{variant}")
    if missing:
        raise RuntimeError(f"18/18 smoke gate not satisfied: {missing}")
    state = {
        "pid": os.getpid(),
        "started": time.strftime("%F %T"),
        "protocol": "per-job execution_plan.json; defaults seed2023 and microbatch2 x accumulation4; queued QD/CG instruction,multi,adaptive restart batch32 x accumulation1; EaTR baseline reused; num_workers2; full-batch smoke for changed profiles; val/5 epochs; early stop20 epochs; frozen-best test",
        "lanes": {"0": {"status": "RUNNING", "gpu": args.gpu0, "max_concurrent": 2}, "1": {"status": "RUNNING", "gpu": args.gpu1, "max_concurrent": 2}},
        "workers": {},
        "jobs": {},
    }
    write_status(state)
    pending = {lane: tasks(lane) for lane in (0, 1)}
    pending_locks = {lane: threading.Lock() for lane in (0, 1)}
    gpu = {0: args.gpu0, 1: args.gpu1}
    threads = [threading.Thread(target=worker, args=(lane, slot, gpu[lane], state, pending[lane], pending_locks[lane]), daemon=False)
               for lane in (0, 1) for slot in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state["lanes"]["0"]["status"] = "COMPLETED"
    state["lanes"]["1"]["status"] = "COMPLETED"
    failures = sorted(key for key, value in state["jobs"].items() if value["status"] == "FAILED")
    state["failed_jobs"] = failures
    state["status"] = "COMPLETED" if not failures else "COMPLETED_WITH_FAILURES"
    state["finished"] = time.strftime("%F %T")
    write_status(state)


if __name__ == "__main__":
    main()

