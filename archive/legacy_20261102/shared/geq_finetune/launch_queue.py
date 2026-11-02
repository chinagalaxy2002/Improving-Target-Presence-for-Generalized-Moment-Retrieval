#!/usr/bin/env python3
"""Persistent GEQ dispatcher: at most two CUDA processes per physical GPU."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from common import BACKBONES, OUTPUT, VARIANTS, dump, read

HERE = Path(__file__).resolve().parent


def gpu_processes():
    mapping = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits",
    ], text=True)
    uuid_to_index = {line.split(",")[1].strip(): line.split(",")[0].strip()
                     for line in mapping.splitlines() if line.strip()}
    applications = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits",
    ], text=True)
    result = {index: set() for index in uuid_to_index.values()}
    for line in applications.splitlines():
        parts = [item.strip() for item in line.split(",", 2)]
        if len(parts) == 3 and parts[0] in uuid_to_index:
            result[uuid_to_index[parts[0]]].add(int(parts[1]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", choices=("baseline", *VARIANTS), default=list(VARIANTS))
    parser.add_argument("--state-name", default="QUEUE.json")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT / args.state_name
    lock = state_path.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Reopening a stopped dispatcher cannot duplicate a still-running child.
    previous = read(state_path)
    for record in previous.get("jobs", {}).values():
        pid = record.get("pid")
        if record.get("status") == "RUNNING" and pid and Path(f"/proc/{pid}/cmdline").is_file():
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
            if str(HERE / "run.py").encode() in command:
                raise RuntimeError(f"Previous job {pid} is still live; no duplicate dispatcher allowed")
    jobs = {f"{backbone}/{variant}": dict(backbone=backbone, variant=variant,
            status="COMPLETED" if read(OUTPUT / backbone / variant / "COMPLETE.json").get("status") == "COMPLETED" else "QUEUED",
            attempts=0) for backbone in BACKBONES for variant in args.variants}
    active = {}
    gpu_indices = (os.environ.get("GPU0", "0"), os.environ.get("GPU1", "1"))
    if len(set(gpu_indices)) != 2:
        raise ValueError("Two distinct physical GPU indices are required")
    print("QUEUE_STARTED", os.getpid(), gpu_indices, flush=True)
    while True:
        for key, (process, stream, gpu) in list(active.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            stream.close()
            job = jobs[key]
            complete = read(OUTPUT / key / "COMPLETE.json").get("status") == "COMPLETED"
            job.update(returncode=returncode, ended_at=time.time(),
                       status="COMPLETED" if returncode == 0 and complete else "FAILED" if job["attempts"] >= 3 else "QUEUED")
            del active[key]
            print("JOB_EXIT", key, job["status"], returncode, flush=True)
            if not args.no_report:
                report = subprocess.run([sys.executable, str(HERE / "collect_and_plot.py")])
                if report.returncode:
                    print("REPORT_RETRY_PENDING", report.returncode, flush=True)
        try:
            occupancy = gpu_processes()
        except (subprocess.CalledProcessError, ValueError) as error:
            # Transient nvidia-smi failure must neither kill dispatch nor launch
            # extra jobs based on an assumed empty GPU.
            print("GPU_POLL_RETRY", str(error), flush=True)
            time.sleep(8)
            continue
        for key, (process, _, gpu) in active.items():
            occupancy.setdefault(gpu, set()).add(process.pid)  # Covers CUDA initialization race.
        for gpu in gpu_indices:
            while len(occupancy.get(gpu, set())) < 2:
                key = next((key for key, job in jobs.items() if job["status"] == "QUEUED" and key not in active), None)
                if key is None:
                    break
                job = jobs[key]
                directory = OUTPUT / key
                directory.mkdir(parents=True, exist_ok=True)
                stream = (directory / "process.log").open("a", buffering=1)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                           OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
                command = [sys.executable, "-u", str(HERE / "run.py"), "--backbone", job["backbone"], "--variant", job["variant"]]
                process = subprocess.Popen(command, cwd=HERE, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[key] = (process, stream, gpu)
                job.update(status="RUNNING", gpu=gpu, pid=process.pid,
                           attempts=job["attempts"] + 1, started_at=time.time(), command=command)
                occupancy.setdefault(gpu, set()).add(process.pid)
                print("JOB_STARTED", key, gpu, process.pid, flush=True)
        terminal = all(job["status"] in ("COMPLETED", "FAILED") for job in jobs.values())
        state = dict(status="COMPLETED" if terminal and all(job["status"] == "COMPLETED" for job in jobs.values())
                     else "FAILED" if terminal else "RUNNING", pid=os.getpid(),
                     updated_at=time.time(), per_gpu_limit=2, includes_external_cuda_processes=True,
                     gpu_occupancy={gpu: sorted(pids) for gpu, pids in occupancy.items()}, jobs=jobs)
        dump(state_path, state)
        if terminal:
            if not args.no_report:
                subprocess.run([sys.executable, str(HERE / "collect_and_plot.py")], check=True)
            print("QUEUE_TERMINAL", state["status"], flush=True)
            return 0 if state["status"] == "COMPLETED" else 1
        time.sleep(8)


if __name__ == "__main__":
    sys.exit(main())
