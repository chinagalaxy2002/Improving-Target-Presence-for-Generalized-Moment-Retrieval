#!/usr/bin/env python3
"""Fill one GPU0 slot and the third GPU1 slot with the two pending adaptives."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
JOBS = (("qd_detr", 0, 2), ("cg_detr", 0, 2))
LOCK = threading.Lock()
STATE = {"pid": os.getpid(), "started": time.strftime("%F %T"), "limits": {"0": 2, "1": 3}, "jobs": {}}


def dump(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def status(backbone, **values):
    with LOCK:
        STATE["jobs"].setdefault(f"{backbone}/adaptive", {}).update(values)
        dump(HERE / "EXTERNAL_QUEUE.json", STATE)


def gpu_process_count(gpu):
    uuids = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    uuid = next(line.split(",")[1].strip() for line in uuids.splitlines() if int(line.split(",")[0]) == gpu)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    return len({line.split(",")[1].strip() for line in processes.splitlines() if line.startswith(uuid + ",")})


def worker(backbone, gpu, limit):
    run = HERE / "runs" / backbone / "adaptive"
    for attempt in range(1, 4):
        if (run / "status.json").exists() and json.loads((run / "status.json").read_text()).get("status") == "COMPLETED":
            status(backbone, status="COMPLETED", gpu=str(gpu), finished=time.strftime("%F %T"))
            return
        while gpu_process_count(gpu) >= limit:
            status(backbone, status="WAITING_FOR_SLOT", gpu=str(gpu))
            time.sleep(5)
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1")
        command = [sys.executable, "-u", str(HERE / "run.py"), "--backbone", backbone, "--variant", "adaptive", "--mode", "train", "--test-after-training"]
        log = HERE / "logs" / f"{backbone}_adaptive_external.log"
        with log.open("a", buffering=1) as stream:
            stream.write(f"EXTERNAL_DISPATCH {time.strftime('%F %T')} gpu={gpu} attempt={attempt} max_tasks={limit}\n")
            process = subprocess.Popen(command, cwd=HERE.parent.parent, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            status(backbone, status="RUNNING", gpu=str(gpu), child_pid=process.pid, attempt=attempt, log=str(log), started=time.strftime("%F %T"))
            code = process.wait()
        if code == 0:
            status(backbone, status="COMPLETED", gpu=str(gpu), finished=time.strftime("%F %T"))
            return
        status(backbone, status="RETRYING" if attempt < 3 else "FAILED", returncode=code)
        if attempt < 3:
            time.sleep(5)


def main():
    (HERE / "job_locks").mkdir(exist_ok=True)
    with (HERE / "job_locks" / "pending_adaptive_dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Mark only these jobs: the original scheduler will wait on their job
        # locks and skip stages after this pipeline has completed their Test.
        markers = HERE / "external_dispatch"
        markers.mkdir(exist_ok=True)
        for backbone, gpu, limit in JOBS:
            dump(markers / f"{backbone}_adaptive.json", {"dispatcher_pid": os.getpid(), "backbone": backbone, "variant": "adaptive", "gpu": gpu, "max_tasks_on_gpu": limit})
            status(backbone, status="PENDING", gpu=str(gpu))
        threads = [threading.Thread(target=worker, args=job) for job in JOBS]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        STATE["status"] = "COMPLETED" if all(job.get("status") == "COMPLETED" for job in STATE["jobs"].values()) else "COMPLETED_WITH_FAILURES"
        dump(HERE / "EXTERNAL_QUEUE.json", STATE)


if __name__ == "__main__":
    main()
