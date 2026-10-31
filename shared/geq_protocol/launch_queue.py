#!/usr/bin/env python3
"""Parallel full/frozen studies, counting all existing CUDA jobs in the limit."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from study_paths import BACKBONES, KIT, MODES, OUTPUT, VARIANTS, dump


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def occupancy():
    def query(fields, kind):
        return subprocess.check_output(["nvidia-smi", f"--query-{kind}={fields}",
            "--format=csv,noheader,nounits"], text=True).splitlines()
    mapping = {line.split(",")[1].strip(): line.split(",")[0].strip() for line in query("index,uuid", "gpu")}
    result = {index: set() for index in mapping.values()}
    for line in query("gpu_uuid,pid", "compute-apps"):
        uuid, pid = [s.strip() for s in line.split(",")]
        if uuid in mapping:
            result[mapping[uuid]].add(int(pid))
    return result


def report(publish=False):
    script = KIT / "report.py"
    if script.is_file():
        code = subprocess.run([sys.executable, str(script)]).returncode
        print("REPORT_EXIT", code, flush=True)
    script = KIT / "publish.py"
    if publish and script.is_file():
        code = subprocess.run([sys.executable, str(script)]).returncode
        print("PUBLISH_EXIT", code, flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = read(OUTPUT / "QUEUE.json")
        for job in old.get("jobs", {}).values():
            p = Path(f"/proc/{job.get('pid')}/cmdline")
            if job.get("status") == "RUNNING" and p.is_file() and str(KIT / "run_study.py").encode() in p.read_bytes():
                raise RuntimeError("An earlier child is live; refusing duplicate dispatch")
        jobs = {}
        # Run requested shared/multi studies first; baseline controls use the
        # identical checkpoint family for fair four-curve figures afterwards.
        for variant in VARIANTS:
            for backbone in BACKBONES:
                for mode in MODES:
                    key = f"{backbone}/{variant}/{mode}"
                    root = OUTPUT / key
                    root.mkdir(parents=True, exist_ok=True)
                    jobs[key] = dict(backbone=backbone, variant=variant, mode=mode, attempts=0,
                        status="COMPLETED" if read(root / "COMPLETE.json").get("status") == "COMPLETED" else "QUEUED")
        active = {}
        while True:
            changed = False
            for key, (process, stream, gpu) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                done = read(OUTPUT / key / "COMPLETE.json").get("status") == "COMPLETED"
                jobs[key].update(ended_at=time.time(), returncode=code,
                    status="COMPLETED" if done and code == 0 else "FAILED" if jobs[key]["attempts"] >= 3 else "QUEUED")
                del active[key]
                changed = True
                print("JOB_EXIT", key, jobs[key]["status"], code, flush=True)
            try:
                used = occupancy()
            except (subprocess.CalledProcessError, ValueError) as error:
                print("GPU_POLL_RETRY", str(error), flush=True)
                time.sleep(8)
                continue
            for process, _, gpu in active.values():
                used[gpu].add(process.pid)
            failed = any(job["status"] == "FAILED" for job in jobs.values())
            if not failed:
                for gpu in ("0", "1"):
                    while len(used[gpu]) < 2:
                        key = next((k for k, j in jobs.items() if j["status"] == "QUEUED"), None)
                        if key is None:
                            break
                        job = jobs[key]
                        stream = (OUTPUT / key / "process.log").open("a", buffering=1)
                        command = [sys.executable, "-u", str(KIT / "run_study.py"),
                            "--backbone", job["backbone"], "--variant", job["variant"], "--mode", job["mode"]]
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
                        process = subprocess.Popen(command, cwd=KIT, env=env, stdout=stream,
                            stderr=subprocess.STDOUT, start_new_session=True)
                        active[key] = process, stream, gpu
                        job.update(status="RUNNING", pid=process.pid, gpu=gpu,
                            attempts=job["attempts"] + 1, started_at=time.time())
                        used[gpu].add(process.pid)
                        print("JOB_STARTED", key, gpu, process.pid, flush=True)
            done = all(job["status"] == "COMPLETED" for job in jobs.values())
            terminal = done or (failed and not active)
            dump(OUTPUT / "QUEUE.json", dict(status="COMPLETED" if done else "FAILED" if terminal else "RUNNING",
                pid=os.getpid(), updated_at=time.time(), per_gpu_limit=2, includes_external_jobs=True,
                full_and_frozen_parallel=True, jobs=jobs, gpu_pids={g: sorted(p) for g, p in used.items()}))
            if changed or terminal:
                report(publish=True)
            if terminal:
                return 0 if done else 1
            time.sleep(8)


if __name__ == "__main__":
    sys.exit(main())
