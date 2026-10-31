#!/usr/bin/env python3
"""Share two GPU slots per card between joint and frozen GEQ experiments."""
import argparse
import fcntl
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from study_paths import BACKBONES, KIT, MODES, OUTPUT


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def gpu_pids():
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    mapping = {line.split(",")[1].strip(): line.split(",")[0].strip() for line in lines.splitlines()}
    lines = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    result = {index: set() for index in mapping.values()}
    for line in lines.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] in mapping:
            result[mapping[fields[0]]].add(int(fields[1]))
    return result


def process_info(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(state=fields[0], start_ticks=int(fields[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def running_job(job):
    pid = job.get("pid")
    info = process_info(pid) if pid else None
    if not info or info["state"] == "Z":
        return False
    if job.get("start_ticks") not in (None, info["start_ticks"]):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return (str(KIT / "run_study.py").encode() in command
            and job["backbone"].encode() in command and job["mode"].encode() in command)


def next_job(jobs):
    # Alternate schemes within each backbone. Running jobs keep their slots.
    return next((f"{bone}/{mode}" for bone in BACKBONES for mode in MODES
                 if jobs[f"{bone}/{mode}"]["status"] == "QUEUED"), None)


def retire_scheduler(retired):
    if not retired:
        return None
    info = process_info(retired["pid"])
    if not info or info["state"] == "Z" or info["start_ticks"] != retired["start_ticks"]:
        return None
    if any(running_job(job) for job in retired["children"]):
        return retired
    # Keep the old tmux pane alive until its original training children exit.
    # Terminating only the scheduler now cannot hang up an active training job.
    os.kill(retired["pid"], signal.SIGTERM)
    os.kill(retired["pid"], signal.SIGCONT)
    print("PREVIOUS_SCHEDULER_RETIRED", retired["pid"], flush=True)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--takeover-pid", type=int)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # A separate lock permits safe adoption from the original, paused scheduler.
    lock = (OUTPUT / "parallel_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = read(OUTPUT / "QUEUE.json")
    retired = old.get("retired_scheduler")
    if args.takeover_pid:
        info = process_info(args.takeover_pid)
        if old.get("pid") != args.takeover_pid or not info:
            raise RuntimeError("Takeover PID does not match the active queue")
        command = Path(f"/proc/{args.takeover_pid}/cmdline").read_bytes().split(b"\0")
        if not any(token.endswith(b"launch_study_queue.py") for token in command):
            raise RuntimeError("Refusing to pause an unrelated process")
        children = [job for job in old.get("jobs", {}).values() if running_job(job)]
        retired = dict(pid=args.takeover_pid, start_ticks=info["start_ticks"], children=children)
        write(OUTPUT / "TAKEOVER.json", retired)
        os.kill(args.takeover_pid, signal.SIGSTOP)
        print("PREVIOUS_SCHEDULER_PAUSED_TRAINING_UNCHANGED", args.takeover_pid, flush=True)
    elif old.get("pid") and old["pid"] != os.getpid():
        info = process_info(old["pid"])
        if info and info["state"] != "Z":
            raise RuntimeError("Another live queue exists; explicit takeover required")
    jobs = {}
    for mode in MODES:
        for backbone in BACKBONES:
            root = OUTPUT / backbone / mode
            if not read(root / "PREFLIGHT.json").get("passed"):
                raise RuntimeError(f"Missing real-data preflight: {root}")
            key = f"{backbone}/{mode}"
            job = dict(old.get("jobs", {}).get(key, {}))
            job.update(backbone=backbone, mode=mode, attempts=job.get("attempts", 0))
            alive = running_job(job)
            job["status"] = "RUNNING" if alive else (
                "COMPLETED" if read(root / "COMPLETE.json").get("status") == "COMPLETED" else "QUEUED")
            if alive:
                job["start_ticks"] = process_info(job["pid"])["start_ticks"]
            jobs[key] = job
    active = {}
    adopted = {key for key, job in jobs.items() if job["status"] == "RUNNING"}
    print("STUDY_QUEUE_STARTED", os.getpid(), flush=True)
    for key in adopted:
        print("JOB_ADOPTED", key, jobs[key]["pid"], flush=True)
    while True:
        for key in list(adopted):
            if running_job(jobs[key]):
                continue
            done = read(OUTPUT / key / "COMPLETE.json").get("status") == "COMPLETED"
            jobs[key].update(ended_at=time.time(), status="COMPLETED" if done else
                             "FAILED" if jobs[key]["attempts"] >= 3 else "QUEUED")
            adopted.remove(key)
            print("ADOPTED_JOB_EXIT", key, jobs[key]["status"], flush=True)
            subprocess.run([sys.executable, str(KIT / "collect_study.py")])
        for key, (process, stream, gpu) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close()
            done = read(OUTPUT / key / "COMPLETE.json").get("status") == "COMPLETED"
            jobs[key].update(returncode=code, ended_at=time.time(),
                status="COMPLETED" if code == 0 and done else "FAILED" if jobs[key]["attempts"] >= 3 else "QUEUED")
            del active[key]
            print("JOB_EXIT", key, jobs[key]["status"], code, flush=True)
            subprocess.run([sys.executable, str(KIT / "collect_study.py")])
        retired = retire_scheduler(retired)
        try:
            occupied = gpu_pids()
        except (subprocess.CalledProcessError, ValueError) as error:
            print("GPU_POLL_RETRY", str(error), flush=True)
            time.sleep(8)
            continue
        for key, (process, _, gpu) in active.items():
            occupied.setdefault(gpu, set()).add(process.pid)
        for key in adopted:
            occupied.setdefault(jobs[key]["gpu"], set()).add(jobs[key]["pid"])
        failure = any(job["status"] == "FAILED" for job in jobs.values())
        if not failure:
            for gpu in ("0", "1"):
                while len(occupied.get(gpu, set())) < 2:
                    key = next_job(jobs)
                    if key is None:
                        break
                    root = OUTPUT / key
                    stream = (root / "process.log").open("a", buffering=1)
                    job = jobs[key]
                    command = [sys.executable, "-u", str(KIT / "run_study.py"),
                               "--backbone", job["backbone"], "--mode", job["mode"]]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONUNBUFFERED="1",
                               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
                    process = subprocess.Popen(command, cwd=KIT, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    active[key] = (process, stream, gpu)
                    job.update(status="RUNNING", pid=process.pid, gpu=gpu,
                               attempts=job["attempts"] + 1, started_at=time.time())
                    info = process_info(process.pid)
                    job["start_ticks"] = info["start_ticks"] if info else None
                    occupied.setdefault(gpu, set()).add(process.pid)
                    print("JOB_STARTED", key, gpu, process.pid, flush=True)
        done = all(job["status"] == "COMPLETED" for job in jobs.values())
        terminal = done or (failure and not active and not adopted)
        state = dict(status="COMPLETED" if done else "FAILED" if terminal else "RUNNING", stage="parallel",
            pid=os.getpid(), updated_at=time.time(), per_gpu_limit=2, includes_external_jobs=True,
            stage_barrier=None, scheduling="full and frozen share one queue; alternate schemes by backbone",
            retired_scheduler=retired,
            jobs=jobs, gpu_pids={gpu: sorted(pids) for gpu, pids in occupied.items()})
        write(OUTPUT / "QUEUE.json", state)
        old = state
        if terminal:
            subprocess.run([sys.executable, str(KIT / "collect_study.py")], check=True)
            return 0 if done else 1
        time.sleep(8)


if __name__ == "__main__":
    sys.exit(main())
