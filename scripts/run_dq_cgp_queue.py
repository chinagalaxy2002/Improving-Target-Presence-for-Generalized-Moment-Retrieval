#!/usr/bin/env python3
"""Queue dispatcher for training and evaluating all 8 GMR backbones with DQ-CGPv3.

Manages 2 GPU slots (cuda:0 and cuda:1), training stage 1 (gmr) and paired evaluation (gmr_test).
Automatically resumes on interruption and logs status to runs/dq_cgp_v3/queue_state.json.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ALL_BACKBONES = (
    "moment_detr",
    "eatr",
    "qd_detr",
    "cg_detr",
    "sim_detr",
    "bm_detr",
    "bam_detr",
    "uvcom",
)


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "runs" / "dq_cgp_v3"))
    parser.add_argument(
        "--python", default="/home/guoxiangyu/miniconda3/envs/gmr/bin/python"
    )
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--backbones", default=",".join(ALL_BACKBONES))
    parser.add_argument("--max_per_gpu", type=int, default=3)
    args = parser.parse_args()

    backbones = tuple(b.strip() for b in args.backbones.split(",") if b.strip())
    gpus = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]

    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)

    lock_file = root / ".queue.lock"
    lock = lock_file.open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    configs = {name: ROOT / "configs" / "experiments" / f"{name}.json" for name in backbones}
    for name, p in configs.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing config for {name}: {p}")

    workers = {}
    done = set()
    failure = None

    jobs = [(name, kind) for name in backbones for kind in ("gmr", "gmr_test", "geq_full", "geq_full_test")]

    def complete(job):
        name, kind = job
        directory = root / name / kind
        if kind.endswith("_test"):
            for split in ("test_standard", "test_full"):
                part = directory / split
                marker = part / "complete.json"
                if not marker.is_file():
                    return False
                data = json.loads(marker.read_text())
                for file, key in [
                    ("predictions.jsonl", "predictions_sha256"),
                    ("metrics.json", "metrics_sha256"),
                ]:
                    if not (part / file).is_file() or digest(part / file) != data[key]:
                        return False
            return True
        marker = directory / "complete.json"
        best_pt = directory / "best.pt"
        if not marker.is_file() or not best_pt.is_file():
            return False
        data = json.loads(marker.read_text())
        return data.get("stage") == kind and digest(best_pt) == data["best_sha256"]

    def deps(job):
        name, kind = job
        if kind == "gmr":
            return []
        elif kind in ("gmr_test", "geq_full"):
            return [(name, "gmr")]
        elif kind == "geq_full_test":
            return [(name, "geq_full")]

    def command(job, gpu):
        name, kind = job
        output = root / name / kind
        common = [args.python, "-m", "soccergmr"]
        if kind.endswith("_test"):
            stage = kind[:-5]
            return common + [
                "test",
                "--config",
                str(configs[name]),
                "--output",
                str(output),
                "--checkpoint",
                str(root / name / stage / "best.pt"),
                "--device",
                f"cuda:{gpu}",
            ]
        cmd = common + [
            "train",
            "--config",
            str(configs[name]),
            "--output",
            str(output),
            "--stage",
            kind,
            "--device",
            f"cuda:{gpu}",
        ]
        if kind == "geq_full":
            cmd += ["--baseline", str(root / name / "gmr" / "best.pt")]
        if (output / "last.pt").is_file():
            cmd += ["--resume"]
        return cmd

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.chdir(ROOT)

    # Pre-populate done set from existing complete runs
    for job in jobs:
        if complete(job):
            done.add(job)

    print(f"Starting DQ-CGPv3 Queue with GPUs={gpus}, Backbones={backbones}")
    print(f"Already completed: {['/'.join(j) for j in sorted(done)]}")

    while True:
        for job, (process, gpu, log) in list(workers.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del workers[job]
            if code != 0 or not complete(job):
                failure = {
                    "job": job,
                    "exit_code": code,
                    "log": str(root / "logs" / ("_".join(job) + ".log")),
                }
                print(f"JOB FAILED: {job}, exit={code}, see {failure['log']}")
            else:
                done.add(job)
                print(f"JOB COMPLETED: {job} on GPU {gpu}")
                try:
                    subprocess.run([args.python, str(ROOT / "scripts" / "plot_dq_cgp_progress.py")], capture_output=True)
                    subprocess.run([args.python, str(ROOT / "scripts" / "plot_test_thresholds_dq_cgp.py")], capture_output=True)
                except Exception:
                    pass

        if failure is None:
            for job in jobs:
                if job in done or job in workers:
                    continue
                if not all(x in done for x in deps(job)):
                    continue
                # At most args.max_per_gpu workers per GPU
                counts = {gpu: sum(value[1] == gpu for value in workers.values()) for gpu in gpus}
                available = [gpu for gpu in gpus if counts[gpu] < args.max_per_gpu]
                if not available:
                    break
                gpu = min(available, key=lambda x: counts[x])
                log_path = root / "logs" / ("_".join(job) + ".log")
                log = log_path.open("a", buffering=1)
                cmd = command(job, gpu)
                log.write(f"\nDispatch: {repr(cmd)}\n")
                print(f"Dispatching {job} on GPU {gpu} (log: {log_path})")
                process = subprocess.Popen(
                    cmd, stdout=log, stderr=subprocess.STDOUT, env=env, pass_fds=(lock.fileno(),)
                )
                workers[job] = (process, gpu, log)

        state = {
            "supervisor_pid": os.getpid(),
            "completed": ["/".join(j) for j in sorted(done)],
            "running": {
                "/".join(j): {
                    "pid": p.pid,
                    "gpu": g,
                    "log": str(root / "logs" / ("_".join(j) + ".log")),
                }
                for j, (p, g, _) in workers.items()
            },
            "failure": failure,
            "status": "failed"
            if failure
            else "complete"
            if len(done) == len(jobs)
            else "running",
        }
        atomic_json(root / "queue_state.json", state)

        if not workers and failure:
            return 1
        if not workers and len(done) == len(jobs):
            print("\nALL JOBS COMPLETED SUCCESSFULLY!")
            return 0

        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
