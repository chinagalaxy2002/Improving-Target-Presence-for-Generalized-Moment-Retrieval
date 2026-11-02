#!/usr/bin/env python3
"""Run all 18 real-data smoke gates, with at most two processes per GPU."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
HERE = Path(os.environ.get('GMR_RUNTIME_ROOT', CODE_ROOT))
from execution_plan import job_profile

RUNNER = CODE_ROOT / "run.py"
BACKBONES = ("eatr", "qd_detr", "cg_detr")
VARIANTS = ("baseline", "shared", "sepffn", "instruction", "multi", "adaptive")
TASKS = [(backbone, variant) for backbone in BACKBONES for variant in VARIANTS]
LOCK = threading.Lock()


def passed(backbone, variant):
    profile = job_profile(backbone, variant)
    run = HERE / 'runs' / backbone / variant
    if profile.get('action') == 'reuse_completed_baseline':
        return all((run / name).is_file() for name in ('reused_baseline.json', 'best.ckpt', 'test_metrics.json'))
    path = HERE / "runs" / backbone / variant / "smoke.json"
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    valid = report.get("status") == "PASSED" and report.get("num_workers") == 2
    if profile.get('action') == 'restart':
        valid = valid and report.get('measured_batch_size') == profile['microbatch']
    return valid


def worker(gpu, pending, results):
    while True:
        with LOCK:
            if not pending:
                return
            backbone, variant = pending.pop(0)
        if passed(backbone, variant):
            results[f"{backbone}/{variant}"] = "PASSED_EXISTING"
            continue
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log = HERE / "logs" / f"smoke_{backbone}_{variant}.log"
        log.parent.mkdir(exist_ok=True)
        command = [sys.executable, "-u", str(RUNNER), "--backbone", backbone, "--variant", variant, "--mode", "smoke"]
        with log.open("w") as stream:
            process = subprocess.run(command, cwd=CODE_ROOT.parent, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        results[f"{backbone}/{variant}"] = "PASSED" if process.returncode == 0 and passed(backbone, variant) else f"FAILED({process.returncode})"


def main():
    gpu0 = os.getenv("GPU0", "0")
    gpu1 = os.getenv("GPU1", "1")
    pending = list(TASKS)
    results = {}
    threads = [threading.Thread(target=worker, args=(gpu, pending, results)) for gpu in (gpu0, gpu0, gpu1, gpu1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    report = {"timestamp": time.strftime("%F %T"), "passed": sum(value.startswith("PASSED") for value in results.values()), "total": len(TASKS), "results": dict(sorted(results.items()))}
    (HERE / "SMOKE_MATRIX.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
