#!/usr/bin/env python3
"""Run all 18 real-data smoke gates, with at most two processes per GPU."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run.py"
BACKBONES = ("eatr", "qd_detr", "cg_detr")
VARIANTS = ("baseline", "shared", "sepffn", "instruction", "multi", "adaptive")
TASKS = [(backbone, variant) for backbone in BACKBONES for variant in VARIANTS]
LOCK = threading.Lock()


def passed(backbone, variant):
    path = HERE / "runs" / backbone / variant / "smoke.json"
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    return report.get("status") == "PASSED" and report.get("num_workers") == 2


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
            process = subprocess.run(command, cwd=HERE.parent.parent, env=environment, stdout=stream, stderr=subprocess.STDOUT)
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
