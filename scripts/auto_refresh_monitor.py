#!/usr/bin/env python3
"""Automated background monitor daemon that refreshes progress_monitor.png every minute."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/guoxiangyu/miniconda3/envs/gmr/bin/python"
PLOT_SCRIPT = str(ROOT / "scripts" / "plot_dq_cgp_progress.py")


def main():
    interval = 60
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except ValueError:
            pass

    print(f"Auto-refresh monitor daemon started: interval={interval}s")
    while True:
        try:
            res = subprocess.run([PYTHON, PLOT_SCRIPT], capture_output=True, text=True, cwd=str(ROOT))
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            if res.returncode == 0:
                print(f"[{now}] Successfully updated progress_monitor.png", flush=True)
            else:
                print(f"[{now}] Error updating plot: {res.stderr}", flush=True)
        except Exception as e:
            print(f"Exception during auto-refresh: {e}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
