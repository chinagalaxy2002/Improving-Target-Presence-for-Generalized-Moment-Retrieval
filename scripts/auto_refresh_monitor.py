#!/usr/bin/env python3
"""Automated background monitor daemon that refreshes progress_monitor.png, threshold plots, and ablation tables every minute."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/guoxiangyu/miniconda3/envs/gmr/bin/python"
PLOT_SCRIPT = str(ROOT / "scripts" / "plot_dq_cgp_progress.py")
PLOT_THRESHOLDS_SCRIPT = str(ROOT / "scripts" / "plot_test_thresholds_dq_cgp.py")
ABLATION_TABLE_SCRIPT = str(ROOT / "scripts" / "ablation_study_table.py")
GENERATE_SVG_PDF_SCRIPT = str(ROOT / "scripts" / "generate_ablation_svg_pdf.py")
ABLATION_GEQ_MAP_SCRIPT = str(ROOT / "scripts" / "ablation_study_table_geq_map.py")
GENERATE_GEQ_MAP_SVG_SCRIPT = str(ROOT / "scripts" / "generate_ablation_svg_pdf_geq_map.py")


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
            res1 = subprocess.run([PYTHON, PLOT_SCRIPT], capture_output=True, text=True, cwd=str(ROOT))
            res2 = subprocess.run([PYTHON, PLOT_THRESHOLDS_SCRIPT], capture_output=True, text=True, cwd=str(ROOT))
            res3 = subprocess.run([PYTHON, ABLATION_TABLE_SCRIPT], capture_output=True, text=True, cwd=str(ROOT))
            res4 = subprocess.run([PYTHON, GENERATE_SVG_PDF_SCRIPT], capture_output=True, text=True, cwd=str(ROOT))
            res5 = subprocess.run([PYTHON, ABLATION_GEQ_MAP_SCRIPT], capture_output=True, text=True, cwd=str(ROOT))
            res6 = subprocess.run([PYTHON, GENERATE_GEQ_MAP_SVG_SCRIPT], capture_output=True, text=True, cwd=str(ROOT))
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Successfully updated progress_monitor.png, *_TEST.png, and ABLATION_TABLE.*", flush=True)
            if res1.returncode == 0 and res2.returncode == 0 and res3.returncode == 0 and res4.returncode == 0:
                print(f"[{now}] Successfully updated progress_monitor.png, *_TEST.png, and ABLATION_TABLE.*", flush=True)
            else:
                print(f"[{now}] Update outputs: r1={res1.returncode} r2={res2.returncode} r3={res3.returncode} r4={res4.returncode}", flush=True)
        except Exception as e:
            print(f"Exception during auto-refresh: {e}", flush=True)
        time.sleep(interval)


if __name__ == "__main__":
    main()
