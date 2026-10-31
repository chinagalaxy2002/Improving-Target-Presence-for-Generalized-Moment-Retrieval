#!/usr/bin/env python3
"""QD-DETR-GMR fixed-backbone entry point for smoke, training, and testing."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CORE = Path(__file__).resolve().parents[2] / "run.py"
raise SystemExit(subprocess.call([sys.executable, "-u", str(CORE), "--backbone", "qd_detr", *sys.argv[1:]]))
