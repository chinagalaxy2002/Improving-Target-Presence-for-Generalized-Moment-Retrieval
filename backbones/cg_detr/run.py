#!/usr/bin/env python3
"""CG-DETR-GMR fixed-backbone entry point for smoke, training, and testing."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


CORE = Path(__file__).resolve().parents[2] / "shared" / "run.py"
raise SystemExit(subprocess.call([sys.executable, "-u", str(CORE), "--backbone", "cg_detr", *sys.argv[1:]]))
