#!/usr/bin/env python3
"""Gate-I Write Scope Checker: Ensures all write targets and environment variables are strictly inside LAB_ROOT."""
import argparse
import os
from pathlib import Path
import sys

def check_path_inside(path_to_check: Path, allowed_root: Path, name: str = "Path"):
    real_check = path_to_check.resolve()
    real_root = allowed_root.resolve()
    try:
        real_check.relative_to(real_root)
    except ValueError:
        print(f"SECURITY VIOLATION: {name} '{real_check}' is outside allowed root '{real_root}'!")
        return False
    return True

def check_env(allowed_root: Path):
    violations = []
    real_root = allowed_root.resolve()
    env_vars = [
        "TMPDIR", "TEMP", "TMP", "HOME", "XDG_CACHE_HOME",
        "PYTHONPYCACHEPREFIX", "TORCH_HOME", "HF_HOME", "PIP_CACHE_DIR",
        "MPLCONFIGDIR", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR", "WANDB_DIR"
    ]
    for var in env_vars:
        val = os.environ.get(var)
        if val:
            p = Path(val).resolve()
            try:
                p.relative_to(real_root)
            except ValueError:
                violations.append(f"ENV VIOLATION: {var}='{p}' is outside allowed root '{real_root}'!")
    return violations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab", type=Path, required=True, help="Allowed LAB_ROOT directory")
    parser.add_argument("--check-paths", nargs="*", type=Path, help="Specific paths that must be inside LAB_ROOT")
    parser.add_argument("--check-env", action="store_true", help="Check relevant environment variables")
    args = parser.parse_args()

    lab_root = args.lab.resolve()
    failed = False

    if args.check_paths:
        for p in args.check_paths:
            if not check_path_inside(p, lab_root):
                failed = True

    if args.check_env:
        violations = check_env(lab_root)
        for v in violations:
            print(v)
            failed = True

    if failed:
        print("FAIL: Gate-I Write Scope check failed.")
        sys.exit(1)
    else:
        print("PASS: Gate-I Write Scope verified successfully.")
