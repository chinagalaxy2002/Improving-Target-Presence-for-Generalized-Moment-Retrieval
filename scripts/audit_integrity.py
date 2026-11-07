#!/usr/bin/env python3
"""Gate-I Integrity Auditor: Generates and verifies SHA-256 hashes of original codebase and checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ORIGINAL_REPO = Path("/home/guoxiangyu/SoccerGMR-DQ-CGPv3").resolve()
CHECKPOINTS_DOC = ORIGINAL_REPO / "docs" / "checkpoints_and_paths.md"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def get_original_code_files(repo_path: Path):
    extensions = {".py", ".json", ".md", ".sh", ".yaml", ".toml", ".txt"}
    files = []
    for root, dirs, filenames in os.walk(repo_path):
        # Skip git, LAB_ROOT (if present), __pycache__, .pytest_cache
        dirs[:] = [d for d in dirs if d not in {".git", "LAB_ROOT", "__pycache__", ".pytest_cache"}]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                p = Path(root) / fname
                files.append(p)
    return sorted(files)

def get_checkpoint_files(doc_path: Path):
    if not doc_path.exists():
        return []
    import re
    text = doc_path.read_text(encoding="utf-8")
    matches = re.findall(r"/home/guoxiangyu/[^`\s\)]+\.(?:ckpt|pt)", text)
    unique_paths = sorted(set(matches))
    return [Path(p) for p in unique_paths if Path(p).is_file()]

def generate_baseline(output_path: Path):
    print(f"Generating baseline hashes into {output_path}...")
    code_files = get_original_code_files(ORIGINAL_REPO)
    checkpoint_files = get_checkpoint_files(CHECKPOINTS_DOC)
    
    code_hashes = {}
    for cf in code_files:
        rel = str(cf.relative_to(ORIGINAL_REPO))
        code_hashes[rel] = {
            "path": str(cf),
            "sha256": sha256_file(cf),
            "size": cf.stat().st_size,
        }
        
    ckpt_hashes = {}
    for ckp in checkpoint_files:
        ckpt_hashes[str(ckp)] = {
            "path": str(ckp),
            "sha256": sha256_file(ckp),
            "size": ckp.stat().st_size,
        }
        
    data = {
        "original_repo": str(ORIGINAL_REPO),
        "code_files_count": len(code_hashes),
        "checkpoint_files_count": len(ckpt_hashes),
        "code_hashes": code_hashes,
        "checkpoint_hashes": ckpt_hashes,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Baseline saved: {len(code_hashes)} code files, {len(ckpt_hashes)} checkpoint files.")

def verify_baseline(baseline_path: Path) -> bool:
    print(f"Verifying against baseline {baseline_path}...")
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    mismatches = []
    # Verify code files
    for rel, meta in data["code_hashes"].items():
        p = Path(meta["path"])
        if not p.exists():
            mismatches.append(f"MISSING: {p}")
            continue
        curr_sha = sha256_file(p)
        if curr_sha != meta["sha256"]:
            mismatches.append(f"MODIFIED CODE: {p} (expected {meta['sha256'][:10]}, got {curr_sha[:10]})")
            
    # Verify checkpoint files (sample check or full check)
    for path_str, meta in data["checkpoint_hashes"].items():
        p = Path(path_str)
        if not p.exists():
            mismatches.append(f"MISSING CKPT: {p}")
            continue
        curr_sha = sha256_file(p)
        if curr_sha != meta["sha256"]:
            mismatches.append(f"MODIFIED CKPT: {p} (expected {meta['sha256'][:10]}, got {curr_sha[:10]})")
            
    if mismatches:
        print("FAIL: Baseline verification failed with modifications:")
        for m in mismatches:
            print("  ", m)
        return False
    print("PASS: Baseline verification successful. Zero modifications detected.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--action", choices=["generate", "verify"], default="generate")
    args = parser.parse_args()
    
    if args.action == "generate":
        generate_baseline(args.baseline)
    elif args.action == "verify":
        if not verify_baseline(args.baseline):
            sys.exit(1)
