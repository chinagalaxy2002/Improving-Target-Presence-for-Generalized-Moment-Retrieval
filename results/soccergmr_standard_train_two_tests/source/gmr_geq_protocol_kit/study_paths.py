"""Paths for the Val-selected shared/multi study; historical outputs stay read-only."""
import hashlib
import json
import os
from pathlib import Path

KIT = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get("GMR_PROJECT_ROOT", str(KIT.parent)))
REPO = Path(os.environ.get("GMR_BACKBONE_REPO", "/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval"))
CORE = Path(os.environ.get("GMR_RUNTIME_ROOT", str(REPO / "backbonefolder/multiroute_gmr")))
OUTPUT = Path(os.environ.get("GMR_PROTOCOL_OUTPUT", str(PROJECT / "geq_protocol_experiments/20261031")))
FIGURES = Path(os.environ.get("GMR_PROTOCOL_FIGURES", str(PROJECT / "newidea/figure/gmr_mr_detr_test_20261030/geq_finetune/val_selection")))
BACKBONES = ("moment_detr", "eatr", "qd_detr", "cg_detr")
VARIANTS = ("shared", "multi", "baseline")
MODES = ("full", "frozen")
SEED = 2023
MAX_EPOCHS = 20


def source_dir(backbone, variant):
    return PROJECT / "mr_experiments" / variant if backbone == "moment_detr" else CORE / "runs" / backbone / variant


def original_prediction(backbone, variant, split):
    root = source_dir(backbone, variant)
    return root / split / "submission.jsonl" if backbone == "moment_detr" else root / f"{split}_predictions.jsonl"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    temporary.replace(path)
