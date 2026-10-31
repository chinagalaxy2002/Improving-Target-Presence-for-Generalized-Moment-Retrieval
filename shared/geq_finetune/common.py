from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PROJECT = Path(os.environ.get("GMR_PROJECT_ROOT", "/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval"))
REPO = Path(os.environ.get("GMR_BACKBONE_REPO", "/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval"))
CORE = Path(os.environ.get("GMR_RUNTIME_ROOT", str(REPO / "backbonefolder/multiroute_gmr")))
OUTPUT = Path(os.environ.get("GMR_GEQ_OUTPUT", str(PROJECT / "geq_finetune_experiments/20261031")))
FIGURES = Path(os.environ.get("GMR_GEQ_FIGURES", str(PROJECT / "newidea/figure/gmr_mr_detr_test_20261030/geq_finetune")))
BACKBONES = ("moment_detr", "eatr", "qd_detr", "cg_detr")
VARIANTS = ("shared", "multi")
REFERENCE_COMMIT = "39ab0eb74427709851476b2bb86b698c8a3f4186"
SEED = 2023
EPOCHS = 4


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    os.replace(temporary, path)


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records))
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_dir(backbone, variant):
    return PROJECT / "mr_experiments" / variant if backbone == "moment_detr" else CORE / "runs" / backbone / variant


def original_prediction(backbone, variant, split="test"):
    root = source_dir(backbone, variant)
    return root / split / "submission.jsonl" if backbone == "moment_detr" else root / f"{split}_predictions.jsonl"


def original_metrics(backbone, variant):
    root = source_dir(backbone, variant)
    return root / "test/metrics_gmiou_0.4.json" if backbone == "moment_detr" else root / "test_metrics.json"
