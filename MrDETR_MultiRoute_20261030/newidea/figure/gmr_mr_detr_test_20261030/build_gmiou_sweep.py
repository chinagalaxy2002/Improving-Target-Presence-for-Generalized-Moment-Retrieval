"""Compute G-mIoU@3 over the complete existence-threshold domain."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
EVAL_DIR = ROOT / "generalized-moment-retrieval/eval"
sys.path.insert(0, str(EVAL_DIR))

from metrics import compute_G_mIoU, prepare_submission_for_gmiou  # noqa: E402
from normalization import normalize_ground_truth  # noqa: E402


ORDER = ("baseline", "shared", "sepffn", "instruction", "multi", "adaptive")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    gt_path = ROOT / "mr_experiments/test_ground_truth.jsonl"
    ground_truth, stats = normalize_ground_truth(load_jsonl(gt_path), None, drop_empty_gt=False)
    if stats["kept"] != 1036:
        raise RuntimeError(f"Unexpected normalized test size: {stats}")

    thresholds = np.linspace(0.0, 1.0, 101)
    rows = []
    for method in ORDER:
        submission_path = ROOT / "mr_experiments" / method / "test/submission.jsonl"
        submission = load_jsonl(submission_path)
        if len(submission) != len(ground_truth):
            raise RuntimeError(f"{method}: {len(submission)} predictions for {len(ground_truth)} GT records")
        for threshold in thresholds:
            gated, gate_stats = prepare_submission_for_gmiou(
                submission,
                cls_threshold=float(threshold),
                max_pred_windows=10,
            )
            metrics = compute_G_mIoU(gated, ground_truth, k_list=(3,))
            rows.append(
                {
                    "experiment": method,
                    "threshold": f"{threshold:.2f}",
                    "G-mIoU@3": metrics["G-mIoU@3"],
                    "predicted_positive": gate_stats["num_predicted_positive"],
                    "predicted_negative": gate_stats["num_predicted_negative"],
                }
            )

    output = OUT / "gmiou_threshold_sweep.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
