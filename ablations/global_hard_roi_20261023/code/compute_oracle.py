"""Compute Perfect-Existence Oracle for E0 baseline predictions.

The oracle uses E0's frozen temporal windows but makes perfect binary keep/reject
decisions based on ground truth labels. This isolates the maximum G-mIoU gain
attributable to existence classification alone.
"""
import gzip
import json
import sys
from pathlib import Path

# Add eval utilities to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baseline_source'))

from eval.metrics import compute_G_mIoU


def read_gzip(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    ROOT = Path(__file__).resolve().parents[1]

    # Read E0 baseline predictions (temporal windows frozen)
    e0_predictions = read_gzip(ROOT / 'results/E0/test_predictions.jsonl.gz')

    # Read ground truth
    ground_truth = read_jsonl(ROOT / 'data/test.jsonl')

    # Build ground truth lookup
    is_positive = {
        row['qid']: bool(row.get('relevant_windows')) for row in ground_truth
    }

    # Oracle: use E0 windows but perfect existence decision
    oracle_submission = []
    for row in e0_predictions:
        oracle_submission.append({
            'qid': row['qid'],
            'pred_relevant_windows': (
                row.get('pred_relevant_windows', [])
                if is_positive[row['qid']]
                else []
            ),
        })

    # Compute oracle G-mIoU@1 (threshold-independent)
    oracle_gmiou = compute_G_mIoU(oracle_submission, ground_truth, (1,))['G-mIoU@1']

    print(f"Perfect-Existence Oracle G-mIoU@1: {oracle_gmiou:.2f}%")
    print(f"\nThis oracle value is threshold-independent because existence")
    print(f"classification is perfect. It represents the theoretical upper bound")
    print(f"for any existence classifier using E0's frozen temporal localizer.")

    return oracle_gmiou


if __name__ == '__main__':
    main()
