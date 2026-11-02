"""Recompute all published Test metrics and figures from frozen predictions."""
from __future__ import annotations

import pathlib
import os
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = pathlib.Path(os.environ.get('GMR_UPSTREAM_REPO', ROOT / 'generalized-moment-retrieval'))
EVALUATOR = REPO / "eval/eval_main.py"
GT = ROOT / "mr_experiments/test_ground_truth.jsonl"
VARIANTS = ("baseline", "shared", "sepffn", "instruction", "multi", "adaptive")


def run(command: list[str], *, cwd: pathlib.Path | None = None, env: dict | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    if not EVALUATOR.is_file():
        raise FileNotFoundError(
            f"Missing upstream evaluator: {EVALUATOR}. Follow README section 7 first."
        )
    for variant in VARIANTS:
        test_dir = ROOT / "mr_experiments" / variant / "test"
        submission = test_dir / "submission.jsonl"
        for threshold in (0.4, 0.5, 0.6):
            run(
                [
                    sys.executable,
                    str(EVALUATOR),
                    "--submission_path", str(submission),
                    "--gt_path", str(GT),
                    "--save_path", str(test_dir / f"metrics_gmiou_{threshold:.1f}.json"),
                    "--k_list", "1", "3", "5",
                    "--cls_thresholds", "0.4", "0.5", "0.6",
                    "--gmiou_cls_threshold", str(threshold),
                    "--map_num_workers", "1",
                    "--not_verbose",
                ],
                cwd=REPO / "eval",
            )

    run(
        [sys.executable, str(ROOT / "gmr_multiroute_kit/collect_results.py")],
        env={**os.environ, 'GMR_PROJECT_ROOT': str(ROOT)},
    )
    figure_dir = ROOT / "newidea/figure/gmr_mr_detr_test_20261030"
    run([sys.executable, str(figure_dir / "build_gmiou_sweep.py")])
    run([sys.executable, str(figure_dir / "render.py")])
    print("Recomputed six Test reports, the 0-1 sweep, and all figure formats.")


if __name__ == "__main__":
    main()
