"""Evaluate one current best checkpoint on the complete SoccerGMR split."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT / "generalized-moment-retrieval"
sys.path[:0] = [str(ROOT / "gmr_multiroute_kit"), str(REPO / "training/moment_detr_gmr"), str(REPO)]

from dataset import StartEndDataset, start_end_collate  # noqa: E402
from evaluate import compute_mr_results  # noqa: E402
from train import build_dataset_config  # noqa: E402
from models.moment_detr_gmr.utils.basic_utils import save_jsonl  # noqa: E402
from run_train import setup  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_path = ROOT / "mr_experiments" / args.variant / "best.ckpt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    opt, model, _ = setup(args.variant, args.device)
    labels = ROOT / "soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard"
    gt_path = labels / f"{args.split}.jsonl"
    opt.eval_path = str(gt_path)
    opt.eval_bsz = 4
    opt.num_workers = 2

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    dataset = StartEndDataset(
        **build_dataset_config(
            opt,
            str(gt_path),
            load_labels=True,
            keep_empty_gt=True,
        )
    )
    expected = sum(1 for line in gt_path.read_text(encoding="utf-8").splitlines() if line.strip())
    if len(dataset) != expected:
        raise RuntimeError(f"Dataset lost samples: loaded={len(dataset)}, expected={expected}")

    loader = DataLoader(
        dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
    )
    with torch.no_grad():
        submission, _ = compute_mr_results(None, model, loader, opt, criterion=None)
    if len(submission) != expected:
        raise RuntimeError(f"Prediction count mismatch: predictions={len(submission)}, expected={expected}")

    output_dir = ROOT / "mr_experiments" / args.variant / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.jsonl"
    save_jsonl(submission, str(submission_path))

    evaluator = REPO / "eval/eval_main.py"
    metric_paths = {}
    for threshold in (0.4, 0.5, 0.6):
        metric_path = output_dir / f"metrics_gmiou_{threshold:.1f}.json"
        command = [
            sys.executable,
            str(evaluator),
            "--submission_path", str(submission_path),
            "--gt_path", str(gt_path),
            "--save_path", str(metric_path),
            "--k_list", "1", "3", "5",
            "--cls_thresholds", "0.4", "0.5", "0.6",
            "--gmiou_cls_threshold", str(threshold),
            "--map_num_workers", "1",
            "--not_verbose",
        ]
        subprocess.run(command, cwd=REPO / "eval", check=True)
        metric_paths[str(threshold)] = str(metric_path)

    manifest = {
        "variant": args.variant,
        "split": args.split,
        "best_epoch": int(checkpoint["epoch"]) + 1,
        "checkpoint": str(checkpoint_path),
        "samples": expected,
        "submission": str(submission_path),
        "metrics": metric_paths,
    }
    (output_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
