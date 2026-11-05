#!/usr/bin/env python3
"""Run GEQ-Full fine-tuning and evaluation using Val mAP as the model selection criterion.
Outputs to:
  runs/dq_cgp_v3/<backbone>/geq_map/
  runs/dq_cgp_v3/<backbone>/geq_map_test/
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import soccergmr.train as sgt
import soccergmr.selection as sgs
import soccergmr.test as sgtest
from soccergmr.config import load_config


def patch_for_map_selection():
    sgt.score = lambda stage, metrics: float(metrics["brief"]["mAP"])
    original_selector_cls = sgs.Selector
    class MapSelector(original_selector_cls):
        def __init__(self, metric, patience, **kwargs):
            super().__init__(metric="Val mAP", patience=patience, **kwargs)
    sgt.Selector = MapSelector


def run_backbone(backbone, device="cuda:0", root=None):
    if root is None:
        root = ROOT / "runs" / "dq_cgp_v3"
    root = Path(root).resolve()
    
    config_path = ROOT / "configs" / "experiments" / f"{backbone}.json"
    gmr_best = root / backbone / "gmr" / "best.pt"
    if not gmr_best.is_file():
        raise FileNotFoundError(f"GMR baseline not found: {gmr_best}")

    patch_for_map_selection()

    train_out = root / backbone / "geq_map"
    train_out.mkdir(parents=True, exist_ok=True)

    print(f"[{backbone}] Starting GEQ-Full fine-tuning (Selection: Highest Val mAP)...")
    res = sgt.train_experiment(
        config_path=config_path,
        output_dir=train_out,
        stage="geq_full",
        device=device,
        baseline=gmr_best,
        resume=False,
    )
    print(f"[{backbone}] Training complete: best_epoch={res.get('best_epoch')}, best_mAP={res.get('best_score')}")

    test_out = root / backbone / "geq_map_test"
    test_out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)

    print(f"[{backbone}] Testing best_mAP checkpoint on Standard Test & Full Test...")
    sgtest.test(
        config=cfg,
        checkpoint=train_out / "best.pt",
        output=test_out,
        device=device,
    )
    print(f"[{backbone}] Evaluation complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root", default=str(ROOT / "runs" / "dq_cgp_v3"))
    args = parser.parse_args()

    run_backbone(args.backbone, device=args.device, root=args.root)


if __name__ == "__main__":
    main()
