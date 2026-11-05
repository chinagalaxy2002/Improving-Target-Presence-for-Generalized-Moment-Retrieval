#!/usr/bin/env python3
"""Execute a single step (geq_map training or geq_map_test evaluation) for a given backbone.
Selection metric is strictly Val mAP.
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


def train_geq_map(backbone, device="cuda:0", root=None):
    if root is None:
        root = ROOT / "runs" / "dq_cgp_v3"
    root = Path(root).resolve()
    
    config_path = ROOT / "configs" / "experiments" / f"{backbone}.json"
    gmr_best = root / backbone / "gmr" / "best.pt"
    if not gmr_best.is_file():
        raise FileNotFoundError(f"GMR baseline checkpoint not found: {gmr_best}")

    patch_for_map_selection()

    train_out = root / backbone / "geq_map"
    train_out.mkdir(parents=True, exist_ok=True)

    print(f"[{backbone}] Starting GEQ fine-tuning with Val mAP model selection on {device}...")
    cfg = load_config(config_path)
    res = sgt.train(
        config=cfg,
        output=train_out,
        stage="geq_full",
        device=device,
        baseline=gmr_best,
        resume=(train_out / "last.pt").is_file(),
    )
    print(f"[{backbone}] GEQ (mAP selection) complete: best_epoch={res.get('best_epoch')}, best_mAP={res.get('best_score'):.2f}%")


def test_geq_map(backbone, device="cuda:0", root=None):
    if root is None:
        root = ROOT / "runs" / "dq_cgp_v3"
    root = Path(root).resolve()
    
    config_path = ROOT / "configs" / "experiments" / f"{backbone}.json"
    train_out = root / backbone / "geq_map"
    best_pt = train_out / "best.pt"
    if not best_pt.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {best_pt}")

    test_out = root / backbone / "geq_map_test"
    test_out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)

    print(f"[{backbone}] Running Standard and Full test evaluation on {device}...")
    sgtest.test(
        config=cfg,
        checkpoint=best_pt,
        output=test_out,
        device=device,
    )
    print(f"[{backbone}] Test evaluation complete: output in {test_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--stage", required=True, choices=["geq_map", "geq_map_test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root", default=str(ROOT / "runs" / "dq_cgp_v3"))
    args = parser.parse_args()

    if args.stage == "geq_map":
        train_geq_map(args.backbone, device=args.device, root=args.root)
    elif args.stage == "geq_map_test":
        test_geq_map(args.backbone, device=args.device, root=args.root)


if __name__ == "__main__":
    main()
