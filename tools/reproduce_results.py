#!/usr/bin/env python3
"""Regenerate the three tables and optional backbone plots from saved TEST outputs.

No video features, checkpoints, Torch, or GPU are needed for this entry point.
Historical model-training scripts are separate source snapshots.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Phase2(addbackbone_with_S4_minusFusion)"
FIGURES = ROOT / "backbonefolder/figures"
sys.path.insert(0, str(ARCHIVE / "shared"))
sys.path.insert(0, str(FIGURES))
from build_table2_pdf import main as build_table

MODELS = ("moment_detr", "qd_detr", "cg_detr", "eatr", "flashvtg")
NAMES = dict(zip(MODELS, ("Moment-DETR", "QD-DETR", "CG-DETR", "EaTR", "FlashVTG")))


def load_series():
    series = {}
    for model in MODELS:
        variants = {}
        for variant in (("gmr",) if model == "flashvtg" else ("original", "gmr")):
            path = ARCHIVE / "shared/plot_baselines" / f"{model}_{variant}.json"
            variants[variant] = (path, json.loads(path.read_text()))
        path = FIGURES / "complete_comparison/data" / f"{model}_s4_epoch4.json"
        variants["s4"] = (path, json.loads(path.read_text()))
        series[model] = variants
    return series


def prediction_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recompute_metrics(series):
    """Official CPU scoring of the same frozen predictions at three thresholds."""
    from official_eval.eval_main import evaluate_gmr
    from official_eval.metrics import prepare_submission_for_gmiou, compute_G_mIoU

    gt_path = ARCHIVE / "shared/labels/test.jsonl"
    gt = prediction_rows(gt_path)
    output = ROOT / "reproduced/metrics"
    rows = []
    fields = ["Model", "AUROC", "Rej-F1@0.4", "mAP", "mR@1", "mR@5", "mR+@5", "G-mIoU@1", "G-mIoU@3"]
    for model, variants in series.items():
        for variant, (cache_path, cache) in variants.items():
            path = ROOT / cache["source_predictions"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != cache["source_sha256"]:
                raise ValueError(f"Saved predictions differ from their recorded hash: {path}")
            predictions = prediction_rows(path)
            by_id = {str(row["qid"]): row for row in predictions}
            if len(predictions) != len(by_id) or set(by_id) != {str(row["qid"]) for row in gt}:
                raise ValueError(f"Prediction coverage mismatch: {model}/{variant}")
            predictions = [{**by_id[str(row["qid"])], "qid": row["qid"]} for row in gt]
            official = evaluate_gmr(predictions, gt, cls_thresholds=(0.4, 0.5, 0.6),
                                    gmiou_cls_threshold=0.4, map_num_workers=0, verbose=False)
            name = (NAMES[model] if variant == "original" else NAMES[model] + "-GMR")
            if variant == "s4":
                name += " + S4_minus_Fusion (epoch 4)"
            elif model == "flashvtg":
                name += " (E0)"
            folder = output / model / variant
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "official_full_tau_0.4.json").write_text(json.dumps(official, indent=2) + "\n")
            base = {key: official["brief"][key] for key in fields[1:]}
            cache["tau_0.4"] = base
            for threshold in (0.4, 0.5, 0.6):
                gated, _ = prepare_submission_for_gmiou(predictions, cls_threshold=threshold, max_pred_windows=10)
                gmiou = compute_G_mIoU(gated, gt, k_list=(1, 3))
                cls = official["GMR-CLS"]["per_threshold"][str(threshold)]
                point = next(point for point in cache["grid"] if point["threshold"] == threshold)
                point.update({key: gmiou[key] for key in ("G-mIoU@1", "G-mIoU@3")})
                point["Rej-F1"] = cls["Rej-F1"]
                report = {"model": name, "threshold": threshold, "queries": len(gt),
                          "predictions_sha256": cache["source_sha256"],
                          "brief": {"AUROC": base["AUROC"], "Rej-F1": cls["Rej-F1"],
                                    **{key: base[key] for key in ("mAP", "mR@1", "mR@5", "mR+@5")},
                                    **gmiou}}
                (folder / f"tau_{threshold:g}.json").write_text(json.dumps(report, indent=2) + "\n")
            cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n")
            rows.append({"Model": name, **base})
            print(f"Scored {model}/{variant}: {len(gt)} queries, thresholds 0.4/0.5/0.6", flush=True)
    with (FIGURES / "complete_comparison/TABLE2_OUR_RESULTS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=float, nargs="+", choices=(0.4, 0.5, 0.6), default=(0.4, 0.5, 0.6))
    parser.add_argument("--recompute-metrics", action="store_true", help="Run official CPU scoring on all 14 saved TEST predictions before typesetting.")
    parser.add_argument("--figures", action="store_true", help="Also redraw the five six-panel backbone figures from archived curves.")
    args = parser.parse_args()
    series = load_series()
    if args.recompute_metrics:
        recompute_metrics(series)
    for threshold in args.thresholds:
        build_table(threshold)
    if args.figures:
        from build_complete_comparison import configure_style, render
        configure_style()
        for model, variants in series.items():
            render(model, {key: value for key, (_, value) in variants.items()})
            print(f"Redrew {NAMES[model]} six-panel comparison", flush=True)


if __name__ == "__main__":
    main()
