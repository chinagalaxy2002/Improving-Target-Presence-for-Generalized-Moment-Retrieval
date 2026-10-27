#!/usr/bin/env python3
"""Build five D-GMR Standard-Test presence-threshold figures.

The formal Test checkpoint is read from each backbone's TEST_COMPLETE.json, so
this diagnostic never selects a checkpoint or threshold on Test.  Default mode
replots the saved curve CSV; --recompute rebuilds all 101 operating points from
the retained raw predictions and checks both G-mIoU curves with the official
evaluator.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
CURVE_CSV = REPORTS / "test_threshold_sweep_101.csv"
AUDIT_JSON = REPORTS / "TEST_THRESHOLD_SWEEP_AUDIT.json"
GT_PATH = ROOT / "data/test.jsonl"

BACKBONES = ("moment_detr", "qd_detr", "cg_detr", "eatr", "flashvtg")
BACKBONE_LABELS = {
    "moment_detr": "Moment-DETR",
    "qd_detr": "QD-DETR",
    "cg_detr": "CG-DETR",
    "eatr": "EaTR",
    "flashvtg": "FlashVTG",
}
METHODS = (
    ("original_gmr", "Original GMR", "#6B7280", "--", "o"),
    ("joint_geq", "Joint GMR + GEQ", "#0072B2", ":", "s"),
    ("d_gmr", "D-GMR", "#D55E00", "-", "^"),
)
THRESHOLDS = np.linspace(0.0, 1.0, 101)

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# Import the unchanged evaluator before backbone packages can shadow its names.
sys.path.insert(0, str(ROOT / "evaluation"))
from official_eval.metrics import (  # noqa: E402
    _clean_pred_windows,
    _compute_set_iou_score,
    compute_G_mIoU,
    prepare_submission_for_gmiou,
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def formal_test_directory(backbone: str) -> tuple[Path, dict]:
    completed = json.loads((RUNS / backbone / "TEST_COMPLETE.json").read_text(encoding="utf-8"))
    best = json.loads((RUNS / backbone / "best_epoch.json").read_text(encoding="utf-8"))
    assert completed["selection_split"] == "val"
    assert completed["epoch"] == best["best_epoch"]
    result = RUNS / completed["output"]
    assert result.is_dir()
    return result, completed


def sources(backbone: str) -> tuple[Path, dict[str, tuple[Path, Path]]]:
    result, completed = formal_test_directory(backbone)
    evaluation = json.loads((result / "evaluation.json").read_text(encoding="utf-8"))
    values = {
        "original_gmr": (
            result / evaluation["original_gmr_predictions"],
            result / "original_gmr_metrics.json",
        ),
        "joint_geq": (
            result / "joint_geq_predictions.jsonl",
            result / "joint_geq_metrics.json",
        ),
        "d_gmr": (result / "predictions.jsonl", result / "metrics.json"),
    }
    for prediction, metric in values.values():
        assert prediction.is_file() and metric.is_file()
    return result, values


def compute_curves() -> tuple[list[dict], dict]:
    ground_truth = read_jsonl(GT_PATH)
    assert len(ground_truth) == 1036
    qids = [row["qid"] for row in ground_truth]
    assert len(set(qids)) == len(qids)
    positive = np.asarray([bool(row.get("relevant_windows", [])) for row in ground_truth])
    assert int(positive.sum()) == 544 and int((~positive).sum()) == 492

    output: list[dict] = []
    source_records: dict[str, dict] = {}
    official_point_checks = 0
    for backbone in BACKBONES:
        result, method_sources = sources(backbone)
        for method, label, *_ in METHODS:
            prediction_path, metric_path = method_sources[method]
            prediction_rows = read_jsonl(prediction_path)
            by_qid = {row["qid"]: row for row in prediction_rows}
            assert len(by_qid) == len(prediction_rows) == len(ground_truth)
            assert set(by_qid) == set(qids)
            predictions = [by_qid[qid] for qid in qids]
            scores = np.asarray([float(row["pred_exist_score"]) for row in predictions])
            assert np.isfinite(scores).all()
            assert ((scores >= 0) & (scores <= 1)).all()
            fixed_metrics = json.loads(metric_path.read_text(encoding="utf-8"))["brief"]

            raw_iou = {
                k: np.asarray(
                    [
                        _compute_set_iou_score(
                            [
                                window[:2]
                                for window in _clean_pred_windows(
                                    prediction.get("pred_relevant_windows", []), 10
                                )[:k]
                            ],
                            target.get("relevant_windows", []),
                        )
                        for prediction, target in zip(predictions, ground_truth)
                    ]
                )
                for k in (1, 3)
            }

            for threshold in THRESHOLDS:
                accept = scores > threshold
                tn = int((~accept & ~positive).sum())
                fp = int((accept & ~positive).sum())
                fn = int((~accept & positive).sum())
                row = {
                    "backbone": backbone,
                    "method": method,
                    "method_label": label,
                    "threshold": float(threshold),
                    "G-mIoU@1": float(100 * np.where(accept, raw_iou[1], ~positive).mean()),
                    "G-mIoU@3": float(100 * np.where(accept, raw_iou[3], ~positive).mean()),
                    "PosRecall": float(100 * accept[positive].mean()),
                    "TNR": float(100 * (~accept[~positive]).mean()),
                    "Rej-F1": float(200 * tn / max(1, 2 * tn + fp + fn)),
                    "mAP": float(fixed_metrics["mAP"]),
                }

                gated, _ = prepare_submission_for_gmiou(predictions, float(threshold), 10)
                official = compute_G_mIoU(gated, ground_truth, k_list=(1, 3))
                for k in (1, 3):
                    assert abs(row[f"G-mIoU@{k}"] - official[f"G-mIoU@{k}"]) <= 0.011
                if np.isclose(threshold, 0.4):
                    assert abs(row["G-mIoU@1"] - fixed_metrics["G-mIoU@1"]) <= 0.011
                    assert abs(row["G-mIoU@3"] - fixed_metrics["G-mIoU@3"]) <= 0.011
                official_point_checks += 1
                output.append(row)

            source_records[f"{backbone}|{method}"] = {
                "prediction_path": str(prediction_path.relative_to(ROOT)),
                "metric_path": str(metric_path.relative_to(ROOT)),
                "queries": len(predictions),
                "selected_epoch": int(
                    json.loads((RUNS / backbone / "best_epoch.json").read_text())["best_epoch"]
                ),
                "formal_test_directory": str(result.relative_to(ROOT)),
            }

    REPORTS.mkdir(parents=True, exist_ok=True)
    with CURVE_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    audit = {
        "passed": True,
        "split": "Standard Test",
        "test_queries": len(ground_truth),
        "positive_queries": int(positive.sum()),
        "negative_queries": int((~positive).sum()),
        "backbones": list(BACKBONES),
        "methods": [method for method, *_ in METHODS],
        "thresholds_per_backbone_method": len(THRESHOLDS),
        "method_threshold_points": official_point_checks,
        "official_gmiou_values_checked": official_point_checks * 2,
        "presence_decision": "pred_exist_score > threshold",
        "fixed_protocol_threshold": 0.4,
        "test_based_checkpoint_or_threshold_selection": False,
        "checkpoint_selection": "Val mAP, then Val mR@1, then earlier epoch",
        "ground_truth_path": str(GT_PATH.relative_to(ROOT)),
        "curve_csv": str(CURVE_CSV.relative_to(ROOT)),
        "sources": source_records,
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output, audit


def load_curves() -> list[dict]:
    with CURVE_CSV.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("threshold", "G-mIoU@1", "G-mIoU@3", "PosRecall", "TNR", "Rej-F1", "mAP"):
            row[key] = float(row[key])
    assert len(rows) == len(BACKBONES) * len(METHODS) * len(THRESHOLDS)
    return rows


def draw_one(backbone: str, rows: list[dict]) -> None:
    data = {
        method: sorted(
            [row for row in rows if row["backbone"] == backbone and row["method"] == method],
            key=lambda row: row["threshold"],
        )
        for method, *_ in METHODS
    }
    assert all(len(points) == 101 for points in data.values())
    baseline = data["original_gmr"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.5))
    handles = []
    for index, (method, label, color, style, marker) in enumerate(METHODS):
        points = data[method]
        threshold = [row["threshold"] for row in points]
        style_args = dict(
            color=color,
            linestyle=style,
            marker=marker,
            markevery=10,
            markersize=3.8,
            linewidth=1.8,
        )
        handles.append(
            axes[0, 0].plot(
                threshold,
                [row["G-mIoU@1"] for row in points],
                label=f"{label} (mAP {points[0]['mAP']:.2f})",
                **style_args,
            )[0]
        )
        axes[0, 1].plot(threshold, [row["G-mIoU@3"] for row in points], **style_args)
        axes[0, 2].plot(
            threshold,
            [row["G-mIoU@1"] - base["G-mIoU@1"] for row, base in zip(points, baseline)],
            **style_args,
        )
        recall_order = sorted(points, key=lambda row: row["PosRecall"])
        axes[1, 0].plot(
            [row["PosRecall"] for row in recall_order],
            [row["TNR"] for row in recall_order],
            **style_args,
        )
        fixed = points[40]
        axes[1, 0].scatter(
            fixed["PosRecall"],
            fixed["TNR"],
            s=60,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=5,
        )
        axes[1, 1].plot(threshold, [row["Rej-F1"] for row in points], **style_args)
        x = np.arange(3) + (index - 1) * 0.24
        values = [points[position]["G-mIoU@1"] for position in (40, 50, 60)]
        bars = axes[1, 2].bar(x, values, width=0.23, color=color, edgecolor="white")
        axes[1, 2].bar_label(bars, fmt="%.1f", fontsize=8, padding=2, rotation=90)

    labels = (
        ("(a) Threshold sensitivity", "Presence threshold", "G-mIoU@1 (%)"),
        ("(b) Top-3 retrieval", "Presence threshold", "G-mIoU@3 (%)"),
        ("(c) Relative to original GMR", "Presence threshold", "Δ G-mIoU@1 (pp)"),
        ("(d) Retention–rejection trade-off", "Positive presence recall (%)", "N-acc / TNR (%)"),
        ("(e) Negative-class F1", "Presence threshold", "Rejection F1 (%)"),
        ("(f) Fixed operating points", "Presence threshold", "G-mIoU@1 (%)"),
    )
    for axis, (title, xlabel, ylabel) in zip(axes.flat, labels):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.grid(color="#CBD5E1", alpha=0.45, linewidth=0.6)
    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1]):
        axis.set_xlim(0, 1)
        axis.axvline(0.4, color="#334155", linestyle=(0, (1, 2)), linewidth=1.0)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 2]):
        axis.set_ylim(0, 70)
    delta = max(
        5.0,
        max(
            abs(row["G-mIoU@1"] - base["G-mIoU@1"])
            for method, *_ in METHODS
            for row, base in zip(data[method], baseline)
        ),
    )
    delta = float(np.ceil(delta / 5) * 5)
    axes[0, 2].set_ylim(-delta, delta)
    axes[0, 2].axhline(0, color="#475569", linewidth=0.8)
    axes[1, 0].set(xlim=(0, 100), ylim=(0, 100))
    axes[1, 1].set_ylim(0, 100)
    axes[1, 2].set_xticks(range(3), ("0.4", "0.5", "0.6"))
    axes[1, 0].text(
        2,
        4,
        "Joint GEQ and D-GMR share the same presence scores",
        fontsize=8,
        color="#475569",
    )

    selected_epoch = json.loads((RUNS / backbone / "best_epoch.json").read_text())["best_epoch"]
    fig.suptitle(
        f"{BACKBONE_LABELS[backbone]} | D-GMR Standard Test (Val-selected epoch {selected_epoch})",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False)
    fig.text(
        0.5,
        0.012,
        "Standard Test: 1,036 queries (544 positive / 492 negative), 101 thresholds. "
        "Dotted vertical line / large marker: protocol threshold 0.4.\n"
        "Checkpoints were selected only by Val mAP. Joint GMR + GEQ and D-GMR use identical GEQ presence scores; "
        "their G-mIoU difference reflects proposal reranking and PAT-Select.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.095, 1, 0.9))
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / f"{backbone}_d_gmr_test_threshold_sweep"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all points from retained formal-Test predictions.",
    )
    args = parser.parse_args()
    if args.recompute or not CURVE_CSV.exists():
        rows, audit = compute_curves()
        print(
            f"Checked {audit['method_threshold_points']} method-threshold points and "
            f"{audit['official_gmiou_values_checked']} official G-mIoU values."
        )
    else:
        rows = load_curves()
        print(f"Loaded {len(rows)} saved curve rows from {CURVE_CSV}.")
    for backbone in BACKBONES:
        draw_one(backbone, rows)
    print(f"Wrote five PNG/PDF/SVG figure sets to {FIGURES}.")


if __name__ == "__main__":
    main()
