#!/usr/bin/env python3
"""Recompute or replot the 5-backbone, 3-method Standard Test threshold sweeps.

Default mode rebuilds figures from reports/threshold_sweep_101.csv. Pass
--recompute to load the frozen E0/E1/GAP predictions, compute all 101 operating
points per method with the official G-mIoU evaluator, and refresh the CSV/audit.
No training or threshold selection is performed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
SOURCE = {
    "E0": "ablations/global_hard_roi_20261023/results/{backbone}/gmr",
    "E1": "ablations/end_to_end_geq_pv_20261026/runs/{backbone}/E1",
    "GAP": "ablations/geq_gap_all_backbones_20261026/runs/{backbone}",
}
BACKBONES = ["moment_detr", "qd_detr", "cg_detr", "eatr", "flashvtg"]
BACKBONE_LABELS = {
    "moment_detr": "Moment-DETR", "qd_detr": "QD-DETR", "cg_detr": "CG-DETR",
    "eatr": "EaTR", "flashvtg": "FlashVTG",
}
METHODS = [
    ("E0", "GMR", "#64748b", "--", "o"),
    ("E1", "GMR + GEQ", "#0072B2", "-", "s"),
    ("GAP", "GMR + GEQ + GAP", "#D55E00", "-.", "^"),
]
THRESHOLDS = np.linspace(0.0, 1.0, 101)
GT_PATH = ROOT / "ablations/end_to_end_geq_pv_20261026/data/test.jsonl"
REPORTS = EXP / "reports"
FIGURES = EXP / "figures"
CURVE_CSV = REPORTS / "threshold_sweep_101.csv"
AUDIT_JSON = REPORTS / "THRESHOLD_SWEEP_AUDIT.json"

sys.path.insert(0, str(ROOT / "ablations/geq_gap_publication_20261026"))
from official_eval.metrics import (  # noqa: E402
    _clean_pred_windows,
    _compute_set_iou_score,
    compute_G_mIoU,
    prepare_submission_for_gmiou,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_path(method: str, backbone: str, name: str) -> Path:
    if method == "GAP" and backbone == "flashvtg":
        folder = ROOT / "ablations/geq_gap_search_20261026/runs/GAP_m60_w200"
    else:
        folder = ROOT / SOURCE[method].format(backbone=backbone)
    return folder / name


def compute_curves():
    gt = jsonl(GT_PATH)
    assert len(gt) == 1036
    qids = [x["qid"] for x in gt]
    assert len(set(qids)) == len(gt)
    positive = np.asarray([bool(x.get("relevant_windows", [])) for x in gt], dtype=bool)
    assert int(positive.sum()) == 544 and int((~positive).sum()) == 492

    output, sources, check_count = [], {}, 0
    for backbone in BACKBONES:
        for method, label, *_ in METHODS:
            pred_path = source_path(method, backbone, "test_predictions.jsonl")
            metric_path = source_path(method, backbone, "test_metrics.json")
            preds_raw = jsonl(pred_path)
            by_qid = {x["qid"]: x for x in preds_raw}
            assert len(by_qid) == len(preds_raw) == len(gt), (backbone, method, "prediction count")
            assert set(by_qid) == set(qids), (backbone, method, "qid mismatch")
            preds = [by_qid[qid] for qid in qids]
            scores = np.asarray([float(x["pred_exist_score"]) for x in preds], dtype=float)
            assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()

            raw_iou = {}
            for k in (1, 3):
                raw_iou[k] = np.asarray([
                    _compute_set_iou_score(
                        [w[:2] for w in _clean_pred_windows(p.get("pred_relevant_windows", []), 10)[:k]],
                        g.get("relevant_windows", []),
                    )
                    for p, g in zip(preds, gt)
                ], dtype=float)

            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            summary = metric["0.4"]["brief"]
            for threshold in THRESHOLDS:
                accept = scores > threshold  # Same strict decision used by the official evaluator.
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
                    "Rej_F1": float(200 * tn / max(1, 2 * tn + fp + fn)),
                    "mAP": float(summary["mAP"]),
                }

                # Cross-check every point and both k values against the official evaluator.
                gated, _ = prepare_submission_for_gmiou(preds, float(threshold), 10)
                official = compute_G_mIoU(gated, gt, k_list=(1, 3))
                for k in (1, 3):
                    assert abs(row[f"G-mIoU@{k}"] - official[f"G-mIoU@{k}"]) <= 0.011, (
                        backbone, method, float(threshold), k,
                        row[f"G-mIoU@{k}"], official[f"G-mIoU@{k}"],
                    )
                check_count += 1
                if np.isclose(threshold, [0.4, 0.5, 0.6]).any():
                    key = f"{threshold:.1f}"
                    fixed = metric[key]["brief"]
                    for name in ("G-mIoU@1", "G-mIoU@3"):
                        assert abs(row[name] - fixed[name]) <= 0.011, (backbone, method, key, name)
                output.append(row)

            sources[f"{backbone}|{method}"] = {
                "prediction_path": str(pred_path.relative_to(ROOT)),
                "prediction_sha256": sha256(pred_path),
                "metrics_path": str(metric_path.relative_to(ROOT)),
                "metrics_sha256": sha256(metric_path),
                "prediction_queries": len(preds),
                "score_field": "pred_exist_score",
                "positive_mean_score": float(scores[positive].mean()),
                "negative_mean_score": float(scores[~positive].mean()),
            }

    REPORTS.mkdir(parents=True, exist_ok=True)
    with CURVE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    audit = {
        "passed": True,
        "split": "Standard Test",
        "test_queries": len(gt), "positive_queries": int(positive.sum()),
        "negative_queries": int((~positive).sum()),
        "backbones": BACKBONES,
        "methods": [x[0] for x in METHODS],
        "thresholds": [float(x) for x in THRESHOLDS],
        "thresholds_per_backbone_method": 101,
        "official_threshold_point_checks": check_count,
        "official_gmiou_k_checks": check_count * 2,
        "qid_alignment": "exact; predictions reordered to GT qid order",
        "presence_decision": "pred_exist_score > threshold",
        "official_evaluator_rounding_tolerance_pp": 0.011,
        "fixed_threshold_checks": [0.4, 0.5, 0.6],
        "test_based_selection": False,
        "gap_configuration": "margin=0.6, lambda=2, epoch=4; fixed from Val protocol",
        "ground_truth_path": str(GT_PATH.relative_to(ROOT)),
        "ground_truth_sha256": sha256(GT_PATH),
        "curve_csv_sha256": sha256(CURVE_CSV),
        "sources": sources,
    }
    AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output, audit


def load_curves():
    with CURVE_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("threshold", "G-mIoU@1", "G-mIoU@3", "PosRecall", "TNR", "Rej_F1", "mAP"):
            row[key] = float(row[key])
    expected = len(BACKBONES) * len(METHODS) * len(THRESHOLDS)
    assert len(rows) == expected, (len(rows), expected)
    return rows


def draw_one(backbone: str, rows: list[dict]):
    data = {
        method: sorted(
            [r for r in rows if r["backbone"] == backbone and r["method"] == method],
            key=lambda r: r["threshold"],
        )
        for method, *_ in METHODS
    }
    assert all(len(v) == 101 for v in data.values())
    base = data["E0"]
    fig, axes = plt.subplots(2, 3, figsize=(14.3, 8.5))
    handles = []
    for i, (method, label, color, style, marker) in enumerate(METHODS):
        points = data[method]
        t = [r["threshold"] for r in points]
        kw = dict(color=color, linestyle=style, marker=marker, markevery=10, markersize=3.5, linewidth=1.7)
        handles.append(axes[0, 0].plot(t, [r["G-mIoU@1"] for r in points],
                                      label=f"{label} (mAP {points[0]['mAP']:.2f})", **kw)[0])
        axes[0, 1].plot(t, [r["G-mIoU@3"] for r in points], **kw)
        axes[0, 2].plot(t, [r["G-mIoU@1"] - b["G-mIoU@1"] for r, b in zip(points, base)], **kw)
        # Match the reference's positive-retention / negative-rejection panel.
        by_recall = sorted(points, key=lambda r: r["PosRecall"])
        axes[1, 0].plot([r["PosRecall"] for r in by_recall], [r["TNR"] for r in by_recall], **kw)
        p04 = points[40]
        axes[1, 0].scatter(p04["PosRecall"], p04["TNR"], s=65, marker=marker,
                           color=color, edgecolor="white", zorder=5)
        axes[1, 1].plot(t, [r["Rej_F1"] for r in points], **kw)
        xs = np.arange(3) + (i - 1) * 0.24
        vals = [points[j]["G-mIoU@1"] for j in (40, 50, 60)]
        bars = axes[1, 2].bar(xs, vals, width=0.23, color=color, edgecolor="white")
        axes[1, 2].bar_label(bars, fmt="%.2f", fontsize=8, padding=3, rotation=90)

    specs = [
        ("(a) Threshold sensitivity", "Presence threshold", "G-mIoU@1 (%)"),
        ("(b) Top-3 retrieval", "Presence threshold", "G-mIoU@3 (%)"),
        ("(c) Relative to GMR baseline", "Presence threshold", "Δ G-mIoU@1 (pp)"),
        ("(d) Rejection / positive retention", "Positive presence recall (%)", "N-acc / TNR (%)"),
        ("(e) Negative-class F1", "Presence threshold", "Rejection F1 (%)"),
        ("(f) Fixed operating points", "Presence threshold", "G-mIoU@1 (%)"),
    ]
    for ax, (title, xlabel, ylabel) in zip(axes.flat, specs):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.18)
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1]):
        ax.set_xlim(0, 1)
        ax.axvline(0.4, color="#64748b", linestyle=":", linewidth=0.9)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 2]):
        ax.set_ylim(0, 70)
    axes[0, 2].axhline(0, color="gray", linewidth=0.8)
    axes[1, 0].set(xlim=(0, 100), ylim=(0, 105))
    axes[1, 1].set_ylim(0, 100)
    axes[1, 2].set_xticks(range(3), ["0.4", "0.5", "0.6"])
    fig.suptitle(f"{BACKBONE_LABELS[backbone]} | Soccer-GMR Standard Test", fontsize=14, fontweight="bold", y=0.985)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3,
               frameon=False, fontsize=10)
    fig.text(0.5, 0.012,
             "Standard Test: 1,036 queries (544 positive / 492 negative); all 101 thresholds shown.\n"
             "Dotted line / large marker: fixed threshold 0.4. Localization mAP is computed without an existence gate.\n"
             "GAP uses the fixed Val configuration (margin=0.6, λ=2, epoch 4); no Test-based threshold or checkpoint selection.",
             ha="center", fontsize=8.6, color="#475569")
    fig.tight_layout(rect=(0, 0.105, 1, 0.91))
    FIGURES.mkdir(parents=True, exist_ok=True)
    stem = FIGURES / f"{backbone}_fig7_test_threshold_sweep"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=220, bbox_inches="tight")
    svg_path = stem.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute every point from local frozen predictions and official evaluator.")
    args = parser.parse_args()
    if args.recompute or not CURVE_CSV.exists():
        rows, audit = compute_curves()
        print(f"Recomputed {audit['official_threshold_point_checks']} method-threshold points; "
              f"{audit['official_gmiou_k_checks']} official G-mIoU checks passed.")
    else:
        rows = load_curves()
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8")) if AUDIT_JSON.exists() else {}
        print(f"Loaded {len(rows)} saved threshold points from {CURVE_CSV.relative_to(ROOT)}.")
    for backbone in BACKBONES:
        draw_one(backbone, rows)
    if audit:
        audit["figure_sha256"] = {
            p.name: sha256(p) for p in sorted(FIGURES.glob("*_fig7_test_threshold_sweep.*"))
        }
        audit["figures"] = [p.name for p in sorted(FIGURES.glob("*_fig7_test_threshold_sweep.*"))]
        AUDIT_JSON.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote five backbones × PNG/PDF/SVG threshold-sweep figures.")


if __name__ == "__main__":
    main()
