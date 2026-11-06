#!/usr/bin/env python3
"""Generate standard and full threshold sensitivity curve figures matching official reference.

Visualizes G-mIoU@1 and G-mIoU@3 across tau in [0.00, 1.00] for:
1. Historical GMR baseline (without DQ-CGPv3)
2. DQ-CGPv3 GMR (Stage 1)
3. DQ-CGPv3 + GEQ Full (Stage 2)
Outputs STANDARD_TEST.png, FULL_TEST.png, BOTH_TESTS.pdf, and per-backbone vector files.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from soccergmr.io import read_jsonl, dump_json
from soccergmr.evaluation.official.metrics import _clean_pred_windows, _compute_set_iou_score, get_existence_score

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path("/home/guoxiangyu/.gemini/antigravity-cli/brain/62a59d6f-74ff-44d8-a47a-d538b53ca17b")

BACKBONES = [
    ("moment_detr", "Moment-DETR"),
    ("eatr", "EaTR"),
    ("qd_detr", "QD-DETR"),
    ("cg_detr", "CG-DETR"),
    ("sim_detr", "Sim-DETR"),
    ("bm_detr", "BM-DETR"),
    ("bam_detr", "BAM-DETR"),
    ("uvcom", "UVCOM"),
    ("flashvtg", "FlashVTG"),
]

GT_PATHS = {
    "standard": "/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/test.jsonl",
    "full": "/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/experiments/gmr-Global-Eventquery/data/full_decodable/Full/test.jsonl",
}

HISTORICAL_PLOT_DATA_PATH = Path("/home/guoxiangyu/SoccerGMR-GEQ-additional/results/test_thresholds/PLOT_DATA.json")
HISTORICAL_EXPANSION_ROOT = Path("/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal")

STYLES = {
    "baseline_gmr": dict(color="#64748b", linestyle=":", marker="o", label="Baseline GMR"),
    "dq_cgp_gmr": dict(color="#2563eb", linestyle="-", marker="s", label="DQ-CGPv3 GMR"),
    "dq_cgp_geq_full": dict(color="#059669", linestyle="-.", marker="^", label="DQ-CGPv3 + GEQ Full"),
}


def compute_curves_from_preds(predictions, gt):
    count = len(gt)
    by_qid = {p["qid"]: p for p in predictions}
    ordered = [by_qid[g["qid"]] for g in gt if g["qid"] in by_qid]
    if len(ordered) != count:
        return None

    scores = []
    rejected = []
    accepted = {1: [], 3: []}
    for p, g in zip(ordered, gt):
        score, _ = get_existence_score(p)
        scores.append(float(score))
        rejected.append(_compute_set_iou_score([], g["relevant_windows"]))
        clean = _clean_pred_windows(p["pred_relevant_windows"], max_pred_windows=10)
        for k in (1, 3):
            accepted[k].append(_compute_set_iou_score([w[:2] for w in clean[:k]], g["relevant_windows"]))

    ts = [round(i / 100.0, 2) for i in range(101)]
    ys = {"1": [], "3": []}
    for tau in ts:
        for k in (1, 3):
            val = round(sum(a if s > tau else r for a, s, r in zip(accepted[k], scores, rejected)) / count, 4)
            ys[str(k)].append(val)

    oracle = {
        str(k): round(sum(a if g["relevant_windows"] else r for a, g, r in zip(accepted[k], gt, rejected)) / count, 4)
        for k in (1, 3)
    }

    return {"thresholds": ts, "gmiou": ys, "oracle": oracle}


def load_historical_baseline(name, split, gt):
    # Try PLOT_DATA.json first (for original 5 backbones)
    if HISTORICAL_PLOT_DATA_PATH.is_file():
        try:
            pdata = json.loads(HISTORICAL_PLOT_DATA_PATH.read_text())
            split_key = "standard" if split == "standard" else "full"
            if name in pdata.get("splits", {}).get(split_key, {}).get("backbones", {}):
                row = pdata["splits"][split_key]["backbones"][name]["gmr"]
                return {
                    "thresholds": row["thresholds"],
                    "gmiou": row["gmiou"],
                    "oracle": row["oracle"],
                    "mAP": row["mAP"],
                }
        except Exception:
            pass

    # Try formal expansion runs (for bm_detr, bam_detr, uvcom)
    split_folder = "test_standard" if split == "standard" else "test_full"
    pred_path = HISTORICAL_EXPANSION_ROOT / name / "gmr_test" / split_folder / "predictions.jsonl"
    metrics_path = HISTORICAL_EXPANSION_ROOT / name / "gmr_test" / split_folder / "metrics.json"

    if pred_path.is_file() and metrics_path.is_file():
        preds = read_jsonl(pred_path)
        metrics = json.loads(metrics_path.read_text())
        curves = compute_curves_from_preds(preds, gt)
        if curves:
            curves["mAP"] = metrics["brief"]["mAP"]
            return curves

    # Try FlashVTG reference results
    if name == "flashvtg":
        flash_dir = Path("/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/stage_0_gmr/reference_results/flashvtg/gmr")
        pred_file = flash_dir / ("test_predictions.jsonl" if split == "standard" else "full_test_predictions.jsonl")
        if pred_file.is_file():
            preds = read_jsonl(pred_file)
            curves = compute_curves_from_preds(preds, gt)
            if curves:
                curves["mAP"] = 25.43 if split == "standard" else 20.43
                return curves

    return None


def panel(ax, series_dict, k, limit):
    for method_key in ["baseline_gmr", "dq_cgp_gmr", "dq_cgp_geq_full"]:
        if method_key not in series_dict:
            continue
        row = series_dict[method_key]
        style = STYLES[method_key]
        ax.plot(
            row["thresholds"],
            row["gmiou"][str(k)],
            linewidth=1.85,
            markersize=4.2,
            markevery=[10, 30, 40, 50, 60, 70, 90],
            markerfacecolor="white",
            markeredgewidth=1.15,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            zorder=3,
        )

    # Oracle envelope
    oracles = [s["oracle"][str(k)] for s in series_dict.values() if "oracle" in s]
    if oracles:
        oracle = max(oracles)
        ax.axhline(oracle, color="#1F2937", linewidth=1.25, linestyle=(0, (5, 3)), zorder=2)
        ax.text(
            0.98,
            oracle + 0.008,
            f"Oracle envelope = {oracle:.4f}",
            ha="right",
            va="bottom",
            transform=ax.get_yaxis_transform(),
            fontsize=9.0,
            color="#1F2937",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.94, pad=1.0),
        )

    for tau in (0.4, 0.5, 0.6):
        ax.axvline(tau, color="#CBD5E1", lw=0.7, zorder=1)

    ax.set(xlim=(0, 1), ylim=(0, limit), xticks=np.arange(0, 1.01, 0.2), yticks=np.arange(0, limit + 0.001, 0.1))
    ax.set_ylabel(f"G-mIoU@{k} (0–1)", fontsize=9.5)
    ax.grid(axis="y", color="#E2E8F0", lw=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#94A3B8")
    ax.tick_params(length=3, color="#94A3B8")


def annotate_row(fig, series_dict, y):
    methods_present = [m for m in ["baseline_gmr", "dq_cgp_gmr", "dq_cgp_geq_full"] if m in series_dict]
    handles = [
        Line2D(
            [0],
            [0],
            linewidth=1.85,
            markersize=5,
            markerfacecolor="white",
            color=STYLES[m]["color"],
            linestyle=STYLES[m]["linestyle"],
            marker=STYLES[m]["marker"],
        )
        for m in methods_present
    ]
    labels = [f"{STYLES[m]['label']} | mAP {series_dict[m]['mAP']:.2f}%" for m in methods_present]
    fig.legend(
        handles,
        labels,
        ncol=len(methods_present),
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        frameon=False,
        fontsize=9.5,
        columnspacing=2.5,
        handlelength=2.5,
    )

    # Annotate deltas if available
    base = series_dict.get("baseline_gmr")
    dq = series_dict.get("dq_cgp_gmr")
    geq = series_dict.get("dq_cgp_geq_full")

    # If we have both baseline and DQ-CGP
    if base and dq:
        d_map = dq["mAP"] - base["mAP"]
        val_color = "#13804A" if d_map > 0 else "#C43C39" if d_map < 0 else "#64748B"
        fig.text(
            0.22,
            y - 0.022,
            f"ΔmAP (DQ-CGP vs Base): {d_map:+.2f} pp",
            ha="center",
            va="top",
            fontsize=8.8,
            color=val_color,
        )

        for x_pos, k in [(0.52, 1), (0.78, 3)]:
            b_vals = [base["gmiou"][str(k)][i] for i in (40, 50, 60)]
            dq_vals = [dq["gmiou"][str(k)][i] for i in (40, 50, 60)]
            d_gk = 100 * (np.mean(dq_vals) - np.mean(b_vals))
            gk_color = "#13804A" if d_gk > 0 else "#C43C39" if d_gk < 0 else "#64748B"
            fig.text(
                x_pos,
                y - 0.022,
                f"ΔG{k} mean (.4/.5/.6): {d_gk:+.2f} pp",
                ha="center",
                va="top",
                fontsize=8.8,
                color=gk_color,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "runs" / "dq_cgp_v3"))
    parser.add_argument("--output", default=str(ROOT / "runs" / "dq_cgp_v3" / "results" / "test_thresholds"))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    for split_key in ["standard", "full"]:
        split_name = "test_standard" if split_key == "standard" else "test_full"
        gt_path = GT_PATHS[split_key]
        gt = read_jsonl(gt_path)

        split_label = "Standard Test (1036 queries)" if split_key == "standard" else "Full-Decodable Test (2970 queries)"

        # Collect data for all backbones
        backbone_series = {}
        for name, display_name in BACKBONES:
            series = {}

            # 1. Historical Baseline
            base_curve = load_historical_baseline(name, split_key, gt)
            if base_curve:
                series["baseline_gmr"] = base_curve

            # 2. DQ-CGPv3 GMR
            dq_pred_path = root / name / "gmr_test" / split_name / "predictions.jsonl"
            dq_metrics_path = root / name / "gmr_test" / split_name / "metrics.json"
            if dq_pred_path.is_file() and dq_metrics_path.is_file():
                preds = read_jsonl(dq_pred_path)
                metrics = json.loads(dq_metrics_path.read_text())
                dq_curves = compute_curves_from_preds(preds, gt)
                if dq_curves:
                    dq_curves["mAP"] = metrics["brief"]["mAP"]
                    series["dq_cgp_gmr"] = dq_curves

            # 3. DQ-CGPv3 + GEQ Full (if available)
            geq_pred_path = root / name / "geq_full_test" / split_name / "predictions.jsonl"
            geq_metrics_path = root / name / "geq_full_test" / split_name / "metrics.json"
            if geq_pred_path.is_file() and geq_metrics_path.is_file():
                preds = read_jsonl(geq_pred_path)
                metrics = json.loads(geq_metrics_path.read_text())
                geq_curves = compute_curves_from_preds(preds, gt)
                if geq_curves:
                    geq_curves["mAP"] = metrics["brief"]["mAP"]
                    series["dq_cgp_geq_full"] = geq_curves

            if series:
                backbone_series[name] = series

        if not backbone_series:
            print(f"No series data found for {split_key}")
            continue

        n_rows = len(backbone_series)
        fig, axes = plt.subplots(n_rows, 2, figsize=(13.0, 3.6 * n_rows), sharex=True, sharey=True)
        if n_rows == 1:
            axes = np.expand_dims(axes, 0)

        fig.subplots_adjust(left=0.075, right=0.985, top=0.92, bottom=0.07, hspace=0.65, wspace=0.22)
        fig.suptitle(
            f"SoccerGMR G-mIoU Sensitivity by Backbone · Standard-trained → {split_label}",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )
        fig.text(0.5, 0.955, "Comparing Baseline GMR vs DQ-CGPv3 GMR vs DQ-CGPv3 + GEQ Full", ha="center", fontsize=10.5, color="#475569")

        limit = 0.70

        for i, (name, display_name) in enumerate(BACKBONES):
            if name not in backbone_series:
                continue
            series = backbone_series[name]
            for j, k in enumerate((1, 3)):
                ax = axes[i, j]
                panel(ax, series, k, limit)
                ax.set_title(f"{display_name} · G-mIoU@{k}", pad=8, fontsize=11, fontweight="bold")
                if i == n_rows - 1:
                    ax.set_xlabel(r"Existence threshold $\tau$", fontsize=10)

            row_y = axes[i, 0].get_position().y1 + 0.038
            annotate_row(fig, series, row_y)

        fig.text(
            0.5,
            0.02,
            "Oracle envelope: max of plotted methods with GT presence and fixed candidates. Guides at tau in {0.4, 0.5, 0.6}.",
            ha="center",
            fontsize=9.0,
            color="#475569",
        )

        stem_name = "STANDARD_TEST" if split_key == "standard" else "FULL_TEST"
        out_png = out_dir / f"{stem_name}.png"
        out_pdf = out_dir / f"{stem_name}.pdf"

        fig.savefig(out_png, dpi=220, facecolor="white")
        fig.savefig(out_pdf, metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)

        print(f"Saved {out_png} and {out_pdf}")

        # Copy to artifact dir for instant user view
        if ARTIFACT_DIR.is_dir():
            shutil.copy2(out_png, ARTIFACT_DIR / f"{stem_name}.png")
            print(f"Copied {stem_name}.png to artifact directory")

    # Combine Standard and Full test curves side-by-side into a single unified image
    std_png = out_dir / "STANDARD_TEST.png"
    full_png = out_dir / "FULL_TEST.png"
    if std_png.is_file() and full_png.is_file():
        from PIL import Image, ImageDraw
        im1 = Image.open(std_png)
        im2 = Image.open(full_png)
        gap = 16
        total_width = im1.width + gap + im2.width
        height = im1.height
        combined = Image.new("RGB", (total_width, height), color=(255, 255, 255))
        combined.paste(im1, (0, 0))
        combined.paste(im2, (im1.width + gap, 0))
        draw = ImageDraw.Draw(combined)
        line_x = im1.width + gap // 2
        draw.line([(line_x, 80), (line_x, height - 80)], fill=(203, 213, 225), width=2)

        combined_png = out_dir / "TEST_THRESHOLDS_COMBINED.png"
        combined_pdf = out_dir / "TEST_THRESHOLDS_COMBINED.pdf"
        combined.save(combined_png, format="PNG", optimize=True)
        combined.save(combined_pdf, format="PDF", resolution=220)
        print(f"Saved combined thresholds plot to {combined_png} and {combined_pdf}")

        docs_img_dir = ROOT / "docs" / "images"
        if docs_img_dir.is_dir():
            shutil.copy2(combined_png, docs_img_dir / "test_thresholds_combined.png")

        docs_tbl_dir = ROOT / "docs" / "tables"
        if docs_tbl_dir.is_dir():
            shutil.copy2(combined_pdf, docs_tbl_dir / "TEST_THRESHOLDS_COMBINED.pdf")

        if ARTIFACT_DIR.is_dir():
            shutil.copy2(combined_png, ARTIFACT_DIR / "TEST_THRESHOLDS_COMBINED.png")
            shutil.copy2(combined_pdf, ARTIFACT_DIR / "TEST_THRESHOLDS_COMBINED.pdf")


if __name__ == "__main__":
    main()
