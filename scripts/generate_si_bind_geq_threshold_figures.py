#!/usr/bin/env python3
"""Generate publication-quality threshold sensitivity figures comparing:
1. GMR baseline
2. GMR + GEQ Full (historical)
3. GMR + SI-Bind (E6) + GEQ (Ours)

Matches the visual style and protocol of:
results/test_thresholds_11_backbones/STANDARD_TEST.png
Outputs:
- STANDARD_TEST.png, STANDARD_TEST.pdf, STANDARD_TEST.svg
- FULL_TEST.png, FULL_TEST.pdf, FULL_TEST.svg
- BOTH_TESTS.pdf
- STANDARD_BY_BACKBONE.pdf, FULL_BY_BACKBONE.pdf
- PLOT_DATA.json, METRICS.csv, CURVES.csv
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Ensure soccergmr is importable
WORKSPACE_ROOT = Path("/home/guoxiangyu/SoccerGMR_SI_Bind_IsolatedLab")
sys.path.insert(0, str(WORKSPACE_ROOT / "revisions" / "rev_0002_si_bind_geq" / "src"))

from soccergmr.evaluation.official.metrics import _clean_pred_windows, _compute_set_iou_score

BACKBONES = [
    ("moment_detr", "Moment-DETR"),
    ("eatr", "EaTR"),
    ("qd_detr", "QD-DETR"),
    ("cg_detr", "CG-DETR"),
    ("sim_detr", "Sim-DETR"),
    ("bm_detr", "BM-DETR"),
    ("bam_detr", "BAM-DETR"),
    ("uvcom", "UVCOM"),
]

GT_PATHS = {
    "standard": Path("/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/test.jsonl"),
    "full": Path("/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/experiments/gmr-Global-Eventquery/data/full_decodable/Full/test.jsonl"),
}

HISTORICAL_DATA_PATH = WORKSPACE_ROOT / "historical_11_backbones_plot_data.json"
CHECKPOINTS_ROOT = WORKSPACE_ROOT / "runs" / "rev_0001"

STYLES = {
    "gmr": dict(color="#535353", linestyle=":", marker="o", label="GMR baseline"),
    "geq_full": dict(color="#009E73", linestyle="-.", marker="^", label="GMR + GEQ Full"),
    "si_bind_e6_geq": dict(color="#7c3aed", linestyle="-", marker="s", label="GMR + SI-Bind (E6) + GEQ (Ours)"),
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_curves_from_predictions(preds, gt):
    count = len(gt)
    byid = {p["qid"]: p for p in preds}
    ordered = [byid[g["qid"]] for g in gt if g["qid"] in byid]
    if len(ordered) != count:
        return None

    scores = np.array([p["pred_exist_score"] for p in ordered])
    truth = np.array([bool(g["relevant_windows"]) for g in gt])
    rejected = (~truth).astype(float)
    accepted = {
        k: np.array([
            _compute_set_iou_score(
                [w[:2] for w in _clean_pred_windows(r["pred_relevant_windows"], max_pred_windows=10)[:k]],
                g["relevant_windows"]
            )
            for r, g in zip(ordered, gt)
        ])
        for k in [1, 3]
    }

    sweep = {
        str(k): [float(np.where(scores > i / 100.0, accepted[k], rejected).mean()) for i in range(101)]
        for k in [1, 3]
    }
    oracle = {
        str(k): float(np.where(truth, accepted[k], rejected).mean())
        for k in [1, 3]
    }

    return {
        "thresholds": [round(i / 100.0, 2) for i in range(101)],
        "gmiou": sweep,
        "oracle": oracle,
        "all_rejected": float(rejected.mean()),
    }


def panel(ax, series_dict, k, limit):
    for method in ["gmr", "geq_full", "si_bind_e6_geq"]:
        if method not in series_dict:
            continue
        row = series_dict[method]
        style = STYLES[method]
        ax.plot(
            row["thresholds"],
            row["gmiou"][str(k)],
            linewidth=1.85,
            markersize=4.6,
            markevery=[10, 30, 40, 50, 60, 70, 90],
            markerfacecolor="white",
            markeredgewidth=1.15,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            zorder=3,
        )

    # Oracle envelope: max oracle across plotted methods
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


def footer(ax, series_dict):
    ax.axis("off")
    methods_present = [m for m in ["gmr", "geq_full", "si_bind_e6_geq"] if m in series_dict]
    handles = [
        Line2D(
            [0],
            [0],
            lw=1.85,
            markersize=4.6,
            markerfacecolor="white",
            color=STYLES[m]["color"],
            linestyle=STYLES[m]["linestyle"],
            marker=STYLES[m]["marker"],
        )
        for m in methods_present
    ]
    labels = [f"{STYLES[m]['label']} | mAP {series_dict[m]['mAP']:.2f}%" for m in methods_present]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(methods_present),
        frameon=False,
        fontsize=9.5,
        columnspacing=2.5,
        handlelength=2.8,
        borderaxespad=0,
    )

    if "si_bind_e6_geq" in series_dict and "gmr" in series_dict:
        ours = series_dict["si_bind_e6_geq"]
        base = series_dict["gmr"]
        d_map_base = ours["mAP"] - base["mAP"]
        d_g1_base = 100 * np.mean([ours["gmiou"]["1"][i] - base["gmiou"]["1"][i] for i in [40, 50, 60]])
        d_g3_base = 100 * np.mean([ours["gmiou"]["3"][i] - base["gmiou"]["3"][i] for i in [40, 50, 60]])

        # Compare vs GEQ Full if present
        extra_str = ""
        if "geq_full" in series_dict:
            geq = series_dict["geq_full"]
            d_map_geq = ours["mAP"] - geq["mAP"]
            extra_str = f"  (vs GEQ: {d_map_geq:+.2f} pp)"

        metrics_items = [
            (0.18, "ΔmAP vs GMR", d_map_base, extra_str),
            (0.52, "ΔG1 mean vs GMR (.4/.5/.6)", d_g1_base, ""),
            (0.84, "ΔG3 mean vs GMR (.4/.5/.6)", d_g3_base, ""),
        ]
        for x, label, val, suffix in metrics_items:
            color = "#13804A" if val > 0 else "#C43C39" if val < 0 else "#64748B"
            ax.text(
                x,
                0.05,
                f"{label}: {val:+.2f} pp{suffix}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold" if "mAP" in label else "normal",
                color=color,
            )


def build_plot_data():
    with open(HISTORICAL_DATA_PATH, "r", encoding="utf-8") as f:
        hist_data = json.load(f)

    gt_data = {
        "standard": load_jsonl(GT_PATHS["standard"]),
        "full": load_jsonl(GT_PATHS["full"]),
    }

    plot_snapshot = {
        "schema": 3,
        "title": "SoccerGMR Threshold Sensitivity: GMR Baseline vs. GMR + GEQ Full vs. GMR + SI-Bind (E6) + GEQ (Ours)",
        "train_split": "Standard",
        "test_used_for_selection": False,
        "splits": {},
    }

    metrics_rows = []
    curves_rows = []

    for split_key, split_label in [("standard", "Standard Test (1,036 queries)"), ("full", "Full-Decodable Test (2,970 queries)")]:
        gt = gt_data[split_key]
        split_folder = "test_standard" if split_key == "standard" else "test_full"
        split_dict = {
            "label": f"SoccerGMR {split_label}",
            "rows": len(gt),
            "backbones": {},
        }

        for bb_key, bb_name in BACKBONES:
            bb_pair = {}

            # 1. Historical GMR baseline
            if bb_key in hist_data["splits"][split_key]["backbones"]:
                bb_pair["gmr"] = hist_data["splits"][split_key]["backbones"][bb_key]["gmr"]

            # 2. Historical GEQ Full
            if bb_key in hist_data["splits"][split_key]["backbones"]:
                bb_pair["geq_full"] = hist_data["splits"][split_key]["backbones"][bb_key]["geq_full"]

            # 3. SI-Bind (E6) + GEQ (Ours)
            pred_path = CHECKPOINTS_ROOT / bb_key / "E6" / "gmr_geq" / "test_eval" / split_folder / "predictions.jsonl"
            metrics_path = CHECKPOINTS_ROOT / bb_key / "E6" / "gmr_geq" / "test_eval" / split_folder / "metrics.json"

            if pred_path.is_file() and metrics_path.is_file():
                preds = load_jsonl(pred_path)
                with open(metrics_path, "r", encoding="utf-8") as mf:
                    met = json.load(mf)
                curve = compute_curves_from_predictions(preds, gt)
                if curve:
                    curve["mAP"] = met.get("brief", {}).get("mAP", met.get("mAP_detail", {}).get("mAP", 0.0))
                    curve["AUROC"] = met.get("GMR-CLS", {}).get("AUROC", 0.0)
                    curve["rej_f1_50"] = met.get("brief", {}).get("Rej-F1@0.5", 0.0)
                    curve["protocol"] = "SI-Bind (E6) + GEQ Full fine-tuning; min Val Orc-gap selection; patience 20; seed 2023"
                    bb_pair["si_bind_e6_geq"] = curve

                    # Export curve records
                    for t_idx, tau in enumerate(curve["thresholds"]):
                        curves_rows.append({
                            "split": split_key,
                            "backbone": bb_key,
                            "method": "si_bind_e6_geq",
                            "threshold": tau,
                            "G1": curve["gmiou"]["1"][t_idx],
                            "G3": curve["gmiou"]["3"][t_idx],
                        })

                    # Export metrics record
                    row_rec = {
                        "split": split_key,
                        "backbone": bb_key,
                        "method": "si_bind_e6_geq",
                        "mAP": curve["mAP"],
                        "AUROC": curve["AUROC"],
                        "Rej-F1@0.5": curve["rej_f1_50"],
                        "oracle_G1": curve["oracle"]["1"],
                        "oracle_G3": curve["oracle"]["3"],
                        "G1_mean_456": np.mean([curve["gmiou"]["1"][i] for i in [40, 50, 60]]),
                        "G3_mean_456": np.mean([curve["gmiou"]["3"][i] for i in [40, 50, 60]]),
                    }
                    if "gmr" in bb_pair:
                        base_rec = bb_pair["gmr"]
                        row_rec["delta_mAP_vs_gmr"] = curve["mAP"] - base_rec["mAP"]
                        row_rec["delta_G1_mean_vs_gmr"] = 100 * (row_rec["G1_mean_456"] - np.mean([base_rec["gmiou"]["1"][i] for i in [40, 50, 60]]))
                    metrics_rows.append(row_rec)

            split_dict["backbones"][bb_key] = bb_pair
        plot_snapshot["splits"][split_key] = split_dict

    return plot_snapshot, metrics_rows, curves_rows


def render_figures(snapshot, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    limit = 0.65
    for split in snapshot["splits"].values():
        for pair in split["backbones"].values():
            for s in pair.values():
                for k in [1, 3]:
                    if "oracle" in s and str(k) in s["oracle"]:
                        limit = max(limit, s["oracle"][str(k)] + 0.04)
    limit = round(np.ceil(limit * 10) / 10, 2)

    n_bb = len(BACKBONES)
    fig_h = 4.2 * n_bb + 3.0

    metadata = {"CreationDate": None, "ModDate": None}

    with PdfPages(output_dir / "BOTH_TESTS.pdf", metadata=metadata) as combined:
        for split_key, split in snapshot["splits"].items():
            fig = plt.figure(figsize=(12.8, fig_h))
            grid = fig.add_gridspec(n_bb, 1, left=0.075, right=0.985, top=0.955, bottom=0.035, hspace=0.42)

            fig.suptitle("SoccerGMR Multi-Backbone Evaluation · GMR Baseline vs. GMR + GEQ vs. SI-Bind (E6) + GEQ", fontsize=16, fontweight="bold", y=0.990)
            fig.text(0.5, 0.978, f"Standard-trained, fixed Val-selected best checkpoints → {split['label']}", ha="center", fontsize=11.5)
            fig.text(0.5, 0.966, "Head-to-head threshold sensitivity comparison: G-mIoU@1, G-mIoU@3 and Oracle Presence Envelopes across τ ∈ [0.00, 1.00]", ha="center", fontsize=10, color="#475569")

            for i, (name, label) in enumerate(BACKBONES):
                inner = grid[i].subgridspec(2, 2, height_ratios=[1, 0.32], hspace=0.48, wspace=0.24)
                series = split["backbones"].get(name, {})

                for j, k in enumerate([1, 3]):
                    ax = fig.add_subplot(inner[0, j])
                    panel(ax, series, k, limit)
                    status_note = "" if "si_bind_e6_geq" in series else " (GEQ Fine-tuning in progress)"
                    ax.set_title(f"{label} · G-mIoU@{k}{status_note}", pad=6, fontsize=10.5, fontweight="bold")
                    if i == n_bb - 1:
                        ax.set_xlabel(r"Existence threshold $\tau$", fontsize=10)

                footer(fig.add_subplot(inner[1, :]), series)

            fig.text(0.5, 0.020, "Curves: 101 fixed existence thresholds (0.00–1.00). Vertical guides indicate τ = 0.4, 0.5, 0.6. Δ: GMR + SI-Bind (E6) + GEQ − GMR baseline.", ha="center", fontsize=9.5, color="#475569")
            fig.text(0.5, 0.012, "Oracle envelope: GT presence with each method's fixed candidate windows; GT-assisted ceiling diagnostic.", ha="center", fontsize=9.5, color="#475569")
            fig.text(0.5, 0.004, "Evaluated without Test tuning or Test checkpoint selection. Native family ranking and duration scaling preserved.", ha="center", fontsize=9.5, color="#475569")

            stem = "STANDARD_TEST" if split_key == "standard" else "FULL_TEST"
            for ext in ["png", "pdf", "svg"]:
                fig.savefig(output_dir / f"{stem}.{ext}", dpi=200, facecolor="white", **({"metadata": metadata} if ext == "pdf" else {}))
            combined.savefig(fig)
            plt.close(fig)

            # Generate individual per-backbone pages
            with PdfPages(output_dir / f"{stem}_BY_BACKBONE.pdf", metadata=metadata) as pages:
                for i, (name, label) in enumerate(BACKBONES):
                    fig_single = plt.figure(figsize=(12.4, 4.8))
                    grid_single = fig_single.add_gridspec(2, 2, left=0.075, right=0.985, top=0.79, bottom=0.08, height_ratios=[1, 0.28], hspace=0.22, wspace=0.24)
                    series = split["backbones"].get(name, {})
                    for j, k in enumerate([1, 3]):
                        ax = fig_single.add_subplot(grid_single[0, j])
                        panel(ax, series, k, limit)
                        ax.set_title(f"G-mIoU@{k}", fontsize=11)
                        ax.set_xlabel(r"Existence threshold $\tau$")
                    footer(fig_single.add_subplot(grid_single[1, :]), series)
                    fig_single.suptitle(f"{label} · {split['label']}", fontsize=13.5, fontweight="bold", y=0.97)
                    fig_single.text(0.5, 0.88, "SI-Bind (E6) + GEQ Full fine-tuning; min Val Orc-gap selection (20 epochs patience)", ha="center", fontsize=10)
                    pages.savefig(fig_single)
                    plt.close(fig_single)

    print(f"Figures rendered successfully in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(WORKSPACE_ROOT / "results" / "test_thresholds_si_bind_geq"))
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building plot snapshot and extracting curves...")
    snapshot, metrics_rows, curves_rows = build_plot_data()

    # Save data
    with open(out_dir / "PLOT_DATA.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    if metrics_rows:
        with open(out_dir / "METRICS.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)

    if curves_rows:
        with open(out_dir / "CURVES.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(curves_rows[0].keys()))
            writer.writeheader()
            writer.writerows(curves_rows)

    print("Rendering publication figures...")
    render_figures(snapshot, out_dir)


if __name__ == "__main__":
    main()
