"""Render the six-route SoccerGMR test evaluation figure."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / "mr_experiments/test_summary/results.json"
SWEEP_SOURCE = OUT / "gmiou_threshold_sweep.csv"

ORDER = ("baseline", "shared", "sepffn", "instruction", "multi", "adaptive")
LABELS = {
    "baseline": "E0 Baseline",
    "shared": "E1 Shared O2M",
    "sepffn": "E2 Separate FFN",
    "instruction": "E3 Instruction",
    "multi": "E4 Multi-Route",
    "adaptive": "E5 Adaptive",
}
COLORS = {
    "baseline": "#4D4D4D",
    "shared": "#0072B2",
    "sepffn": "#E69F00",
    "instruction": "#009E73",
    "multi": "#CC79A7",
    "adaptive": "#D55E00",
}
HATCHES = {
    "baseline": "",
    "shared": "///",
    "sepffn": "\\\\\\",
    "instruction": "xxx",
    "multi": "...",
    "adaptive": "ooo",
}
MARKERS = {
    "baseline": "o",
    "shared": "s",
    "sepffn": "^",
    "instruction": "D",
    "multi": "P",
    "adaptive": "X",
}
LINESTYLES = {
    "baseline": "--",
    "shared": "-",
    "sepffn": "-.",
    "instruction": ":",
    "multi": (0, (3, 1, 1, 1)),
    "adaptive": (0, (5, 1)),
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.07, label, transform=ax.transAxes, fontsize=11, fontweight="bold")


def annotate_best_bar(ax: plt.Axes, bars, values) -> None:
    best = int(np.argmax(values))
    y_min, y_max = ax.get_ylim()
    label_offset = 0.025 * (y_max - y_min)
    for index, (bar, value) in enumerate(zip(bars, values)):
        if index == best:
            bar.set_linewidth(1.5)
            bar.set_edgecolor("#111111")
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold" if index == best else "normal",
        )


def main() -> None:
    configure_style()
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = {row["experiment"]: row for row in raw}
    if set(rows) != set(ORDER):
        raise ValueError(f"Expected {ORDER}, found {tuple(rows)}")
    with SWEEP_SOURCE.open(encoding="utf-8", newline="") as handle:
        sweep_rows = list(csv.DictReader(handle))
    sweeps = {
        method: [row for row in sweep_rows if row["experiment"] == method]
        for method in ORDER
    }
    if any(len(method_rows) != 101 for method_rows in sweeps.values()):
        raise ValueError("The G-mIoU sweep must contain 101 thresholds per experiment")

    figure, axes = plt.subplots(2, 3, figsize=(13.4, 7.5), constrained_layout=False)
    figure.subplots_adjust(left=0.065, right=0.985, top=0.88, bottom=0.09, wspace=0.29, hspace=0.39)

    # (a) Main localization metrics.
    ax = axes[0, 0]
    metrics = ("mAP", "mR@1", "mR@5")
    x = np.arange(len(metrics))
    width = 0.125
    for method_index, method in enumerate(ORDER):
        values = [rows[method][metric] for metric in metrics]
        offset = (method_index - (len(ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            hatch=HATCHES[method],
            label=LABELS[method],
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.22,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.3,
                rotation=90,
            )
    ax.set_title("Positive-query localization")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 15)
    ax.set_yticks(np.arange(0, 16, 3))
    ax.grid(axis="y", zorder=0)
    add_panel_label(ax, "(a)")

    # (b) Extra-moment recall, the key multi-moment metric.
    ax = axes[0, 1]
    values = [rows[method]["mR+@5"] for method in ORDER]
    bars = ax.bar(
        np.arange(len(ORDER)),
        values,
        color=[COLORS[method] for method in ORDER],
        edgecolor="white",
        hatch=[HATCHES[method] for method in ORDER],
        zorder=3,
    )
    ax.set_title("Additional-moment recall")
    ax.set_ylabel("mR+@5 (%)")
    ax.set_xticks(np.arange(len(ORDER)), [f"E{i}" for i in range(6)])
    ax.set_ylim(0, 1.75)
    ax.set_yticks(np.arange(0, 1.76, 0.35))
    ax.grid(axis="y", zorder=0)
    annotate_best_bar(ax, bars, values)
    add_panel_label(ax, "(b)")

    # (c) Threshold-independent presence ranking.
    ax = axes[0, 2]
    values = [rows[method]["AUROC"] for method in ORDER]
    bars = ax.bar(
        np.arange(len(ORDER)),
        values,
        color=[COLORS[method] for method in ORDER],
        edgecolor="white",
        hatch=[HATCHES[method] for method in ORDER],
        zorder=3,
    )
    ax.set_title("Existence-score ranking")
    ax.set_ylabel("AUROC (%)")
    ax.set_xticks(np.arange(len(ORDER)), [f"E{i}" for i in range(6)])
    ax.set_ylim(66, 72)
    ax.set_yticks(np.arange(66, 73, 1))
    ax.grid(axis="y", zorder=0)
    annotate_best_bar(ax, bars, values)
    ax.text(0.02, 0.03, "Zoomed y-axis", transform=ax.transAxes, color="#555555", fontsize=7)
    add_panel_label(ax, "(c)")

    # (d) Rejection F1 sensitivity.
    thresholds = np.array([0.4, 0.5, 0.6])
    ax = axes[1, 0]
    for method in ORDER:
        values = [rows[method][f"Rej-F1@{threshold:.1f}"] for threshold in thresholds]
        ax.plot(
            thresholds,
            values,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.7,
            markersize=5.2,
            zorder=3,
        )
    ax.set_title("Null-set rejection sensitivity")
    ax.set_xlabel("Existence threshold $\\tau$")
    ax.set_ylabel("Rej-F1 (%)")
    ax.set_xlim(0.385, 0.615)
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 82)
    ax.grid(zorder=0)
    add_panel_label(ax, "(d)")

    # (e) Combined GMR sensitivity over the full threshold domain.
    ax = axes[1, 1]
    for method in ORDER:
        full_thresholds = [float(row["threshold"]) for row in sweeps[method]]
        values = [float(row["G-mIoU@3"]) for row in sweeps[method]]
        ax.plot(
            full_thresholds,
            values,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.7,
            zorder=3,
        )
        selected = [int(round(threshold * 100)) for threshold in thresholds]
        ax.scatter(
            thresholds,
            [values[index] for index in selected],
            color=COLORS[method],
            marker=MARKERS[method],
            s=25,
            zorder=4,
        )
    ax.set_title("Localization–rejection quality")
    ax.set_xlabel("Existence threshold $\\tau$")
    ax.set_ylabel("G-mIoU@3 (%)")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_ylim(0, 53)
    ax.grid(zorder=0)
    add_panel_label(ax, "(e)")

    # (f) The central trade-off: localization quality versus extra-moment coverage.
    ax = axes[1, 2]
    for method_index, method in enumerate(ORDER):
        row = rows[method]
        ax.scatter(
            row["mAP"],
            row["mR+@5"],
            s=62 if method != "baseline" else 54,
            color=COLORS[method],
            marker=MARKERS[method],
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
        )
        offsets = [(4, 5), (4, -11), (4, -11), (4, -11), (4, 5), (4, 5)]
        ax.annotate(
            f"E{method_index}",
            (row["mAP"], row["mR+@5"]),
            xytext=offsets[method_index],
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
        )
    ax.axvline(rows["baseline"]["mAP"], color="#AAAAAA", linestyle=":", linewidth=0.9)
    ax.axhline(rows["baseline"]["mR+@5"], color="#AAAAAA", linestyle=":", linewidth=0.9)
    ax.set_title("Localization–coverage trade-off")
    ax.set_xlabel("mAP (%) →")
    ax.set_ylabel("mR+@5 (%) →")
    ax.set_xlim(5.5, 7.8)
    ax.set_ylim(0.15, 1.62)
    ax.grid(zorder=0)
    add_panel_label(ax, "(f)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
        frameon=True,
        columnspacing=1.8,
        handlelength=2.3,
    )
    legend.get_frame().set_edgecolor("#C8C8C8")
    legend.get_frame().set_linewidth(0.7)

    for extension in ("pdf", "svg", "png"):
        figure.savefig(
            OUT / f"gmr_mr_detr_test_evaluation.{extension}",
            format=extension,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)

    fields = list(raw[0])
    with (OUT / "figure_data.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(raw)
    print(f"Rendered six-route evaluation from {SOURCE}")


if __name__ == "__main__":
    main()
