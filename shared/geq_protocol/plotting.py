"""Reusable six-panel figures; measurements and offline rendering are separate."""
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np

NAMES = {"moment_detr": "Moment-DETR", "eatr": "EaTR", "qd_detr": "QD-DETR", "cg_detr": "CG-DETR", "sim_detr": "Sim-DETR"}
METHODS = ()
LABELS = {}
STYLES = {}
GRID = [i / 100 for i in range(101)]
PALETTE = (("#64748B", "--", "o"), ("#E69F00", ":", "D"), ("#009E73", "-.", "^"), ("#0072B2", "-", "s"))


def setup(methods, labels):
    global METHODS, LABELS, STYLES
    METHODS = tuple(methods)
    LABELS = dict(labels)
    STYLES = {m: dict(color=PALETTE[i][0], linestyle=PALETTE[i][1], marker=PALETTE[i][2]) for i, m in enumerate(methods)}

def configure_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "gmr-geq-three-way-20261031", "savefig.facecolor": "white"})


def save(figure, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        path = destination.with_suffix("." + suffix)
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Date": None} if suffix == "svg" else None
        figure.savefig(temporary, dpi=220, bbox_inches="tight", pad_inches=0.12, metadata=metadata)
        if suffix == "svg":
            temporary.write_text("\n".join(line.rstrip() for line in temporary.read_text().splitlines()) + "\n")
        temporary.replace(path)
    plt.close(figure)


def axis(ax, title, xlabel, ylabel, bounded=True):
    ax.set_title(title, loc="left", pad=10)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", color="#DFE7F1", linewidth=0.7)
    ax.set_axisbelow(True)
    if bounded:
        ax.set_ylim(0, 1)


def line(ax, x, y, method, markers=True):
    style = STYLES[method].copy()
    if not markers:
        style.pop("marker")
    return ax.plot(x, y, **style, linewidth=1.9, markersize=3.8,
                   markevery=10 if markers else None, label=LABELS[method])[0]


def render(backbone, data, output):
    series = data["series"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.9))
    figure.subplots_adjust(left=0.065, right=0.986, bottom=0.11, top=0.855, hspace=0.4, wspace=0.29)
    g1, g3, delta, tradeoff, f1, fixed = axes.flat
    axis(g1, "(a) Threshold sensitivity", "Presence threshold", "G-mIoU@1 (0–1)")
    axis(g3, "(b) Top-3 retrieval", "Presence threshold", "G-mIoU@3 (0–1)")
    axis(delta, "(c) Relative to " + LABELS["baseline"], "Presence threshold", "Δ G-mIoU@1 (0–1 units)", bounded=False)
    axis(tradeoff, "(d) Rejection / positive retention", "Positive recall (0–1)", "N-acc (0–1)")
    axis(f1, "(e) Negative-class F1", "Presence threshold", "Rejection F1 (0–1)")
    fixed.set(xlabel="Presence threshold", ylabel="G-mIoU@1 (0–1)", ylim=(0, 1))
    fixed.set_title("(f) Fixed operating points", loc="left", pad=10)
    fixed.grid(axis="y", color="#DFE7F1", linewidth=0.7)
    fixed.set_axisbelow(True)
    baseline = np.asarray([row["gmiou_1"] for row in series["baseline"]["grid"]])
    deltas = []
    for method in METHODS:
        rows_ = series[method]["grid"]
        values1 = np.asarray([row["gmiou_1"] for row in rows_])
        difference = values1 - baseline
        line(g1, GRID, values1, method)
        line(g3, GRID, [row["gmiou_3"] for row in rows_], method)
        line(delta, GRID, difference, method)
        deltas.extend(difference.tolist())
        empirical = series[method]["empirical_tradeoff"]
        line(tradeoff, [r["positive_recall"] for r in empirical], [r["negative_accuracy"] for r in empirical], method, markers=False)
        for threshold in (0.4, 0.5, 0.6):
            point = rows_[round(threshold * 100)]
            tradeoff.scatter(point["positive_recall"], point["negative_accuracy"],
                             color=STYLES[method]["color"], marker=STYLES[method]["marker"], s=29, zorder=5)
        line(f1, GRID, [row["rejection_f1"] for row in rows_], method)
        index = METHODS.index(method)
        width = 0.18
        positions = np.arange(3) + (index - (len(METHODS) - 1) / 2) * width
        values = [rows_[round(t * 100)]["gmiou_1"] for t in (0.4, 0.5, 0.6)]
        bars = fixed.bar(positions, values, width=width * 0.94, color=STYLES[method]["color"], zorder=3)
        fixed.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, rotation=90, fontsize=8.5)
    fixed.set_xticks(range(3), ("0.4", "0.5", "0.6"))
    for panel in (g1, g3, delta, f1):
        panel.axvline(0.4, color="#94A3B8", linestyle=":", linewidth=0.7, zorder=0)
    delta.axhline(0, color="#94A3B8", linewidth=0.7, zorder=0)
    low, high = min(deltas), max(deltas)
    span = max(0.05, high - low)
    delta.set_ylim(low - 0.10 * span, high + 0.14 * span)
    delta.yaxis.set_major_locator(MaxNLocator(6))
    handles = [Line2D([0], [0], **STYLES[m], linewidth=1.9, markersize=4.5,
                label=f"{LABELS[m]}  (mAP {series[m]['metrics']['mAP']:.2f}%)") for m in METHODS]
    figure.suptitle(NAMES[backbone] + " | SoccerGMR Standard Test", fontsize=14, fontweight="bold", y=0.99)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.948), ncol=4,
                  frameon=False, columnspacing=1.1, handlelength=2.4, fontsize=10)
    note = data["protocol_note"]
    figure.text(0.5, 0.037, note, ha="center", va="bottom", fontsize=9.5)
    protocol = data["seed_note"]
    figure.text(0.5, 0.014, protocol, ha="center", va="bottom", fontsize=9)
    save(figure, output / backbone / data["figure_name"])
    # Large two-panel export isolates the two explicitly requested metrics.
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    figure.subplots_adjust(top=0.72, bottom=0.18, wspace=0.30, left=0.08, right=0.99)
    for ax, metric, title in zip(axes, ("gmiou_1", "gmiou_3"), ("G-mIoU@1", "G-mIoU@3")):
        axis(ax, title, "Presence threshold", title + " (0–1)")
        for method in METHODS:
            line(ax, GRID, [r[metric] for r in series[method]["grid"]], method)
        ax.axvline(0.4, color="#94A3B8", linestyle=":", linewidth=0.7)
    figure.suptitle(NAMES[backbone] + " | " + data["subtitle"], fontsize=12, y=0.995)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, fontsize=9, frameon=False)
    figure.text(0.5, 0.005, "SoccerGMR Standard Test | Val selected | 101 thresholds | G-mIoU: 0–1", ha="center", fontsize=9)
    save(figure, output / backbone / (data["figure_name"] + "_gmiou_1_3"))

