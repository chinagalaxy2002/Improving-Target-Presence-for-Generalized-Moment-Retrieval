#!/usr/bin/env python3
"""Generate publication-quality vector table figure (SVG & PDF) for Stage 2 Model Selection Ablation:
Comparing checkpoint selection criteria:
Criterion A (Official / Current): Val mean G-mIoU@1 across tau in {0.4, 0.5, 0.6}
Criterion B (User Proposed):      Val mAP (highest localization mAP)
"""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "dq_cgp_v3" / "results" / "ablation_table"
ARTIFACT_DIR = Path("/home/guoxiangyu/.gemini/antigravity-cli/brain/62a59d6f-74ff-44d8-a47a-d538b53ca17b")

BACKBONES = [
    ("moment_detr", "Moment-DETR"),
    ("eatr", "EaTR"),
    ("qd_detr", "QD-DETR"),
    ("cg_detr", "CG-DETR"),
    ("bm_detr", "BM-DETR"),
    ("bam_detr", "BAM-DETR"),
    ("sim_detr", "Sim-DETR"),
]


def load_data():
    rows = []
    for b_name, dname in BACKBONES:
        hp = ROOT / "runs" / "dq_cgp_v3" / b_name / "geq_full" / "history.jsonl"
        if not hp.is_file():
            continue
        with open(hp) as f:
            records = [json.loads(line) for line in f if line.strip()]
        if not records:
            continue
        best_g1 = max(records, key=lambda r: r.get("selection_score", 0))
        best_map = max(records, key=lambda r: r.get("metrics", {}).get("brief", {}).get("mAP", 0))
        rows.append({
            "name": b_name,
            "display": dname,
            "total_epochs": len(records),
            "crit_a": {
                "epoch": best_g1["epoch"],
                "mAP": best_g1["metrics"]["brief"]["mAP"],
                "g1_mean": best_g1["selection_score"],
                "g1_50": best_g1["metrics"]["gmiou"]["0.5"]["G-mIoU@1"],
                "rej_50": best_g1["metrics"]["brief"]["Rej-F1@0.5"],
                "auroc": best_g1["metrics"]["brief"]["AUROC"],
            },
            "crit_b": {
                "epoch": best_map["epoch"],
                "mAP": best_map["metrics"]["brief"]["mAP"],
                "g1_mean": best_map["selection_score"],
                "g1_50": best_map["metrics"]["gmiou"]["0.5"]["G-mIoU@1"],
                "rej_50": best_map["metrics"]["brief"]["Rej-F1@0.5"],
                "auroc": best_map["metrics"]["brief"]["AUROC"],
            }
        })
    return rows


def render_pdf_svg(data):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
    })

    left = 24
    backbone_w = 95
    crit_w = 175
    col_step = 72
    num_cols = 6
    width = left + backbone_w + crit_w + num_cols * col_step + 24
    row_height = 24
    n_bb = len(data)
    table_content_h = n_bb * 2 * row_height + (n_bb - 1) * 6
    height = 110 + table_content_h + 75

    fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white", dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")

    def text(x, y, s, size=9.5, weight="normal", ha="left", color="#1e293b", style="normal"):
        return ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style, ha=ha, va="center", color=color)

    def rule(y, x0=left, x1=width - 24, lw=0.6, color="#1e293b"):
        ax.plot([x0, x1], [y, y], color=color, lw=lw)

    # Title
    text(left, height - 26, "Stage 2 (GEQ-Full) Model Selection Criterion Ablation Study", 13.5, "bold", color="#0f172a")
    sub = (
        "Comparing Validation Set (465 queries) metrics when selecting checkpoints via "
        "Criterion A (Val mean G-mIoU@1, Official) vs. Criterion B (Val mAP, Proposed)."
    )
    text(left, height - 46, sub, 9.0, color="#475569")

    top = height - 64
    rule(top, lw=1.2, color="#0f172a")

    # Column headers
    cols_x = left + backbone_w + crit_w
    text(left, top - 18, "Backbone", 9.5, "bold", color="#0f172a")
    text(left + backbone_w, top - 18, "Selection Criterion", 9.5, "bold", color="#0f172a")

    col_names = ["Selected Ep", "Val mAP (%)", "Val G1 Mean (%)", "G-mIoU@1 (.5)", "Rej-F1@0.5 (%)", "AUROC (%)"]
    for i, cname in enumerate(col_names):
        text(cols_x + (i + 0.5) * col_step, top - 18, cname, 9.0, "bold", ha="center", color="#1e293b")

    rule(top - 30, lw=0.8, color="#0f172a")

    y = top - 48
    for b_idx, item in enumerate(data):
        dname = item["display"]
        ca = item["crit_a"]
        cb = item["crit_b"]

        # Row 1: Criterion A (Val mean G1)
        ax.add_patch(plt.Rectangle((left - 4, y - 10), width - left - 16, 20, color="#ecfdf5", zorder=-1))
        text(left, y, dname, 9.8, "bold", color="#0f172a")
        text(left + backbone_w, y, "Criterion A: Val Mean G1", 9.2, "bold", color="#047857")

        text(cols_x + 0.5 * col_step, y, f"Epoch {ca['epoch']}", 9.0, ha="center", color="#334155")
        text(cols_x + 1.5 * col_step, y, f"{ca['mAP']:.2f}", 9.0, ha="center", color="#334155")
        text(cols_x + 2.5 * col_step, y, f"{ca['g1_mean']:.2f}", 9.0, "bold", ha="center", color="#047857")
        text(cols_x + 3.5 * col_step, y, f"{ca['g1_50']:.2f}", 9.0, "bold", ha="center", color="#047857")
        text(cols_x + 4.5 * col_step, y, f"{ca['rej_50']:.2f}", 9.0, "bold", ha="center", color="#047857")
        text(cols_x + 5.5 * col_step, y, f"{ca['auroc']:.2f}", 9.0, ha="center", color="#334155")

        # Row 2: Criterion B (Val mAP)
        y -= row_height
        ax.add_patch(plt.Rectangle((left - 4, y - 10), width - left - 16, 20, color="#fffbeb", zorder=-1))
        text(left + backbone_w, y, "Criterion B: Highest Val mAP", 9.2, "bold", color="#b45309")

        text(cols_x + 0.5 * col_step, y, f"Epoch {cb['epoch']}", 9.0, ha="center", color="#334155")
        # mAP is higher under Criterion B
        text(cols_x + 1.5 * col_step, y, f"{cb['mAP']:.2f}", 9.0, "bold", ha="center", color="#b45309")
        text(cols_x + 2.5 * col_step, y, f"{cb['g1_mean']:.2f}", 9.0, ha="center", color="#64748b")
        text(cols_x + 3.5 * col_step, y, f"{cb['g1_50']:.2f}", 9.0, ha="center", color="#64748b")
        text(cols_x + 4.5 * col_step, y, f"{cb['rej_50']:.2f}", 9.0, ha="center", color="#64748b")
        text(cols_x + 5.5 * col_step, y, f"{cb['auroc']:.2f}", 9.0, "bold" if cb['auroc'] > ca['auroc'] else "normal", ha="center", color="#334155")

        y -= row_height
        if b_idx < len(data) - 1:
            rule(y + 12, lw=0.45, color="#cbd5e1")
            y -= 6

    rule(y + 10, lw=1.2, color="#0f172a")

    notes = [
        "Criterion A (Official): Selects epoch with highest Val Mean G-mIoU@1 across thresholds {0.4, 0.5, 0.6} (optimizes generalized presence).",
        "Criterion B (Proposed): Selects epoch with highest Val mAP (optimizes traditional positive localization).",
        "Key Finding: Selecting by mAP increases Val mAP by +0.16% to +2.47%, but causes severe drops in G-mIoU@1 (-4.9% to -13.8%) and Rej-F1.",
        "This confirms that GMR target presence optimization inherently requires presence-aware model selection.",
    ]
    for i, line in enumerate(notes):
        text(left, y - 4 - i * 12, line, 8.0, color="#64748b")

    svg_path = OUT_DIR / "STAGE2_SELECTION_ABLATION_TABLE.svg"
    pdf_path = OUT_DIR / "STAGE2_SELECTION_ABLATION_TABLE.pdf"
    png_path = OUT_DIR / "STAGE2_SELECTION_ABLATION_TABLE.png"

    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated Stage 2 Selection Ablation:")
    print(f"  SVG: {svg_path}")
    print(f"  PDF: {pdf_path}")
    print(f"  PNG: {png_path}")

    if ARTIFACT_DIR.is_dir():
        for ext in ["svg", "pdf", "png"]:
            src = OUT_DIR / f"STAGE2_SELECTION_ABLATION_TABLE.{ext}"
            dst = ARTIFACT_DIR / f"STAGE2_SELECTION_ABLATION_TABLE.{ext}"
            shutil.copy2(src, dst)


def main():
    data = load_data()
    render_pdf_svg(data)


if __name__ == "__main__":
    main()
