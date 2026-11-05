#!/usr/bin/env python3
"""Generate publication-quality vector table figure (SVG & PDF) for completed backbones.
Ablation across:
1. GMR Baseline
2. GMR + DQ-CGPv3
3. GMR + DQ-CGPv3 + GEQ

Evaluated on:
- Standard Test (1,036 queries)
- Full-Decodable Test (2,970 queries)

Metrics:
- mAP (%)
- G-mIoU@1 (0-1 Integral / AUC)
- Rej-F1@0.5 (%)
- Rej-F1@0.4 (%)
- Rej-F1@0.6 (%)
- AUROC (%)
"""
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "runs" / "dq_cgp_v3" / "results" / "ablation_study_results.json"
OUT_DIR = ROOT / "runs" / "dq_cgp_v3" / "results" / "ablation_table"
ARTIFACT_DIR = Path("/home/guoxiangyu/.gemini/antigravity-cli/brain/62a59d6f-74ff-44d8-a47a-d538b53ca17b")

ALL_BACKBONES = [
    ("moment_detr", "Moment-DETR"),
    ("eatr", "EaTR"),
    ("qd_detr", "QD-DETR"),
    ("cg_detr", "CG-DETR"),
    ("bm_detr", "BM-DETR"),
    ("bam_detr", "BAM-DETR"),
    ("sim_detr", "Sim-DETR"),
    ("uvcom", "UVCOM"),
    ("flashvtg", "FlashVTG"),
]

METHODS = [
    ("gmr", "1. GMR Baseline"),
    ("gmr_dqcgpv3", "2. GMR + DQ-CGPv3"),
    ("gmr_dqcgpv3_geq", "3. GMR + DQ-CGPv3 + GEQ"),
]


def render_table(backbones, filename_stem, title_count_str):
    if not RESULTS_FILE.is_file():
        return
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    # Filter backbones that actually exist in data
    valid_backbones = []
    for b_name, d_name in backbones:
        if b_name in data:
            # Check if at least gmr_dqcgpv3 is available
            if data[b_name]["splits"]["standard"].get("gmr_dqcgpv3") is not None:
                valid_backbones.append((b_name, d_name))

    if not valid_backbones:
        return

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
    })

    left = 24
    backbone_w = 95
    config_w = 175
    col_step = 68
    num_std_cols = 6
    num_full_cols = 6
    
    std_left = left + backbone_w + config_w
    full_left = std_left + num_std_cols * col_step + 18
    width = full_left + num_full_cols * col_step + 24
    
    row_height = 24
    n_bb = len(valid_backbones)
    table_content_h = n_bb * 3 * row_height + (n_bb - 1) * 4
    header_h = 100
    footer_h = 60
    height = header_h + table_content_h + footer_h

    fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white", dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")

    def text(x, y, s, size=9.5, weight="normal", ha="left", color="#1e293b", style="normal"):
        return ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style, ha=ha, va="center", color=color)

    def rule(y, x0=left, x1=width - 24, lw=0.6, color="#1e293b"):
        ax.plot([x0, x1], [y, y], color=color, lw=lw)

    # Title & Subtitle
    text(left, height - 26, f"SoccerGMR Ablation Study: {title_count_str} × 3 Configurations", 14, "bold", color="#0f172a")
    sub = (
        "Comparing Baseline GMR vs. GMR + DQ-CGPv3 vs. GMR + DQ-CGPv3 + GEQ across Standard Test (1,036 queries) and Full-Decodable Test (2,970 queries). "
        "G1-AUC = integral of G-mIoU@1 across tau in [0.00, 1.00]."
    )
    text(left, height - 46, sub, 9.2, color="#475569")

    top = height - 64
    rule(top, lw=1.2, color="#0f172a")

    # Super-headers for Splits
    std_mid = std_left + (num_std_cols * col_step) / 2
    text(std_mid, top - 14, "Standard Test Split (1,036 queries)", 11, "bold", ha="center", color="#1e3a8a")
    rule(top - 24, std_left + 2, std_left + num_std_cols * col_step - 4, lw=0.9, color="#2563eb")

    full_mid = full_left + (num_full_cols * col_step) / 2
    text(full_mid, top - 14, "Full-Decodable Test Split (2,970 queries)", 11, "bold", ha="center", color="#065f46")
    rule(top - 24, full_left + 2, full_left + num_full_cols * col_step - 4, lw=0.9, color="#059669")

    for base_x in (std_left, full_left):
        text(base_x + 0.5 * col_step, top - 35, "Loc", 8.5, ha="center", style="italic", color="#64748b")
        text(base_x + 1.5 * col_step, top - 35, "Overall", 8.5, ha="center", style="italic", color="#64748b")
        text(base_x + 4.0 * col_step, top - 35, "Existence & Null-set Rejection", 8.5, ha="center", style="italic", color="#64748b")
        rule(top - 44, base_x + 2, base_x + num_std_cols * col_step - 4, lw=0.4, color="#94a3b8")

    text(left, top - 55, "Backbone", 9.5, "bold", color="#0f172a")
    text(left + backbone_w, top - 55, "Configuration", 9.5, "bold", color="#0f172a")

    col_names = ["mAP (%)", "G1-AUC (%)", "Rej-F1@.5", "Rej-F1@.4", "Rej-F1@.6", "AUROC (%)"]
    for i, cname in enumerate(col_names):
        text(std_left + (i + 0.5) * col_step, top - 55, cname, 8.8, "bold", ha="center", color="#1e293b")
        text(full_left + (i + 0.5) * col_step, top - 55, cname, 8.8, "bold", ha="center", color="#1e293b")

    rule(top - 67, lw=0.8, color="#0f172a")

    y = top - 86

    for b_idx, (b_name, display_name) in enumerate(valid_backbones):
        b_data = data[b_name]["splits"]

        for m_idx, (m_key, m_label) in enumerate(METHODS):
            curr_y = y - m_idx * row_height

            if m_key == "gmr_dqcgpv3_geq":
                ax.add_patch(plt.Rectangle((left - 4, curr_y - 10), width - left - 16, 20, color="#ecfdf5", zorder=-1))
            elif m_key == "gmr_dqcgpv3":
                ax.add_patch(plt.Rectangle((left - 4, curr_y - 10), width - left - 16, 20, color="#f0f9ff", zorder=-1))

            if m_idx == 0:
                text(left, curr_y, display_name, 9.8, "bold", color="#0f172a")

            m_color = "#047857" if m_key == "gmr_dqcgpv3_geq" else "#0369a1" if m_key == "gmr_dqcgpv3" else "#334155"
            m_weight = "bold" if m_key == "gmr_dqcgpv3_geq" else "normal"
            text(left + backbone_w, curr_y, m_label, 9.2, m_weight, color=m_color)

            for split_i, (split_key, base_x) in enumerate([("standard", std_left), ("full", full_left)]):
                row_metrics = b_data[split_key].get(m_key)
                if not row_metrics:
                    text(base_x + 2.5 * col_step, curr_y, "-- running / pending --", size=8.2, style="italic", ha="center", color="#94a3b8")
                    continue

                vals = [
                    row_metrics.get("mAP"),
                    row_metrics.get("g1_auc_pct"),
                    row_metrics.get("rej_f1_50"),
                    row_metrics.get("rej_f1_40"),
                    row_metrics.get("rej_f1_60"),
                    row_metrics.get("auroc"),
                ]

                all_avail = [b_data[split_key][mk] for mk, _ in METHODS if b_data[split_key].get(mk)]
                
                for c_idx, val in enumerate(vals):
                    if val is None:
                        continue
                    
                    metric_keys = ["mAP", "g1_auc_pct", "rej_f1_50", "rej_f1_40", "rej_f1_60", "auroc"]
                    all_vals = [m[metric_keys[c_idx]] for m in all_avail if m.get(metric_keys[c_idx]) is not None]
                    best_val = max(all_vals) if all_vals else val
                    is_best = (val == best_val) and (len(set(all_vals)) > 1)

                    v_str = f"{val:.2f}"
                    val_color = "#047857" if (is_best and m_key == "gmr_dqcgpv3_geq") else "#0f172a" if is_best else "#475569"
                    val_weight = "bold" if is_best else "normal"

                    text(base_x + (c_idx + 0.5) * col_step, curr_y, v_str, 9.0, val_weight, ha="center", color=val_color)

        y -= 3 * row_height
        if b_idx < len(valid_backbones) - 1:
            rule(y + 12, lw=0.45, color="#cbd5e1")
            y -= 4

    rule(y + 10, lw=1.2, color="#0f172a")

    notes = [
        "1. GMR Baseline: Historical unaugmented GMR. 2. GMR + DQ-CGPv3: Plug-and-play DETR Query CGP (beta=0.05, K=16) between Layer 0 & 1.",
        "3. GMR + DQ-CGPv3 + GEQ: Global Event Query head fine-tuned with BCE + native loss (lr=1e-5 / 3e-5, patience=10 on Val G1 mean).",
        "G1-AUC (%): Area Under the Curve of G-mIoU@1 across 101 discrete presence thresholds tau in [0.00, 1.00] (higher is strictly better).",
        "Bold values: Best performance among the available ablation variants within each backbone family.",
    ]
    for i, line in enumerate(notes):
        text(left, y - 4 - i * 12, line, 8.0, color="#64748b")

    svg_path = OUT_DIR / f"{filename_stem}.svg"
    pdf_path = OUT_DIR / f"{filename_stem}.pdf"
    png_path = OUT_DIR / f"{filename_stem}.png"

    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated {filename_stem}: SVG, PDF, PNG")
    if ARTIFACT_DIR.is_dir():
        for ext in ["svg", "pdf", "png"]:
            src = OUT_DIR / f"{filename_stem}.{ext}"
            dst = ARTIFACT_DIR / f"{filename_stem}.{ext}"
            shutil.copy2(src, dst)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    # 1. Generate the 5-backbone table
    five_backbones = ALL_BACKBONES[:5]
    render_table(five_backbones, "ABLATION_TABLE_5_BACKBONES", "5 Backbones")

    # 2. Generate the 7-backbone table (as requested by user)
    seven_backbones = ALL_BACKBONES[:7]
    render_table(seven_backbones, "ABLATION_TABLE_7_BACKBONES", "7 Backbones")

    # 3. Check fully completed backbones
    fully_completed = []
    for b_name, d_name in ALL_BACKBONES:
        if b_name in data:
            std_geq = data[b_name]["splits"]["standard"].get("gmr_dqcgpv3_geq")
            if std_geq is not None:
                fully_completed.append((b_name, d_name))

    if len(fully_completed) > 5:
        count_str = f"{len(fully_completed)} Backbones"
        render_table(fully_completed, f"ABLATION_TABLE_{len(fully_completed)}_BACKBONES", count_str)
        render_table(fully_completed, "ABLATION_TABLE_ALL", count_str)


if __name__ == "__main__":
    main()
