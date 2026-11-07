#!/usr/bin/env python3
"""Generate publication-quality vector table figure (SVG & PDF) for the 4-way ablation study:
1. GMR Baseline
2. GMR + DQ-CGPv3
3. GMR + DQ-CGPv3 + GEQ (Val G1 Selection - Official)
4. GMR + DQ-CGPv3 + GEQ (Val mAP Selection - Proposed)

Evaluated on:
- Standard Test (1,036 queries)
- Full-Decodable Test (2,970 queries)

Metrics reported:
- mAP (%)            [Higher is better]
- Orc-Gap (%)        [Lower is better: Oracle envelope integral (0-1) minus G1-AUC]
- Rej-F1@.5 (%)      [Higher is better]
- AUROC (%)          [Higher is better]
"""
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "runs" / "dq_cgp_v3" / "results" / "ablation_study_geq_map_results.json"
OUT_DIR = ROOT / "runs" / "dq_cgp_v3" / "results" / "ablation_table"
DOCS_IMAGES = ROOT / "docs" / "images"
DOCS_TABLES = ROOT / "docs" / "tables"
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
    ("gmr", "1. GMR Baseline", "#334155", "normal"),
    ("gmr_dqcgpv3", "2. GMR + DQ-CGPv3", "#0369a1", "normal"),
    ("gmr_dqcgpv3_geq_g1", "3. GMR + DQ-CGPv3 + GEQ (G1 Sel)", "#047857", "bold"),
    ("gmr_dqcgpv3_geq_map", "4. GMR + DQ-CGPv3 + GEQ (mAP Sel)", "#b45309", "bold"),
    ("gmr_si_bind", "5. GMR + SI-Bind (Ours)", "#6d28d9", "bold"),
]


def render_table(backbones, filename_stem):
    if not RESULTS_FILE.is_file():
        return
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    valid_backbones = []
    for b_name, d_name in backbones:
        if b_name in data and data[b_name]["splits"]["standard"].get("gmr") is not None:
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
    config_w = 210
    col_step = 72
    num_std_cols = 4  # mAP, Orc-Gap, Rej-F1@.5, AUROC
    num_full_cols = 4

    std_left = left + backbone_w + config_w
    full_left = std_left + num_std_cols * col_step + 18
    width = full_left + num_full_cols * col_step + 24

    row_height = 23
    n_bb = len(valid_backbones)
    table_content_h = n_bb * len(METHODS) * row_height + (n_bb - 1) * 6
    height = 110 + table_content_h + 88

    fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white", dpi=200)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")

    def text(x, y, s, size=9.5, weight="normal", ha="left", color="#1e293b", style="normal"):
        return ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style, ha=ha, va="center", color=color)

    def rule(y, x0=left, x1=width - 24, lw=0.6, color="#1e293b"):
        ax.plot([x0, x1], [y, y], color=color, lw=lw)

    # Title
    text(left, height - 26, f"SoccerGMR Ablation Study: DQ-CGP & SI-Bind Inter-layer Binding ({len(valid_backbones)} Backbones)", 13.5, "bold", color="#0f172a")
    sub = (
        "Comparing GMR Baseline vs. DQ-CGPv3 vs. GEQ Selection vs. SI-Bind (Inter-layer Residual Binding & Dominance Refresh) "
        "on Standard Test (1,036 queries) and Full-Decodable Test (2,970 queries)."
    )
    text(left, height - 46, sub, 9.0, color="#475569")

    top = height - 64
    rule(top, lw=1.2, color="#0f172a")

    # Split super-headers
    std_mid = std_left + (num_std_cols * col_step) / 2
    text(std_mid, top - 14, "Standard Test Split (1,036 queries)", 10.5, "bold", ha="center", color="#1e3a8a")
    rule(top - 24, std_left + 2, std_left + num_std_cols * col_step - 4, lw=0.9, color="#2563eb")

    full_mid = full_left + (num_full_cols * col_step) / 2
    text(full_mid, top - 14, "Full-Decodable Test Split (2,970 queries)", 10.5, "bold", ha="center", color="#065f46")
    rule(top - 24, full_left + 2, full_left + num_full_cols * col_step - 4, lw=0.9, color="#059669")

    # Column subheaders
    text(left, top - 38, "Backbone", 9.5, "bold", color="#0f172a")
    text(left + backbone_w, top - 38, "Configuration", 9.5, "bold", color="#0f172a")

    col_names = ["mAP (%)", "Orc-Gap (%)", "Rej-F1@.5 (%)", "AUROC (%)"]
    for i, cname in enumerate(col_names):
        text(std_left + (i + 0.5) * col_step, top - 38, cname, 8.8, "bold", ha="center", color="#1e293b")
        text(full_left + (i + 0.5) * col_step, top - 38, cname, 8.8, "bold", ha="center", color="#1e293b")

    rule(top - 48, lw=0.8, color="#0f172a")

    y = top - 64
    for b_idx, (b_name, display_name) in enumerate(valid_backbones):
        b_data = data[b_name]["splits"]

        for m_idx, (m_key, m_label, m_color, m_weight) in enumerate(METHODS):
            curr_y = y - m_idx * row_height

            # Background tints
            if m_key == "gmr_dqcgpv3_geq_g1":
                ax.add_patch(plt.Rectangle((left - 4, curr_y - 9), width - left - 16, 18, color="#ecfdf5", zorder=-1))
            elif m_key == "gmr_dqcgpv3_geq_map":
                ax.add_patch(plt.Rectangle((left - 4, curr_y - 9), width - left - 16, 18, color="#fffbeb", zorder=-1))
            elif m_key == "gmr_si_bind":
                ax.add_patch(plt.Rectangle((left - 4, curr_y - 9), width - left - 16, 18, color="#f5f3ff", zorder=-1))

            if m_idx == 0:
                text(left, curr_y, display_name, 9.8, "bold", color="#0f172a")

            text(left + backbone_w, curr_y, m_label, 9.0, m_weight, color=m_color)

            for split_i, (split_key, base_x) in enumerate([("standard", std_left), ("full", full_left)]):
                row_metrics = b_data[split_key].get(m_key)
                if not row_metrics:
                    text(base_x + 2.0 * col_step, curr_y, "-- running / pending --", size=8.2, style="italic", ha="center", color="#94a3b8")
                    continue

                vals = [
                    row_metrics.get("mAP"),
                    row_metrics.get("oracle_gap_pct"),
                    row_metrics.get("rej_f1_50"),
                    row_metrics.get("auroc"),
                ]

                # Best among all available in this backbone
                all_avail = [b_data[split_key][mk] for mk, _, _, _ in METHODS if b_data[split_key].get(mk)]

                metric_keys = ["mAP", "oracle_gap_pct", "rej_f1_50", "auroc"]
                for c_idx, val in enumerate(vals):
                    if val is None:
                        continue
                    all_vals = [m[metric_keys[c_idx]] for m in all_avail if m.get(metric_keys[c_idx]) is not None]

                    # For Orc-Gap (c_idx == 1), LOWER is better!
                    if c_idx == 1:
                        best_val = min(all_vals) if all_vals else val
                    else:
                        best_val = max(all_vals) if all_vals else val

                    is_best = (val == best_val) and (len(set(all_vals)) > 1)

                    v_str = f"{val:.2f}"
                    val_color = m_color if is_best else "#475569"
                    val_weight = "bold" if is_best else "normal"

                    text(base_x + (c_idx + 0.5) * col_step, curr_y, v_str, 9.0, val_weight, ha="center", color=val_color)

        y -= len(METHODS) * row_height
        if b_idx < len(valid_backbones) - 1:
            rule(y + 10, lw=0.45, color="#cbd5e1")
            y -= 6

    rule(y + 8, lw=1.2, color="#0f172a")

    notes = [
        "Method 3 (G1 Sel): Global Event Query fine-tuned with model selection on Val Mean G-mIoU@1 (0.4/0.5/0.6) (Official).",
        "Method 4 (mAP Sel): Global Event Query fine-tuned with model selection on highest Val Localization mAP.",
        "Method 5 (SI-Bind): Inter-layer Residual Binding with Dominance-Aware Slot Refresh and Slot-Private Innovation (Ours).",
        "Orc-Gap (%): Area between Oracle envelope and G-mIoU@1 threshold curve across tau in [0, 1] (Integral of Oracle_G1 minus G1-AUC).",
        "Note: Lower Orc-Gap is better (bold marks lowest gap per backbone), directly quantifying G-mIoU loss caused by imperfect query existence gating.",
    ]
    for i, line in enumerate(notes):
        text(left, y - 4 - i * 11.5, line, 8.0, color="#64748b")

    svg_path = OUT_DIR / f"{filename_stem}.svg"
    pdf_path = OUT_DIR / f"{filename_stem}.pdf"
    png_path = OUT_DIR / f"{filename_stem}.png"

    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated {filename_stem}: SVG, PDF, PNG")

    # Copy to docs/images and docs/tables
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
    DOCS_TABLES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_path, DOCS_IMAGES / f"ablation_table_stage2_selection_9_backbones.png")
    shutil.copy2(pdf_path, DOCS_TABLES / f"ABLATION_TABLE_GEQ_MAP_SELECTION.pdf")
    shutil.copy2(png_path, DOCS_TABLES / f"ABLATION_TABLE_GEQ_MAP_SELECTION.png")

    if ARTIFACT_DIR.is_dir():
        for ext in ["svg", "pdf", "png"]:
            src = OUT_DIR / f"{filename_stem}.{ext}"
            dst = ARTIFACT_DIR / f"{filename_stem}.{ext}"
            shutil.copy2(src, dst)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_table(ALL_BACKBONES, "ABLATION_TABLE_GEQ_MAP_SELECTION")


if __name__ == "__main__":
    main()
