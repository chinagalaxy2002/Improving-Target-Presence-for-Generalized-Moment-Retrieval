#!/usr/bin/env python3
"""Generate publication-quality vector table figure (SVG, PDF, PNG) comparing:
1. GMR Baseline
2. GMR + SI-Bind (E6)
3. GMR + SI-Bind (E6) + GEQ (Ours)

Across 9 temporal DETR backbones:
- Moment-DETR
- EaTR
- QD-DETR
- CG-DETR
- BM-DETR
- BAM-DETR
- Sim-DETR
- UVCOM
- FlashVTG

Evaluated on:
- Standard Test Split (1,036 queries)
- Full-Decodable Test Split (2,970 queries)

Metrics reported:
- mAP (%)            [Higher is better]
- Orc-Gap (%)        [Lower is better]
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
RESULTS_FILE = ROOT / "docs" / "tables" / "ablation_study_si_bind_results.json"
OUT_TABLES = ROOT / "docs" / "tables"
OUT_IMAGES = ROOT / "docs" / "images"
CURRENT_BRAIN = Path("/home/guoxiangyu/.gemini/antigravity-cli/brain/e32cc1be-0019-4e4d-9a30-81c95fb95359")

BACKBONES = [
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
    ("gmr", "1. GMR Baseline", "#334155", "normal", None),
    ("gmr_si_bind", "2. GMR + SI-Bind (Ours - E6)", "#6d28d9", "bold", "#f5f3ff"),
]


def render():
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
    })

    left = 24
    backbone_w = 95
    config_w = 230
    col_step = 72
    num_std_cols = 4  # mAP, Orc-Gap, Rej-F1@.5, AUROC
    num_full_cols = 4

    std_left = left + backbone_w + config_w
    full_left = std_left + num_std_cols * col_step + 18
    width = full_left + num_full_cols * col_step + 24

    row_height = 20
    n_bb = len(BACKBONES)
    table_content_h = n_bb * len(METHODS) * row_height + (n_bb - 1) * 6
    height = 115 + table_content_h + 85

    fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white", dpi=220)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")

    def text(x, y, s, size=9.5, weight="normal", ha="left", color="#1e293b", style="normal"):
        return ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style, ha=ha, va="center", color=color)

    def rule(y, x0=left, x1=width - 24, lw=0.6, color="#1e293b"):
        ax.plot([x0, x1], [y, y], color=color, lw=lw)

    # Title
    text(left, height - 26, "SoccerGMR Performance Benchmark: Baseline vs. SI-Bind (Ours - E6)", 13.0, "bold", color="#0f172a")
    sub = (
        "Head-to-head evaluation across 9 temporal DETR backbones on Standard Test (1,036 queries) "
        "and Full-Decodable Test (2,970 queries). Strict single seed (2023)."
    )
    text(left, height - 46, sub, 8.8, color="#475569")

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

    metric_keys = ["mAP", "oracle_gap_pct", "rej_f1_50", "auroc"]

    y = top - 62
    for b_idx, (b_name, display_name) in enumerate(BACKBONES):
        b_data = data.get(b_name, {}).get("splits", {})

        # Pre-compute best value per metric for this backbone across all available methods
        best_vals = {}
        for split_key in ["standard", "full"]:
            best_vals[split_key] = {}
            for c_idx, m_field in enumerate(metric_keys):
                candidate_vals = []
                for m_k, _, _, _, _ in METHODS:
                    m_row = b_data.get(split_key, {}).get(m_k)
                    if m_row and m_row.get(m_field) is not None:
                        candidate_vals.append((m_k, m_row[m_field]))
                if candidate_vals:
                    if c_idx == 1:  # Orc-Gap: min is best
                        opt_val = min(v for _, v in candidate_vals)
                    else:  # higher is best
                        opt_val = max(v for _, v in candidate_vals)
                    best_vals[split_key][m_field] = opt_val

        for m_idx, (m_key, m_label, m_color, m_weight, tint) in enumerate(METHODS):
            curr_y = y - m_idx * row_height

            # Highlight tint for GEQ row
            if tint:
                ax.add_patch(plt.Rectangle((left - 4, curr_y - 8), width - left - 16, 17, color=tint, zorder=-1))

            if m_idx == 0:
                text(left, curr_y, display_name, 9.8, "bold", color="#0f172a")

            text(left + backbone_w, curr_y, m_label, 8.8, m_weight, color=m_color)

            for split_i, (split_key, base_x) in enumerate([("standard", std_left), ("full", full_left)]):
                row_metrics = b_data.get(split_key, {}).get(m_key)
                if not row_metrics:
                    text(base_x + 2.0 * col_step, curr_y, "-- running / pending --", size=8.0, style="italic", ha="center", color="#94a3b8")
                    continue

                for c_idx, m_field in enumerate(metric_keys):
                    val = row_metrics.get(m_field)
                    if val is None:
                        continue

                    opt_val = best_vals.get(split_key, {}).get(m_field)
                    is_winner = (opt_val is not None and abs(val - opt_val) < 1e-4)

                    v_str = f"{val:.2f}"
                    val_color = m_color if is_winner else "#475569"
                    val_weight = "bold" if is_winner else "normal"

                    text(base_x + (c_idx + 0.5) * col_step, curr_y, v_str, 8.8, val_weight, ha="center", color=val_color)

        y -= len(METHODS) * row_height
        if b_idx < len(BACKBONES) - 1:
            rule(y + 8, lw=0.4, color="#cbd5e1")
            y -= 6

    rule(y + 6, lw=1.2, color="#0f172a")

    notes = [
        "Method 1 (GMR Baseline): Official temporal DETR baseline checkpoints evaluated with native bipartite matching.",
        "Method 2 (GMR + SI-Bind [Ours - E6]): Full Structural Triad (Temporal Residual Stream + Slot-Private Innovation + Dominance-Aware Refresh).",
        "Orc-Gap (%): Area between Oracle envelope and G-mIoU@1 threshold curve across tau in [0, 1] (Integral of Oracle_G1 minus G1-AUC). Lower is better.",
        "Bold values highlight the optimal performance between Baseline and SI-Bind (E6) per backbone. FlashVTG E6 currently in queue.",
    ]
    for i, line in enumerate(notes):
        text(left, y - 6 - i * 11.0, line, 7.8, color="#64748b")

    out_stem = "baseline_vs_si_bind_e6"
    svg_p = OUT_TABLES / f"{out_stem}.svg"
    pdf_p = OUT_TABLES / f"{out_stem}.pdf"
    png_p = OUT_TABLES / f"{out_stem}.png"

    fig.savefig(svg_p, format="svg", bbox_inches="tight")
    fig.savefig(pdf_p, format="pdf", bbox_inches="tight")
    fig.savefig(png_p, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Generated {out_stem}: SVG, PDF, PNG in {OUT_TABLES}")

    # Copy to docs/images
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png_p, OUT_IMAGES / f"{out_stem}.png")

    # Copy to current conversation artifact directory
    if CURRENT_BRAIN.exists():
        shutil.copy2(pdf_p, CURRENT_BRAIN / f"{out_stem}.pdf")
        shutil.copy2(png_p, CURRENT_BRAIN / f"{out_stem}.png")
        shutil.copy2(svg_p, CURRENT_BRAIN / f"{out_stem}.svg")

    print(f"Copied {out_stem} to images and brain directory!")


if __name__ == "__main__":
    render()
