#!/usr/bin/env python3
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = ROOT / "docs" / "tables" / "ablation_variants_detailed_results.json"
OUT_DIR = ROOT / "docs" / "tables"
OUT_IMAGES = ROOT / "docs" / "images"
BRAIN_DIR = Path("/home/guoxiangyu/.gemini/antigravity-cli/brain/62a59d6f-74ff-44d8-a47a-d538b53ca17b")
CURRENT_BRAIN = Path("/home/guoxiangyu/.gemini/antigravity-cli/brain/e32cc1be-0019-4e4d-9a30-81c95fb95359")

BACKBONES = [
    ("moment_detr", "Moment-DETR", ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]),
    ("bm_detr", "BM-DETR", ["E2", "E3", "E4", "E5", "E6", "E7"]),
    ("uvcom", "UVCOM", ["E1", "E2", "E4", "E5", "E6"]),
    ("cg_detr", "CG-DETR", ["E2", "E5", "E6"]),
    ("qd_detr", "QD-DETR", ["E6"]),
]

VAR_INFO = {
    "E1": ("1. E1: Binding-only (Interlayer Anchor)", "#334155", "normal", None),
    "E2": ("2. E2: Lite Binding + Residual Stream", "#0369a1", "normal", None),
    "E3": ("3. E3: E2 + Slot-Private Innovation (B)", "#047857", "bold", "#ecfdf5"),
    "E4": ("4. E4: E2 + Uniform Random Refresh (C Control)", "#64748b", "normal", None),
    "E5": ("5. E5: E2 + Dominance-Aware Refresh (C)", "#b45309", "bold", "#fffbeb"),
    "E6": ("6. E6: E2 + Innovation (B) + Dominance (C)", "#6d28d9", "bold", "#f5f3ff"),
    "E7": ("7. E7: E6 - Binding (bind_coef=0 Ablation)", "#475569", "normal", None),
}

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
    backbone_w = 100
    config_w = 260
    col_step = 72
    num_std_cols = 4  # mAP, Orc-Gap, Rej-F1@.5, AUROC
    num_full_cols = 4

    std_left = left + backbone_w + config_w
    full_left = std_left + num_std_cols * col_step + 18
    width = full_left + num_full_cols * col_step + 24

    row_height = 23
    total_rows = sum(len(variants) for _, _, variants in BACKBONES)
    n_bb = len(BACKBONES)
    table_content_h = total_rows * row_height + (n_bb - 1) * 8
    height = 110 + table_content_h + 80

    fig = plt.figure(figsize=(width / 72, height / 72), facecolor="white", dpi=220)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.axis("off")

    def text(x, y, s, size=9.5, weight="normal", ha="left", color="#1e293b", style="normal"):
        return ax.text(x, y, s, fontsize=size, fontweight=weight, fontstyle=style, ha=ha, va="center", color=color)

    def rule(y, x0=left, x1=width - 24, lw=0.6, color="#1e293b"):
        ax.plot([x0, x1], [y, y], color=color, lw=lw)

    # Title
    text(left, height - 26, "SoccerGMR SI-Bind: Detailed Component Ablation Study (E1 to E7)", 13.5, "bold", color="#0f172a")
    sub = (
        "Fine-grained ablation across mechanisms: Mechanism A (Binding & Residual), Mechanism B (Slot Innovation), "
        "and Mechanism C (Dominance Refresh) on Standard Test (1,036 queries) and Full Test (2,970 queries)."
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
    text(left + backbone_w, top - 38, "Ablation Method / Configuration", 9.5, "bold", color="#0f172a")

    col_names = ["mAP (%)", "Orc-Gap (%)", "Rej-F1@.5 (%)", "AUROC (%)"]
    for i, cname in enumerate(col_names):
        text(std_left + (i + 0.5) * col_step, top - 38, cname, 8.8, "bold", ha="center", color="#1e293b")
        text(full_left + (i + 0.5) * col_step, top - 38, cname, 8.8, "bold", ha="center", color="#1e293b")

    rule(top - 48, lw=0.8, color="#0f172a")

    y = top - 64
    for b_idx, (b_name, display_name, variants) in enumerate(BACKBONES):
        # Find best values within this backbone for bolding
        all_avail_std = [data[f"{b_name}_{v}"]["standard"] for v in variants if f"{b_name}_{v}" in data]
        all_avail_full = [data[f"{b_name}_{v}"]["full"] for v in variants if f"{b_name}_{v}" in data]

        for m_idx, v in enumerate(variants):
            key = f"{b_name}_{v}"
            row_data = data.get(key, {})
            label, color, weight, tint = VAR_INFO[v]

            curr_y = y - m_idx * row_height

            # Background tint
            if tint:
                ax.add_patch(plt.Rectangle((left - 4, curr_y - 9), width - left - 16, 18, color=tint, zorder=-1))

            if m_idx == 0:
                text(left, curr_y, display_name, 9.8, "bold", color="#0f172a")

            text(left + backbone_w, curr_y, label, 8.8, weight, color=color)

            for split_i, (split_key, base_x, avail_list) in enumerate([
                ("standard", std_left, all_avail_std),
                ("full", full_left, all_avail_full)
            ]):
                split_res = row_data.get(split_key)
                if not split_res:
                    text(base_x + 2.0 * col_step, curr_y, "--", size=8.5, ha="center", color="#94a3b8")
                    continue

                vals = [
                    split_res.get("mAP"),
                    split_res.get("oracle_gap_pct"),
                    split_res.get("rej_f1_50"),
                    split_res.get("auroc"),
                ]

                metric_keys = ["mAP", "oracle_gap_pct", "rej_f1_50", "auroc"]
                for c_idx, val in enumerate(vals):
                    if val is None:
                        continue
                    all_metric_vals = [m[metric_keys[c_idx]] for m in avail_list if m.get(metric_keys[c_idx]) is not None]
                    
                    if c_idx == 1: # Orc-Gap: lower is better
                        best_val = min(all_metric_vals) if all_metric_vals else val
                    else:
                        best_val = max(all_metric_vals) if all_metric_vals else val

                    is_best = (val == best_val) and (len(set(all_metric_vals)) > 1)
                    val_str = f"{val:.2f}"
                    c_x = base_x + (c_idx + 0.5) * col_step

                    if is_best:
                        text(c_x, curr_y, val_str, 9.0, "bold", ha="center", color="#1e1b4b")
                    else:
                        text(c_x, curr_y, val_str, 8.8, "normal", ha="center", color="#334155")

        # Dividing rule between backbones
        y = y - len(variants) * row_height - 8
        if b_idx < n_bb - 1:
            rule(y + 4, lw=0.4, color="#cbd5e1")

    # Footer note
    footer_y = 28
    rule(footer_y + 16, lw=0.8, color="#0f172a")
    f_sub = (
        "Note: E1 = Binding only (identity across layers); E2 = Lite Binding + Residual; E3 = E2 + Slot-Private Innovation (B); "
        "E4 = E2 + Uniform Random Refresh (C Control); E5 = E2 + Dominance Refresh (C); E6 = E2+B+C (Champion Triad); E7 = E6 - Binding."
    )
    text(left, footer_y, f_sub, 8.2, color="#64748b")
    text(left, footer_y - 14, "Bold numbers indicate best performance among ablation variants within the same backbone. Oracle Gap: lower is better.", 8.2, color="#64748b")

    out_stem = "SI_BIND_ABLATION_TABLE_E1_E7"
    pdf_p = OUT_DIR / f"{out_stem}.pdf"
    png_p = OUT_DIR / f"{out_stem}.png"
    svg_p = OUT_DIR / f"{out_stem}.svg"

    fig.savefig(svg_p, format="svg", bbox_inches="tight")
    fig.savefig(pdf_p, format="pdf", bbox_inches="tight")
    fig.savefig(png_p, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Copy to images and brain directories
    shutil.copy2(png_p, OUT_IMAGES / f"{out_stem}.png")
    if BRAIN_DIR.exists():
        shutil.copy2(pdf_p, BRAIN_DIR / f"{out_stem}.pdf")
        shutil.copy2(png_p, BRAIN_DIR / f"{out_stem}.png")
        shutil.copy2(svg_p, BRAIN_DIR / f"{out_stem}.svg")
    if CURRENT_BRAIN.exists():
        shutil.copy2(pdf_p, CURRENT_BRAIN / f"{out_stem}.pdf")
        shutil.copy2(png_p, CURRENT_BRAIN / f"{out_stem}.png")
        shutil.copy2(svg_p, CURRENT_BRAIN / f"{out_stem}.svg")

    print(f"Generated {out_stem}: PDF, PNG, SVG in {OUT_DIR}, {OUT_IMAGES}, and brain directories!")

if __name__ == "__main__":
    render()
