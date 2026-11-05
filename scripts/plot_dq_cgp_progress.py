#!/usr/bin/env python3
"""Generate comprehensive dual-stage progress monitoring charts for SoccerGMR with DQ-CGPv3.

Visualizes side-by-side for all 8 backbones:
1. Stage 1: Native GMR + DQ-CGPv3 learning curves (Val mAP vs baseline).
2. Stage 2: GEQ-Full fine-tuning convergence (Val mean G-mIoU@1 vs GEQ baseline, plus mAP).
Saves PNG to runs/dq_cgp_v3/progress_monitor.png and artifacts dir.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# (id, Display Name, Stage 1 Base Val mAP, Stage 2 Base Val Mean G1)
BACKBONES = [
    ("moment_detr", "Moment-DETR", 7.35, 48.91),
    ("eatr", "EaTR", 19.34, 51.52),
    ("qd_detr", "QD-DETR", 17.51, 51.57),
    ("cg_detr", "CG-DETR", 20.98, 49.33),
    ("sim_detr", "Sim-DETR", 22.84, 51.98),
    ("bm_detr", "BM-DETR", 3.61, 49.14),
    ("bam_detr", "BAM-DETR", 25.49, 49.01),
    ("uvcom", "UVCOM", 8.77, 50.28),
]

ARTIFACT_DIR = Path("/home/guoxiangyu/.gemini/antigravity-cli/brain/62a59d6f-74ff-44d8-a47a-d538b53ca17b")


def load_history(path):
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "runs" / "dq_cgp_v3"))
    parser.add_argument("--output", default=str(ROOT / "runs" / "dq_cgp_v3" / "progress_monitor.png"))
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    queue_state_path = root / "queue_state.json"
    queue_state = {}
    if queue_state_path.is_file():
        try:
            queue_state = json.loads(queue_state_path.read_text())
        except Exception:
            pass

    running_jobs = queue_state.get("running", {})
    completed_jobs = set(queue_state.get("completed", []))

    # Create figure: 4 rows x 4 cols (each row contains 2 backbones, each with Stage 1 & Stage 2)
    fig, axes = plt.subplots(4, 4, figsize=(24, 18), dpi=140)
    fig.patch.set_facecolor("#0f172a")  # Dark elegant background

    fig.suptitle(
        "SoccerGMR with DQ-CGPv3 & GEQ-Full | 8 Backbones Dual-Stage Training Convergence Monitor",
        fontsize=20,
        fontweight="bold",
        color="#f8fafc",
        y=0.988,
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subtitle = (
        f"Last Updated: {now_str}  |  Dual RTX 3090 (3 jobs/GPU = 6 concurrent)  |  "
        f"Stage 1: GMR Val mAP  |  Stage 2: GEQ-Full Val Mean G-mIoU@1"
    )
    fig.text(0.5, 0.968, subtitle, ha="center", fontsize=11, color="#94a3b8")

    # Arrange 8 backbones into 4 rows x 2 pairs
    for i, (name, display_name, baseline_gmr, baseline_geq) in enumerate(BACKBONES):
        row = i // 2
        col_pair = (i % 2) * 2  # Col 0 or 2 for Stage 1, Col 1 or 3 for Stage 2

        ax_s1 = axes[row, col_pair]
        ax_s2 = axes[row, col_pair + 1]

        for ax in (ax_s1, ax_s2):
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#94a3b8", labelsize=8.5)
            ax.grid(True, color="#334155", linestyle=":", linewidth=0.8, alpha=0.7)
            for spine in ax.spines.values():
                spine.set_color("#334155")
                spine.set_linewidth(1.0)

        # -----------------------------------------------------------------
        # Panel 1: Stage 1 GMR
        # -----------------------------------------------------------------
        s1_hist_path = root / name / "gmr" / "history.jsonl"
        s1_records = load_history(s1_hist_path)
        is_s1_running = f"{name}/gmr" in running_jobs
        s1_gpu = running_jobs.get(f"{name}/gmr", {}).get("gpu", "?") if is_s1_running else None
        is_s1_done = f"{name}/gmr" in completed_jobs

        ax_s1.axhline(
            y=baseline_gmr,
            color="#fbbf24",
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label=f"Base GMR: {baseline_gmr:.2f}%",
        )

        s1_scores = [r.get("selection_score", 0.0) for r in s1_records]
        s1_epochs = [r.get("epoch", idx + 1) for idx, r in enumerate(s1_records)]
        s1_improved = [r.get("improved", False) for r in s1_records]

        if s1_records:
            ax_s1.plot(
                s1_epochs,
                s1_scores,
                color="#38bdf8",
                linewidth=2.0,
                label=f"DQ-CGPv3 ({s1_scores[-1]:.2f}%)",
                zorder=3,
            )
            imp_e = [e for e, imp in zip(s1_epochs, s1_improved) if imp]
            imp_s = [s for s, imp in zip(s1_scores, s1_improved) if imp]
            if imp_e:
                ax_s1.scatter(imp_e, imp_s, color="#34d399", s=30, zorder=4, label="Improved")

            best_idx = np.argmax(s1_scores)
            best_e = s1_epochs[best_idx]
            best_s = s1_scores[best_idx]
            delta_s1 = best_s - baseline_gmr

            ax_s1.scatter(
                [best_e],
                [best_s],
                color="#f43f5e",
                s=110,
                marker="*",
                zorder=5,
                edgecolors="#ffe4e6",
                linewidth=1.2,
                label=f"Best: {best_s:.2f}% (ep {best_e})",
            )

            status_text = (
                f"TRAINING (GPU {s1_gpu})"
                if is_s1_running
                else "COMPLETED"
                if is_s1_done
                else "DONE"
            )
            badge_color = "#38bdf8" if is_s1_running else "#10b981" if is_s1_done else "#94a3b8"

            dsign = "+" if delta_s1 >= 0 else ""
            dcol = "#34d399" if delta_s1 >= 0 else "#f87171"
            info_str = f"Best: {best_s:.2f}% ({dsign}{delta_s1:.2f}%) | Ep: {s1_epochs[-1]}"
            ax_s1.text(
                0.03,
                0.90,
                info_str,
                transform=ax_s1.transAxes,
                fontsize=9.5,
                fontweight="bold",
                color=dcol,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", edgecolor=dcol, alpha=0.9),
            )
            y_max1 = max(baseline_gmr + 3.0, max(s1_scores) + 3.0)
            y_min1 = max(0.0, min(s1_scores) - 1.0)
            ax_s1.set_ylim(y_min1, y_max1)
        else:
            status_text = "QUEUED"
            badge_color = "#64748b"
            ax_s1.text(
                0.5,
                0.5,
                "Queued in Background Pipeline",
                ha="center",
                va="center",
                transform=ax_s1.transAxes,
                fontsize=11,
                color="#64748b",
            )
            ax_s1.set_ylim(0, baseline_gmr + 5.0)

        ax_s1.set_title(
            f"{display_name} | Stage 1 (GMR) [{status_text}]",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            color=badge_color,
            pad=8,
        )
        ax_s1.set_xlabel("Epoch", fontsize=9, color="#94a3b8")
        ax_s1.set_ylabel("Standard Val mAP (%)", fontsize=9, color="#94a3b8")
        ax_s1.legend(loc="upper right", fontsize=7.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", framealpha=0.9)

        # -----------------------------------------------------------------
        # Panel 2: Stage 2 GEQ-Full
        # -----------------------------------------------------------------
        s2_hist_path = root / name / "geq_full" / "history.jsonl"
        s2_records = load_history(s2_hist_path)
        is_s2_running = f"{name}/geq_full" in running_jobs
        s2_gpu = running_jobs.get(f"{name}/geq_full", {}).get("gpu", "?") if is_s2_running else None
        is_s2_done = f"{name}/geq_full" in completed_jobs

        ax_s2.axhline(
            y=baseline_geq,
            color="#fbbf24",
            linestyle="--",
            linewidth=1.6,
            alpha=0.85,
            label=f"Base GEQ: {baseline_geq:.2f}%",
        )

        s2_scores = [r.get("selection_score", 0.0) for r in s2_records]
        s2_epochs = [r.get("epoch", idx + 1) for idx, r in enumerate(s2_records)]
        s2_improved = [r.get("improved", False) for r in s2_records]
        s2_maps = [r.get("metrics", {}).get("brief", {}).get("mAP", 0.0) for r in s2_records]

        if s2_records:
            ax_s2.plot(
                s2_epochs,
                s2_scores,
                color="#10b981",
                linewidth=2.0,
                label=f"Val G1 Mean ({s2_scores[-1]:.2f}%)",
                zorder=3,
            )
            # Secondary line: Val mAP during GEQ-Full
            if any(s2_maps):
                ax_s2.plot(
                    s2_epochs,
                    s2_maps,
                    color="#818cf8",
                    linestyle=":",
                    linewidth=1.5,
                    label=f"Val mAP ({s2_maps[-1]:.2f}%)",
                    zorder=3,
                )

            imp_e2 = [e for e, imp in zip(s2_epochs, s2_improved) if imp]
            imp_s2 = [s for s, imp in zip(s2_scores, s2_improved) if imp]
            if imp_e2:
                ax_s2.scatter(imp_e2, imp_s2, color="#6ee7b7", s=30, zorder=4, label="G1 Improved")

            best_idx2 = np.argmax(s2_scores)
            best_e2 = s2_epochs[best_idx2]
            best_s2 = s2_scores[best_idx2]
            delta_s2 = best_s2 - baseline_geq

            ax_s2.scatter(
                [best_e2],
                [best_s2],
                color="#f59e0b",
                s=110,
                marker="*",
                zorder=5,
                edgecolors="#fef3c7",
                linewidth=1.2,
                label=f"Best G1: {best_s2:.2f}% (ep {best_e2})",
            )

            status_text2 = (
                f"GEQ TUNING (GPU {s2_gpu})"
                if is_s2_running
                else "COMPLETED"
                if is_s2_done
                else "FINISHED"
            )
            badge_color2 = "#10b981" if is_s2_running else "#34d399" if is_s2_done else "#94a3b8"

            dsign2 = "+" if delta_s2 >= 0 else ""
            dcol2 = "#34d399" if delta_s2 >= 0 else "#f87171"
            info_str2 = f"Best G1: {best_s2:.2f}% ({dsign2}{delta_s2:.2f}%) | Ep: {s2_epochs[-1]}"
            ax_s2.text(
                0.03,
                0.90,
                info_str2,
                transform=ax_s2.transAxes,
                fontsize=9.5,
                fontweight="bold",
                color=dcol2,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f172a", edgecolor=dcol2, alpha=0.9),
            )
            y_max2 = max(baseline_geq + 4.0, max(s2_scores) + 3.0)
            y_min2 = max(0.0, min(min(s2_scores), min(s2_maps) if any(s2_maps) else min(s2_scores)) - 3.0)
            ax_s2.set_ylim(y_min2, y_max2)
        else:
            status_text2 = "QUEUED / WAITING"
            badge_color2 = "#64748b"
            ax_s2.text(
                0.5,
                0.5,
                "Queued for GEQ-Full Fine-Tuning\n(Starts automatically after Stage 1)",
                ha="center",
                va="center",
                transform=ax_s2.transAxes,
                fontsize=10.5,
                color="#64748b",
            )
            ax_s2.set_ylim(20.0, baseline_geq + 8.0)

        ax_s2.set_title(
            f"{display_name} | Stage 2 (GEQ-Full) [{status_text2}]",
            loc="left",
            fontsize=10.5,
            fontweight="bold",
            color=badge_color2,
            pad=8,
        )
        ax_s2.set_xlabel("Epoch", fontsize=9, color="#94a3b8")
        ax_s2.set_ylabel("Standard Val Mean G-mIoU@1 (%)", fontsize=9, color="#94a3b8")
        ax_s2.legend(loc="lower right", fontsize=7.5, facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0", framealpha=0.9)

    plt.tight_layout(rect=[0.015, 0.015, 0.985, 0.965])
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"Saved dual-stage progress monitor plot to {out_path}")

    # Copy to brain artifact directory for display
    if ARTIFACT_DIR.is_dir():
        artifact_png = ARTIFACT_DIR / "progress_monitor.png"
        shutil.copy2(out_path, artifact_png)
        print(f"Copied to artifact dir: {artifact_png}")


if __name__ == "__main__":
    main()
