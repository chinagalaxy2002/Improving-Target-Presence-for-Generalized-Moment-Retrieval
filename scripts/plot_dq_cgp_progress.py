#!/usr/bin/env python3
"""Generate comprehensive progress monitoring charts for SoccerGMR with DQ-CGPv3.

Visualizes:
1. Val mAP learning curves vs baseline across all 8 backbones.
2. Current status, best epoch, and delta improvement.
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
BACKBONES = [
    ("moment_detr", "Moment-DETR", 7.35),
    ("eatr", "EaTR", 19.34),
    ("qd_detr", "QD-DETR", 17.51),
    ("cg_detr", "CG-DETR", 8.65),
    ("sim_detr", "Sim-DETR", 22.84),
    ("bm_detr", "BM-DETR", 3.61),
    ("bam_detr", "BAM-DETR", 25.49),
    ("uvcom", "UVCOM", 8.77),
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

    # Create figure: 4 rows x 2 cols
    fig, axes = plt.subplots(4, 2, figsize=(16, 18), dpi=150)
    fig.patch.set_facecolor("#0f172a")  # Dark elegant background

    fig.suptitle(
        "SoccerGMR with DQ-CGPv3 | 8 Backbones Real-time Training Progress",
        fontsize=20,
        fontweight="bold",
        color="#f8fafc",
        y=0.985,
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    subtitle = f"Last Updated: {now_str}  |  Dual RTX 3090 (3 jobs/card = 6 concurrent)  |  Standard Split Val mAP"
    fig.text(0.5, 0.965, subtitle, ha="center", fontsize=11, color="#94a3b8")

    axes_flat = axes.flatten()

    for idx, (name, display_name, baseline_val) in enumerate(BACKBONES):
        ax = axes_flat[idx]
        ax.set_facecolor("#1e293b")

        history_path = root / name / "gmr" / "history.jsonl"
        complete_path = root / name / "gmr" / "complete.json"

        records = load_history(history_path)
        is_running = f"{name}/gmr" in running_jobs
        gpu_id = running_jobs.get(f"{name}/gmr", {}).get("gpu", "?") if is_running else None
        is_complete = f"{name}/gmr" in completed_jobs

        epochs = [r.get("epoch", i + 1) for i, r in enumerate(records)]
        scores = [r.get("selection_score", 0.0) for r in records]
        improved = [r.get("improved", False) for r in records]

        # Draw baseline line
        ax.axhline(
            y=baseline_val,
            color="#fbbf24",
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label=f"Baseline GMR: {baseline_val:.2f}%",
        )

        if epochs:
            # Curve line
            line_color = "#38bdf8"
            ax.plot(
                epochs,
                scores,
                color=line_color,
                linewidth=2.2,
                label=f"DQ-CGPv3 (curr: {scores[-1]:.2f}%)",
                zorder=3,
            )

            # Highlight improved points
            imp_epochs = [e for e, imp in zip(epochs, improved) if imp]
            imp_scores = [s for s, imp in zip(scores, improved) if imp]
            if imp_epochs:
                ax.scatter(
                    imp_epochs,
                    imp_scores,
                    color="#34d399",
                    s=45,
                    zorder=4,
                    edgecolors="#064e3b",
                    linewidth=1.2,
                    label="Val Improved",
                )

            # Mark best point
            best_idx = np.argmax(scores)
            best_epoch = epochs[best_idx]
            best_score = scores[best_idx]
            delta = best_score - baseline_val

            ax.scatter(
                [best_epoch],
                [best_score],
                color="#f43f5e",
                s=130,
                marker="*",
                zorder=5,
                edgecolors="#ffe4e6",
                linewidth=1.5,
                label=f"Best: {best_score:.2f}% (ep {best_epoch})",
            )

            # Status Badge text
            status_text = (
                f"TRAINING (GPU {gpu_id})"
                if is_running
                else "COMPLETED"
                if is_complete
                else "PAUSED"
            )
            badge_color = "#38bdf8" if is_running else "#10b981" if is_complete else "#94a3b8"

            delta_sign = "+" if delta >= 0 else ""
            delta_color = "#34d399" if delta >= 0 else "#f87171"

            info_str = f"Best: {best_score:.2f}% ({delta_sign}{delta:.2f}%) | Ep: {epochs[-1]}"
            ax.text(
                0.03,
                0.90,
                info_str,
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                color=delta_color,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f172a", edgecolor=delta_color, alpha=0.9),
            )
        else:
            status_text = "PENDING / QUEUED"
            badge_color = "#64748b"
            ax.text(
                0.5,
                0.5,
                "Queued in Background Pipeline\n(Waiting for worker slot)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="#64748b",
            )

        # Title & Subhead
        ax.set_title(
            f"{display_name}  [{status_text}]",
            loc="left",
            fontsize=13,
            fontweight="bold",
            color=badge_color,
            pad=10,
        )

        ax.set_xlabel("Epoch", fontsize=10, color="#94a3b8")
        ax.set_ylabel("Standard Val mAP (%)", fontsize=10, color="#94a3b8")
        ax.tick_params(colors="#94a3b8", labelsize=9)

        # Styling grid & spines
        ax.grid(True, color="#334155", linestyle=":", linewidth=0.8, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.0)

        # Set reasonable y limits
        y_max = max(baseline_val + 4.0, max(scores) + 3.0 if scores else baseline_val + 5.0)
        y_min = max(0.0, min(scores) - 1.0 if scores else 0.0)
        ax.set_ylim(y_min, y_max)

        ax.legend(
            loc="upper right",
            fontsize=8.5,
            facecolor="#0f172a",
            edgecolor="#334155",
            labelcolor="#e2e8f0",
            framealpha=0.9,
        )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"Saved progress monitor plot to {out_path}")

    # Copy to brain artifact directory for display
    if ARTIFACT_DIR.is_dir():
        artifact_png = ARTIFACT_DIR / "progress_monitor.png"
        shutil.copy2(out_path, artifact_png)
        print(f"Copied to artifact dir: {artifact_png}")


if __name__ == "__main__":
    main()
