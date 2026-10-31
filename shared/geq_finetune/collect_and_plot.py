#!/usr/bin/env python3
"""Report all eight matched GEQ comparisons, with full-domain G-mIoU curves."""
from __future__ import annotations

import csv
import fcntl
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import BACKBONES, FIGURES, OUTPUT, VARIANTS, dump, read

TITLES = {"moment_detr": "Moment-DETR-GMR", "eatr": "EaTR-GMR", "qd_detr": "QD-DETR-GMR", "cg_detr": "CG-DETR-GMR"}


def sweep(path):
    with Path(path).open(newline="") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 101 or any(abs(float(row["threshold"]) - i / 100) > 1e-8 for i, row in enumerate(records)):
        raise RuntimeError(f"Incomplete full-domain sweep: {path}")
    values = np.asarray([float(row["G-mIoU@3"]) for row in records])
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise RuntimeError("Invalid 0–1 G-mIoU values")
    return np.arange(101) / 100, values


def save(fig, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        temporary = destination.with_name(destination.name + ".tmp." + suffix)
        fig.savefig(temporary, bbox_inches="tight", dpi=200, facecolor="white")
        if suffix == "svg":
            temporary.write_text("\n".join(line.rstrip() for line in temporary.read_text().splitlines()) + "\n")
        temporary.replace(destination.with_suffix("." + suffix))
    plt.close(fig)


def comparison_heatmap(records):
    matrix = np.full((8, 2), np.nan)
    for index, record in enumerate(records):
        if record["status"] == "COMPLETED":
            matrix[index] = record["before_gmiou"], record["after_gmiou"]
    normalized = np.full_like(matrix, np.nan)
    for index, row in enumerate(matrix):
        if np.isfinite(row).all():
            normalized[index] = (row - row.min()) / (row.max() - row.min()) if row.max() > row.min() else 0.5
    figure, ax = plt.subplots(figsize=(6.4, 5.5), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#EEEEEE")
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i in range(8):
        for j in range(2):
            value = matrix[i, j]
            ax.text(j, i, "pending" if np.isnan(value) else f"{value:.4f}",
                    ha="center", va="center", fontsize=11,
                    color="black" if np.isnan(value) or normalized[i, j] > 0.55 else "white")
    ax.set_xticks((0, 1), ("Original best", "+ GEQ"))
    ax.set_yticks(range(8), [f"{TITLES[r['backbone']]} · {r['variant']}" for r in records], fontsize=10)
    ax.set_title("G-mIoU@3 at threshold 0.4 (0–1)", fontsize=12)
    colorbar = figure.colorbar(image, ax=ax, ticks=(0, 0.5, 1), pad=0.04, shrink=0.8)
    colorbar.ax.set_yticklabels(("Low", "Equal", "High"))
    colorbar.set_label("Within-row relative score", fontsize=10)
    figure.supxlabel("Colors compare only the two values within each row.", fontsize=10)
    save(figure, FIGURES / "cross_backbone_overview/gmiou_at_0_4_before_after")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False})
    FIGURES.mkdir(parents=True, exist_ok=True)
    queue = read(OUTPUT / "QUEUE.json")
    records = []
    threshold_records = []
    lines = ["# Shared / Multi + Global Event Query 微调", "",
        f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}", "",
        "四个骨干各运行 shared、multi 两组，仅新增 GEQ。冻结原模型，固定微调 4 epoch；seed 2023，AdamW lr=3e-5、wd=1e-4，梯度裁剪 0.1；每任务 num_workers=2，每 GPU 最多两个 CUDA 任务（包含原有任务）。Moment-DETR batch=8，其余骨干 batch=32，均不累积。", "",
        "范式参考 main 的 Global-only：39ab0eb74427709851476b2bb86b698c8a3f4186。此轮沿用新路线 seed 2023，参考实验使用 seed 2026。无 LoRA、RoI、边界修正、候选融合或合成数据。固定最终 epoch 4，不使用 Test 选模。", "",
        "Test 对照使用原有 best 的冻结候选文件；GEQ 遍历完整 Test 特征后只替换存在性分数。每个样本的时间窗口、候选分数和顺序逐项保持一致。mAP/mR/mR+ 应完全相同；变化来自存在性判断。G-mIoU 曲线的阈值横轴和指标纵轴均为 0–1，扫描 101 点，阈值 0.4/0.5/0.6 预先固定。", "",
        "| Backbone | 方法 | 状态 | mAP 前/后 | AUROC 前/后 (%) | Rej-F1 前/后 (%) | G-mIoU@3 前 | 后 | Δ (0–1) |",
        "|---|---|---|---|---|---|---:|---:|---:|"]
    overview, overview_axes = plt.subplots(4, 2, figsize=(11, 13), layout="constrained")
    for bone_index, backbone in enumerate(BACKBONES):
        folder = FIGURES / backbone
        folder.mkdir(parents=True, exist_ok=True)
        backbone_records = []
        backbone_sweeps = []
        figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
        for index, variant in enumerate(VARIANTS):
            root = OUTPUT / backbone / variant
            complete = read(root / "COMPLETE.json")
            job = queue.get("jobs", {}).get(f"{backbone}/{variant}", {})
            status = complete.get("status", job.get("status", read(root / "status.json").get("status", "QUEUED")))
            ax = axes[index]
            overview_ax = overview_axes[bone_index, index]
            for panel in (ax, overview_ax):
                panel.set(xlim=(0, 1), ylim=(0, 1), xlabel="Existence threshold", ylabel="G-mIoU@3 (0–1)",
                          title=f"{TITLES[backbone]} · {variant}")
                panel.grid(alpha=0.2)
                panel.axvline(0.4, color="#999999", linewidth=0.8, linestyle=":")
            if status != "COMPLETED":
                for panel in (ax, overview_ax):
                    panel.text(0.5, 0.5, status, ha="center", va="center", transform=panel.transAxes, color="#666666")
                lines.append(f"| {TITLES[backbone]} | {variant} | {status} | — | — | — | — | — | — |")
                records.append(dict(backbone=backbone, variant=variant, status=status))
                backbone_records.append(records[-1])
                continue
            before, after = complete["before"], complete["after"]
            full_before = read(root / "test_before_metrics.json")
            full_after = read(root / "test_after_metrics.json")
            for threshold in ("0.4", "0.5", "0.6"):
                b_metric, a_metric = full_before[threshold]["brief"], full_after[threshold]["brief"]
                threshold_records.append(dict(backbone=backbone, variant=variant,
                    threshold=float(threshold), before_gmiou=b_metric["G-mIoU@3"] / 100,
                    after_gmiou=a_metric["G-mIoU@3"] / 100,
                    delta_gmiou=(a_metric["G-mIoU@3"] - b_metric["G-mIoU@3"]) / 100,
                    before_rejection_f1=b_metric[f"Rej-F1@{threshold}"],
                    after_rejection_f1=a_metric[f"Rej-F1@{threshold}"]))
            stats = full_before["0.4"]["stats"]
            reject_all = stats["num_negative"] / stats["num_total"]
            for panel in (ax, overview_ax):
                panel.axhline(reject_all, color="#888888", linestyle=(0, (1, 3)),
                              linewidth=1, label=f"Reject all ({reject_all:.3f})")
            for key in ("mAP", "mR@1", "mR@5", "mR+@5"):
                if before[key] != after[key]:
                    raise RuntimeError(f"Unexpected localization change: {backbone}/{variant}/{key}")
            for phase, label, color, style, marker in (
                ("before", "Original best", "#0072B2", "--", "o"),
                ("after", "+ Global Event Query", "#D55E00", "-", "s"),
            ):
                x, y = sweep(root / f"test_{phase}_sweep.csv")
                backbone_sweeps.extend(dict(variant=variant, phase=phase, threshold=float(t),
                    gmiou_3_0_1=float(v)) for t, v in zip(x, y))
                for panel in (ax, overview_ax):
                    panel.plot(x, y, label=label, color=color, linestyle=style, linewidth=1.8,
                               marker=marker, markevery=10, markersize=3.8)
                    panel.legend(loc="upper left", fontsize=9)
            b, a = before["G-mIoU@3"] / 100, after["G-mIoU@3"] / 100
            lines.append(f"| {TITLES[backbone]} | {variant} | COMPLETED | {before['mAP']:.2f}/{after['mAP']:.2f} | {before['AUROC']:.2f}/{after['AUROC']:.2f} | {before['Rej-F1@0.4']:.2f}/{after['Rej-F1@0.4']:.2f} | {b:.4f} | {a:.4f} | {a-b:+.4f} |")
            records.append(dict(backbone=backbone, variant=variant, status=status,
                before_map=before["mAP"], after_map=after["mAP"], before_auroc=before["AUROC"],
                after_auroc=after["AUROC"], before_gmiou=b, after_gmiou=a, delta_gmiou=a-b))
            backbone_records.append(records[-1])
        figure.suptitle("Frozen localization · only GEQ presence fine tuning · seed 2023", fontsize=12)
        save(figure, folder / "gmiou_before_after")
        dump(folder / "comparison.json", backbone_records)
        if backbone_sweeps:
            with (folder / "gmiou_threshold_sweep.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(backbone_sweeps[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(backbone_sweeps)
        (folder / "README.md").write_text(f"# {TITLES[backbone]}：GEQ 微调对照\n\n[完整协议与八组结果](../README.md)\n\n![G-mIoU before/after](gmiou_before_after.png)\n\n[PDF](gmiou_before_after.pdf) · [SVG](gmiou_before_after.svg)\n")
    overview.suptitle("Only Global Event Query added to frozen Shared / Multi best checkpoints", fontsize=13)
    save(overview, FIGURES / "cross_backbone_overview/gmiou_before_after")
    comparison_heatmap(records)
    completed = [record for record in records if record["status"] == "COMPLETED"]
    if len(completed) == 8:
        improved = sum(record["delta_gmiou"] > 0 for record in completed)
        decreased = sum(record["delta_gmiou"] < 0 for record in completed)
        auc_decreased = sum(record["after_auroc"] < record["before_auroc"] for record in completed)
        lines += ["", "## 结果解读", "",
            f"固定阈值 0.4 下，{improved}/8 组 G-mIoU@3 提升，{decreased}/8 组下降。定位候选、mAP、mR 与 mR+ 在所有组中保持不变。AUROC 有 {auc_decreased}/8 组下降，不能据此声称 GEQ 普遍改善了有无目标的排序能力；结合完整阈值曲线，本轮结果与存在性分数校准发生变化相符，但未隔离证明原因。", "",
            "[三个预先固定阈值的全部对照 CSV](threshold_comparison.csv)。本轮不依据 Test 再挑选阈值或微调轮数。"]
    lines += ["", "## G-mIoU 前后对照", "", "![固定工作点](cross_backbone_overview/gmiou_at_0_4_before_after.png)", "",
              "颜色仅按行归一化，比较同一 backbone、同一方法的前后两个值；单元格为 0–1 原始值。灰色表示未完成。", "",
              "![全部八组](cross_backbone_overview/gmiou_before_after.png)", "",
              "[PDF](cross_backbone_overview/gmiou_before_after.pdf) · [SVG](cross_backbone_overview/gmiou_before_after.svg)", "",
              "灰色水平线为全部拒绝时的 G-mIoU（负查询占比），用于识别高阈值处由空集判对贡献的分数。阈值扫描只分析冻结预测，不按 Test 曲线选择阈值。", "",
              "每个 backbone 的图和 README 独立放在其同名文件夹内。未完成实验以状态占位，不填入结果。各方法的下降项原样报告，单 Seed 不提供跨 Seed 方差。"]
    markdown = "\n".join(lines) + "\n"
    (FIGURES / "README.md").write_text(markdown)
    if OUTPUT.resolve() != FIGURES.resolve():
        (OUTPUT / "README.md").write_text(markdown.replace("](cross_backbone_overview/", "](" + str(FIGURES / "cross_backbone_overview") + "/"))
    dump(OUTPUT / "comparison.json", records)
    all_fields = list(dict.fromkeys(key for record in records for key in record))
    for destination in (OUTPUT / "comparison.csv", FIGURES / "comparison.csv"):
        with destination.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=all_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    if threshold_records:
        for destination in (OUTPUT / "threshold_comparison.csv", FIGURES / "threshold_comparison.csv"):
            with destination.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(threshold_records[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(threshold_records)
    print("REPORT_UPDATED", sum(record["status"] == "COMPLETED" for record in records), "/8", FIGURES, flush=True)
    from plot_complete_comparison import refresh
    refresh()


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "plot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
