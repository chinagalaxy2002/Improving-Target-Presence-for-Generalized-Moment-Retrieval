#!/usr/bin/env python3
"""Publish one reproducible 24-run report; never loads a model or reruns Test.

Capture: python build_results_report.py --output /path/to/figures
Offline: python build_results_report.py --snapshot results/snapshot.json --output results
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
PROJECT = Path("/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval")
RUNTIME = Path("/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr")
BACKBONES = ("moment_detr", "eatr", "qd_detr", "cg_detr")
TITLES = {"moment_detr": "Moment-DETR-GMR", "eatr": "EaTR-GMR", "qd_detr": "QD-DETR-GMR", "cg_detr": "CG-DETR-GMR"}
FOLDERS = {"moment_detr": "moment_detr_gmr", "eatr": "eatr_gmr", "qd_detr": "qd_detr_gmr", "cg_detr": "cg_detr_gmr"}
VARIANTS = ("baseline", "shared", "sepffn", "instruction", "multi", "adaptive")
SHORT = ("Base", "Shared", "SepFFN", "Instr", "Multi", "Adapt")
COLORS = ("#666666", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
MARKERS = ("o", "s", "^", "D", "P", "X")
STYLES = ("-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))
PROTOCOL_NOTES = {
    "moment_detr": "Seed 2023; stopped checkpoints have different training budgets.",
    "eatr": "Baseline: seed 2026, effective batch 32; routes: seed 2023, effective batch 8.",
    "qd_detr": "Seed 2023; Base / Shared / SepFFN: effective batch 8; Instr / Multi / Adapt: 32.",
    "cg_detr": "Seed 2023; Base / Shared / SepFFN: effective batch 8; Instr / Multi / Adapt: 32.",
}
METRICS = ("mAP", "mR@1", "mR@5", "mR+@5", "AUROC", "Rej-F1@0.4", "G-mIoU@3_0_1")
STATUS_LABELS = {"completed": "已完成 Test", "training": "训练中", "queued": "排队", "testing": "待完成 Test", "failed": "失败"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def read_history(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A training process may still be writing its final line.
    return records


def finite(value):
    return value if value is not None and math.isfinite(float(value)) else None


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def metric_values(metrics: dict) -> dict:
    brief = metrics["brief"]
    threshold = metrics.get("G-mIoU_gate", {}).get("cls_threshold")
    if threshold is not None and not math.isclose(float(threshold), 0.4):
        raise ValueError(f"Expected threshold 0.4, found {threshold}")
    values = {key: float(brief[key]) for key in METRICS[:-1]}
    values[METRICS[-1]] = float(brief["G-mIoU@3"]) / 100.0
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Nonfinite Test metric")
    return values


def capture(project: Path, runtime: Path, figure_root: Path) -> dict:
    queue = read_json(runtime / "QUEUE.json")
    plan = read_json(runtime / "execution_plan.json")
    rows = []
    for backbone in BACKBONES:
        for variant in VARIANTS:
            moment = backbone == "moment_detr"
            run = project / "mr_experiments" / variant if moment else runtime / "runs" / backbone / variant
            config = read_json(run / "experiment_config.json")
            manifest = read_json(run / "test" / "evaluation_manifest.json") if moment else {}
            status = read_json(run / "status.json")
            reuse = read_json(run / "reused_baseline.json")
            profile = plan.get("jobs", {}).get(f"{backbone}/{variant}", {})
            history = read_history(run / ("train.log" if moment else "history.jsonl"))
            last = history[-1] if history else {}
            job = queue.get("jobs", {}).get(f"{backbone}/{variant}", {})
            metric_path = run / "test" / "metrics_gmiou_0.4.json" if moment else run / "test_metrics.json"
            complete = metric_path.is_file() and (bool(manifest) if moment else status.get("status") == "COMPLETED")
            state = "completed" if complete else "failed" if job.get("status") == "FAILED" else "testing" if status.get("status") == "TRAINED" else "training" if job.get("status") in ("RUNNING", "RETRYING") else "queued"
            best_epoch = manifest.get("best_epoch") if moment else last.get("best_epoch")
            if best_epoch is None and not moment:
                validated = [item for item in history if item.get("validation")]
                if validated:
                    best_epoch = max(validated, key=lambda item: item["validation"]["mAP"])["epoch"]
            best_metrics = read_json(run / "best_val_metrics.json")
            values = metric_values(read_json(metric_path)) if complete else None
            batch = config.get("microbatch", config.get("bsz", profile.get("microbatch", 2)))
            accum = config.get("gradient_accumulation", config.get("accum", profile.get("gradient_accumulation", 4)))
            row = {
                "backbone": backbone, "variant": variant, "status": state,
                "latest_epoch": last.get("epoch", status.get("epoch")), "best_epoch": best_epoch,
                "best_val_map": finite(status.get("best_val_map", last.get("best_val_map", best_metrics.get("brief", {}).get("mAP")))),
                "seed": config.get("seed", reuse.get("seed", 2023)),
                "microbatch": batch, "gradient_accumulation": accum, "effective_batch": batch * accum,
                "learning_rate": config.get("learning_rate", config.get("lr", 5e-5)),
                "reused_baseline": bool(reuse), "gpu": job.get("gpu"), "test": values,
                "source": f"mr_experiments/{variant}/test/metrics_gmiou_0.4.json" if moment else f"runs/{backbone}/{variant}/test_metrics.json",
                "source_sha256": digest(metric_path) if complete else None,
            }
            rows.append(row)
    sweeps = []
    for backbone in BACKBONES:
        path = figure_root / FOLDERS[backbone] / "gmiou_threshold_sweep.csv"
        if path.is_file():
            with path.open(newline="") as stream:
                for raw in csv.DictReader(stream):
                    sweeps.append({"backbone": backbone, "variant": raw["variant"], "threshold": float(raw["threshold"]), "G-mIoU@3_0_1": float(raw["G-mIoU@3_0_1"]), "predicted_positive": int(raw["predicted_positive"]), "predicted_negative": int(raw["predicted_negative"])})
    snapshot = {"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "test_split": "SoccerGMR Standard Test", "gmiou_threshold": 0.4, "queue_status": queue.get("status", "RUNNING"), "rows": rows, "sweeps": sweeps}
    validate(snapshot)
    return snapshot


def validate(snapshot: dict) -> None:
    rows = snapshot["rows"]
    keys = {(row["backbone"], row["variant"]) for row in rows}
    if len(rows) != 24 or len(keys) != 24:
        raise ValueError("Expected four backbones × six variants")
    for row in rows:
        if row["status"] != "completed":
            if row["test"] is not None:
                raise ValueError("Unfinished run has Test values")
            continue
        selected = sorted((item for item in snapshot["sweeps"] if (item["backbone"], item["variant"]) == (row["backbone"], row["variant"])), key=lambda item: item["threshold"])
        if len(selected) != 101 or any(not math.isclose(item["threshold"], index / 100, abs_tol=1e-8) for index, item in enumerate(selected)):
            raise ValueError(f"Incomplete 0–1 sweep: {row['backbone']}/{row['variant']}")
        if any(not 0 <= item["G-mIoU@3_0_1"] <= 1 for item in selected):
            raise ValueError("G-mIoU outside 0–1")
        if not math.isclose(selected[40]["G-mIoU@3_0_1"], row["test"]["G-mIoU@3_0_1"], abs_tol=0.00006):
            raise ValueError(f"Stale sweep at tau=0.4: {row['backbone']}/{row['variant']}")


def configure_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False, "grid.alpha": 0.2, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "gmr-mr-detr-20261030"})


def save_figure(figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        path = destination.with_suffix(f".{extension}")
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        figure.savefig(temporary, dpi=200, bbox_inches="tight", facecolor="white")
        if extension == "svg":
            temporary.write_text("\n".join(line.rstrip() for line in temporary.read_text().splitlines()) + "\n")
        os.replace(temporary, path)
    plt.close(figure)


def render_overview(snapshot: dict, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    settings = (("mAP", "(a) Test mAP (%)", 25), ("mR+@5", "(b) Additional-moment recall (%)", 15), ("AUROC", "(c) Test AUROC (%)", 100), ("G-mIoU@3_0_1", "(d) G-mIoU@3 at threshold 0.4", 1))
    for ax, (metric, title, maximum) in zip(axes.flat, settings):
        matrix = np.full((4, 6), np.nan)
        for i, backbone in enumerate(BACKBONES):
            for j, variant in enumerate(VARIANTS):
                row = next(row for row in snapshot["rows"] if row["backbone"] == backbone and row["variant"] == variant)
                if row["test"] is not None:
                    matrix[i, j] = row["test"][metric]
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#EDEDED")
        # AUROC values use a clearly labeled 50–100 range; exact values are printed.
        minimum = 50 if metric == "AUROC" else 0
        maximum = max(maximum, math.ceil(float(np.nanmax(matrix)) / 5) * 5) if metric != "G-mIoU@3_0_1" else 1
        im = ax.imshow(matrix, cmap=cmap, vmin=minimum, vmax=maximum, aspect="auto")
        for i in range(4):
            for j in range(6):
                value = matrix[i, j]
                label = "—" if np.isnan(value) else f"{value:.4f}" if metric == "G-mIoU@3_0_1" else f"{value:.2f}"
                color = "black" if np.isnan(value) or (value - minimum) / (maximum - minimum) > 0.55 else "white"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=10)
        ax.set_xticks(range(6), SHORT, rotation=25, ha="right")
        ax.set_yticks(range(4), ("Moment", "EaTR", "QD-DETR", "CG-DETR"))
        ax.set_title(title, loc="left")
        figure.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    figure.suptitle("Frozen-best Test results · gray cells have no completed Test", fontsize=13)
    figure.supxlabel("Mixed training protocols; see README. G-mIoU is 0–1; other metrics are percentages.", fontsize=9)
    save_figure(figure, output / "cross_backbone_overview" / "all_backbones_metrics")

    states = ("completed", "training", "queued", "testing", "failed")
    palette = ("#CCE7DD", "#B8D9ED", "#ECECEC", "#F2DEB3", "#EDC2C6")
    matrix = np.zeros((4, 6))
    figure, ax = plt.subplots(figsize=(9, 3.8), layout="constrained")
    for i, backbone in enumerate(BACKBONES):
        for j, variant in enumerate(VARIANTS):
            row = next(row for row in snapshot["rows"] if row["backbone"] == backbone and row["variant"] == variant)
            matrix[i, j] = states.index(row["status"])
    ax.imshow(matrix, cmap=ListedColormap(palette), vmin=-0.5, vmax=4.5, aspect="auto")
    for i, backbone in enumerate(BACKBONES):
        for j, variant in enumerate(VARIANTS):
            row = next(row for row in snapshot["rows"] if row["backbone"] == backbone and row["variant"] == variant)
            text = {"completed": "Test done", "training": "Training", "queued": "Queued", "testing": "Testing", "failed": "Failed"}[row["status"]]
            if row["latest_epoch"] is not None:
                text += f"\nepoch {row['latest_epoch']}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10)
    ax.set_xticks(range(6), SHORT)
    ax.set_yticks(range(4), ("Moment", "EaTR", "QD-DETR", "CG-DETR"))
    ax.set_title("Training / frozen-best Test status · " + snapshot["generated_at"][:19].replace("T", " "), fontsize=12)
    save_figure(figure, output / "cross_backbone_overview" / "training_status")


def render_backbone(snapshot: dict, backbone: str, output: Path) -> None:
    rows = [row for row in snapshot["rows"] if row["backbone"] == backbone and row["test"] is not None]
    if not rows:
        return
    folder = output / FOLDERS[backbone]
    figure, axes = plt.subplots(2, 2, figsize=(9, 6.5), layout="constrained")
    settings = (("mAP", "(a) Localization mAP (%)", None), ("mR+@5", "(b) Additional-moment recall (%)", None), ("AUROC", "(c) Existence AUROC (%)", 100), ("G-mIoU@3_0_1", "(d) G-mIoU@3 at threshold 0.4", 1))
    for ax, (metric, title, upper) in zip(axes.flat, settings):
        for index, variant in enumerate(VARIANTS):
            row = next((row for row in rows if row["variant"] == variant), None)
            if row is None:
                ax.text(index, 0.04, "pending", rotation=90, ha="center", va="bottom", transform=ax.get_xaxis_transform(), color="#666666", fontsize=9)
                continue
            value = row["test"][metric]
            ax.bar(index, value, color=COLORS[index], width=0.64, alpha=0.85)
            ax.scatter(index, value, color="black", marker=MARKERS[index], s=24, zorder=4)
            ax.annotate(f"{value:.4f}" if metric == "G-mIoU@3_0_1" else f"{value:.2f}", (index, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
        maximum = upper or max(row["test"][metric] for row in rows) * 1.25
        ax.set_ylim(0, maximum)
        ax.set_xlim(-0.6, 5.6)
        ax.set_xticks(range(6), SHORT, rotation=20)
        ax.set_title(title, loc="left")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    figure.suptitle(TITLES[backbone] + " · completed frozen-best Test runs", fontsize=13)
    figure.supxlabel(PROTOCOL_NOTES[backbone], fontsize=9)
    save_figure(figure, folder / "test_summary")

    figure, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    for row in rows:
        index = VARIANTS.index(row["variant"])
        selected = sorted((item for item in snapshot["sweeps"] if item["backbone"] == backbone and item["variant"] == row["variant"]), key=lambda item: item["threshold"])
        ax.plot([item["threshold"] for item in selected], [item["G-mIoU@3_0_1"] for item in selected], color=COLORS[index], linestyle=STYLES[index], marker=MARKERS[index], markevery=10, markersize=4, linewidth=1.7, label=row["variant"])
    ax.axvline(0.4, color="#999999", linestyle=":", linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Existence threshold", ylabel="G-mIoU@3 (0–1)", title=TITLES[backbone] + " · full-domain threshold sensitivity")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    ax.grid()
    figure.supxlabel(PROTOCOL_NOTES[backbone], fontsize=9)
    save_figure(figure, folder / "gmiou_sensitivity")


def fmt(value, digits=2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def report_markdown(snapshot: dict) -> str:
    rows = snapshot["rows"]
    counts = Counter(row["status"] for row in rows)
    lines = ["# GMR × Mr. DETR：四骨干完整实验结果", "", f"更新时间：**{snapshot['generated_at']}**。本页是冻结最佳检查点 Test 结果与训练状态的统一快照。", "", f"共 **24 组**：**{counts['completed']} 组完成 Test**、{counts['training']} 组训练中、{counts['queued']} 组排队、{counts['testing']} 组待完成 Test、{counts['failed']} 组失败。", "", "Test 指标均来自 SoccerGMR Standard Test；mAP、mR、mR+、AUROC、Rej-F1 沿用官方百分数，**G-mIoU@3 转换为 0–1**。Rej-F1 与表中 G-mIoU 固定使用存在性阈值 **0.4**。未完成 Test 的指标留空，验证集结果只列在后面的训练状态表。", "", "[全部 24 行 CSV](results.csv) · [机器可读快照与来源哈希](snapshot.json)", "", "## 完整 Test 表", "", "| Backbone | 方法 | 状态 | Best Epoch | mAP | mR@1 | mR@5 | mR+@5 | AUROC | Rej-F1@0.4 | G-mIoU@3 |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = row["test"] or {}
        numbers = [fmt(values.get(metric), 4 if metric == METRICS[-1] else 2) for metric in METRICS]
        name = row["variant"] + ("（历史复用）" if row["reused_baseline"] else "")
        lines.append("| " + " | ".join([TITLES[row["backbone"]], name, STATUS_LABELS[row["status"]], str(row["best_epoch"] or "—"), *numbers]) + " |")
    lines += ["", "## 综合图", "", "![四个骨干的 Test 指标对照](cross_backbone_overview/all_backbones_metrics.png)", "", "各路线的收益随骨干和指标变化；灰色单元格表示尚无完成的 Test。每格标注精确数值，AUROC 色阶为 50–100，其余色阶从 0 起；这些图不按 Test 重新选择检查点或阈值。", "", "[PDF](cross_backbone_overview/all_backbones_metrics.pdf) · [SVG](cross_backbone_overview/all_backbones_metrics.svg)", "", "![24 组训练与测试状态](cross_backbone_overview/training_status.png)", "", "[状态图 PDF](cross_backbone_overview/training_status.pdf) · [状态图 SVG](cross_backbone_overview/training_status.svg)", "", "## 训练状态快照", "", "以下为尚未完成 Test 的任务；Val mAP 与上表 Test mAP 属于不同 split。Epoch 是最后完整写入记录的轮次。", "", "| Backbone | 方法 | 状态 | GPU | 已完成 Epoch | Best Epoch | Best Val mAP |", "|---|---|---|---|---:|---:|---:|"]
    for row in rows:
        if row["status"] != "completed":
            lines.append("| " + " | ".join([TITLES[row["backbone"]], row["variant"], STATUS_LABELS[row["status"]], str(row["gpu"] if row["gpu"] is not None else "—"), str(row["latest_epoch"] if row["latest_epoch"] is not None else "—"), str(row["best_epoch"] or "—"), fmt(row["best_val_map"])]) + " |")
    lines += ["", "## 每个 Backbone 的图表", "", "下列曲线同时覆盖 0.00–1.00 的 101 个阈值与 0–1 的 G-mIoU 纵轴。阈值扫描是冻结预测的敏感性分析，不用于 Test 调参。", "", "| Backbone | 指标图 | 全域 G-mIoU 曲线 | 数据 |", "|---|---|---|---|"]
    for backbone in BACKBONES:
        folder = FOLDERS[backbone]
        lines.append(f"| {TITLES[backbone]} | [PNG]({folder}/test_summary.png) / [PDF]({folder}/test_summary.pdf) / [SVG]({folder}/test_summary.svg) | [PNG]({folder}/gmiou_sensitivity.png) / [PDF]({folder}/gmiou_sensitivity.pdf) / [SVG]({folder}/gmiou_sensitivity.svg) | [CSV]({folder}/figure_data.csv) / [阈值扫描]({folder}/gmiou_threshold_sweep.csv) |")
    lines += ["", "## 训练协议与可比性", "", "| Backbone / 配置 | Seed | Microbatch × 累积 | 学习率 |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {TITLES[row['backbone']]} / {row['variant']} | {row['seed']} | {row['microbatch']} × {row['gradient_accumulation']} | {row['learning_rate']:g} |")
    lines += ["", "- 仅 EaTR baseline 复用历史 seed 2026、batch 32、LR 3e-5 的 GMR；当前 mAP 为 16.20。原 seed 2023、mAP 6.86 的运行已归档，不能与当前 baseline 混用。", "- QD/CG 的 instruction、multi、adaptive 使用 batch 32、累积 1；其余新路线通常为 2×4。跨这些配置的差异同时包含结构与训练 Batch 的影响。", "- Moment-DETR 六组此前按指令停止，最后完成轮次不同；已完成的是停止时的 best 检查点测试，不代表统一训练预算下的收敛比较。", "- 新路线使用单 seed 2023；没有多 Seed 均值、标准差或统计显著性结论。", "- 高 G-mIoU 可能受空集拒绝与阈值影响；多片段改善需要同时检查 mR+@5，不能只依据综合分数。", "", "## 六种方法", "", "| 方法 | 定义 |", "|---|---|", "| baseline | 原 GMR O2O 主路线，无 O2M 辅助路线 |", "| shared | 一条固定 K=2 O2M 路线，Decoder 全共享 |", "| sepffn | O2M 路线共享注意力，FFN 与 norm3 独立 |", "| instruction | O2M 路线在 Self-Attention 中加入 4 个指令 Token |", "| multi | Separate FFN 与 Instruction 双路线，各权重 0.15 |", "| adaptive | 双路线；第二正样本要求 tIoU≥0.25 |", "", "## 更新与复现", "", "本报告读取现有指标、训练记录和 101 点阈值 CSV，不加载权重或重新执行 Test。服务器的 `refresh_all.sh` 在每次成功测试后刷新本报告与图；GitHub 展示本次提交时的快照，不会在后台自动推送。", "", "GitHub 离线重绘（只需 NumPy、Matplotlib）：", "", "```bash", "python -m pip install numpy matplotlib", "python shared/build_results_report.py --snapshot results/snapshot.json --output results", "```", "", "服务器统一刷新：", "", "```bash", "bash /home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/newidea/figure/gmr_mr_detr_test_20261030/refresh_all.sh", "```", ""]
    return "\n".join(lines)


def write_tables(snapshot: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_text(output / "snapshot.json", json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    fields = ("backbone", "variant", "status", "latest_epoch", "best_epoch", "best_val_map", "seed", "microbatch", "gradient_accumulation", "effective_batch", "learning_rate", *METRICS)
    temporary = output / "results.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in snapshot["rows"]:
            writer.writerow({key: row.get(key, (row["test"] or {}).get(key)) for key in fields})
    os.replace(temporary, output / "results.csv")
    for backbone in BACKBONES:
        folder = output / FOLDERS[backbone]
        folder.mkdir(exist_ok=True)
        completed = [row for row in snapshot["rows"] if row["backbone"] == backbone and row["test"] is not None]
        with (folder / "figure_data.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("variant", "best_epoch", *METRICS), lineterminator="\n")
            writer.writeheader()
            for row in completed:
                writer.writerow({"variant": row["variant"], "best_epoch": row["best_epoch"], **row["test"]})
        sweep = [item for item in snapshot["sweeps"] if item["backbone"] == backbone]
        if sweep:
            fields = ("variant", "threshold", "G-mIoU@3_0_1", "predicted_positive", "predicted_negative")
            with (folder / "gmiou_threshold_sweep.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows({key: item[key] for key in fields} for item in sweep)
        write_text(folder / "README.md", f"# {TITLES[backbone]}\n\n本目录仅包含该骨干的结果图与数据，更新时间 {snapshot['generated_at']}。\n\n[完整 24 组结果与协议说明](../README.md)\n\n![Test 指标](test_summary.png)\n\n![0–1 全域 G-mIoU](gmiou_sensitivity.png)\n\n指标图同时展示定位、额外片段召回、存在性排序与 G-mIoU 的不同变化。数据来自完成的 frozen-best Test，灰色 pending 标签表示尚未测试；统计口径和配置差异见总表。\n")
        write_text(folder / "CAPTION.md", f"# {TITLES[backbone]}\n\n已完成配置在定位、额外片段召回与存在性指标上呈现不同变化。指标图报告完整 SoccerGMR Standard Test 的 mAP、mR+@5、AUROC 与阈值 0.4 下的 G-mIoU@3；G-mIoU 为 0–1，其余为百分数。曲线图对同一份冻结 Test 预测扫描 101 个存在性阈值，两轴均覆盖 0–1；未完成配置省略，单 Seed 结果不绘制跨 Seed 误差条。\n\n{PROTOCOL_NOTES[backbone]}\n")
    write_text(output / "README.md", report_markdown(snapshot))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Portable snapshot for offline reproduction")
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME)
    parser.add_argument("--figure-root", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    snapshot = read_json(args.snapshot) if args.snapshot else capture(args.project_root, args.runtime_root, args.figure_root)
    validate(snapshot)
    configure_style()
    render_overview(snapshot, args.output)
    for backbone in BACKBONES:
        render_backbone(snapshot, backbone, args.output)
    write_tables(snapshot, args.output)
    print(f"Wrote 24-run report to {args.output / 'README.md'} ({Counter(row['status'] for row in snapshot['rows'])})")


if __name__ == "__main__":
    main()
