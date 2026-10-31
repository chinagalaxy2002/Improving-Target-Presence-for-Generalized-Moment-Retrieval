#!/usr/bin/env python3
"""Live Val selection curves and final Test summaries; no Test-based ranking."""
import csv
import fcntl
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from study_paths import BACKBONES, FIGURES, MODES, OUTPUT


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def history(path):
    result = []
    if path.is_file():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return result


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    queue = read(OUTPUT / "QUEUE.json")
    lines = ["# GEQ 全量微调与完全冻结并行对照", "",
        "四个骨干各两个方案，共八个任务。full 与 frozen 进入同一个队列，按骨干交替启动两种方案，不等待全量阶段全部结束。两方案均从 main E1 的原始预训练 GMR 检查点独立初始化，seed 2023；每任务 num_workers=2，每卡最多两个 CUDA 任务，计入既有作业。", "",
        "full 保留 native GMR 损失并加入 GEQ BCE，GMR lr=1e-5、GEQ lr=3e-5；frozen 冻结原模型并保持 eval，仅训练 GEQ BCE。两方案初始 GMR/GEQ 张量相同。Moment-DETR 特征顺序按检查点记录的 CLIP→SlowFast 修正。旧多路线实验与未完成的 EaTR 重训均不作为这次的初始化来源。", "",
        "逐轮以 **Val G-mIoU@1 在 0.4/0.5/0.6 的均值**选 best，记录每点相对 baseline 的增益及是否三点均超过。连续 **超过5轮（6轮）**均值不提升早停，上限20轮。Test 仅在选模结束后评估，不参与轮次或阈值选择。", "",
        "‘平均值高很多’尚无数值门槛，因此记录实际增益，不把微小提升或未超过 baseline 的 best 宣称为恢复性能。mAP、AUROC、G-mIoU@3 和正负查询指标另行报告。", "",
        "| Backbone | 方案 | 状态 | 完成 Epoch | Best Epoch | Best Val 三点均值 | 相对 baseline | 三点均超过 | Test mAP (%) | Test G-mIoU@1@0.4 |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|"]
    records = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42, "svg.fonttype": "none"})
    for backbone in BACKBONES:
        folder = FIGURES / backbone
        folder.mkdir(parents=True, exist_ok=True)
        figure, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
        baseline_drawn = False
        for mode, color, marker in (("full", "#0072B2", "o"), ("frozen", "#D55E00", "s")):
            root = OUTPUT / backbone / mode
            complete = read(root / "COMPLETE.json")
            items = history(root / "history.jsonl")
            state = queue.get("jobs", {}).get(f"{backbone}/{mode}", {})
            status = complete.get("status", state.get("status", "WAITING"))
            best = complete.get("selection", items[-1]["selection"] if items else {})
            if items:
                ax.plot([i["epoch"] for i in items], [i["selection"]["mean"] for i in items],
                        color=color, marker=marker, markersize=4, label=mode)
                if not baseline_drawn:
                    ax.axhline(items[0]["selection"]["baseline_mean"], color="#666666", linestyle="--", label="Original GMR")
                    baseline_drawn = True
            metric = read(root / "test_metrics.json").get("0.4", {}).get("brief", {}) if complete else {}
            best_mean = best.get("best_mean")
            base_mean = best.get("baseline_mean")
            gain = best_mean - base_mean if best_mean is not None else None
            record = dict(backbone=backbone, mode=mode, status=status,
                epoch=items[-1]["epoch"] if items else None, best_epoch=best.get("best_epoch"),
                best_val_mean=best_mean, val_mean_gain=gain,
                all_three_above=best.get("best_all_three_above_baseline"), test_map=metric.get("mAP"),
                test_gmiou_1_0_4=metric.get("G-mIoU@1", 0) / 100 if metric else None)
            records.append(record)
            fmt = lambda value, precision=4: "—" if value is None else f"{value:.{precision}f}"
            lines.append(f"| {backbone} | {mode} | {status} | {record['epoch'] or '—'} | {record['best_epoch'] or '—'} | {fmt(best_mean)} | {fmt(gain)} | {record['all_three_above']} | {fmt(record['test_map'],2)} | {fmt(record['test_gmiou_1_0_4'])} |")
        ax.set(xlabel="Fine-tuning epoch", ylabel="Val mean G-mIoU@1 (0–1)", ylim=(0, 1), title=backbone + " · Val-based checkpoint selection")
        ax.grid(alpha=0.2)
        if baseline_drawn:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Waiting for first completed Val epoch", transform=ax.transAxes, ha="center")
        for extension in ("png", "pdf", "svg"):
            figure.savefig(folder / f"validation_selection.{extension}", dpi=180)
        plt.close(figure)
        (folder / "README.md").write_text(f"# {backbone}\n\n[全部八组状态](../README.md)\n\n![Val selection](validation_selection.png)\n")
    lines += ["", "## Val 选模曲线", ""]
    for backbone in BACKBONES:
        lines += [f"### {backbone}", "", f"![{backbone} Val selection]({backbone}/validation_selection.png)", ""]
    text = "\n".join(lines) + "\n"
    (FIGURES / "README.md").write_text(text)
    (OUTPUT / "README.md").write_text(text.replace("](", "](" + str(FIGURES) + "/"))
    (OUTPUT / "summary.json").write_text(json.dumps(records, indent=2) + "\n")
    with (OUTPUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    print("STUDY_REPORT_UPDATED", len(records), flush=True)


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "report.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
