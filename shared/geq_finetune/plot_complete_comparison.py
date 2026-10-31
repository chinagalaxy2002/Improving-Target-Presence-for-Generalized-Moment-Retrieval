#!/usr/bin/env python3
"""Six-panel GMR / GMR+GEQ / Multi+GEQ / Shared+GEQ comparisons.

Refresh measurements once with --refresh. Offline rendering from --snapshot
requires only NumPy and Matplotlib; it never loads weights or uses a GPU.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
import numpy as np

from common import (BACKBONES, FIGURES, OUTPUT, PROJECT, dump, original_metrics,
                    original_prediction, read, rows, sha, source_dir)

NAMES = {"moment_detr": "Moment-DETR", "eatr": "EaTR", "qd_detr": "QD-DETR", "cg_detr": "CG-DETR"}
METHODS = ("baseline", "baseline_geq", "multi_geq", "shared_geq")
LABELS = {"baseline": "GMR baseline", "baseline_geq": "GMR baseline + GEQ",
          "multi_geq": "multi + GEQ", "shared_geq": "shared + GEQ"}
STYLES = {
    "baseline": dict(color="#64748B", linestyle="--", marker="o"),
    "baseline_geq": dict(color="#E69F00", linestyle=":", marker="D"),
    "multi_geq": dict(color="#009E73", linestyle="-.", marker="^"),
    "shared_geq": dict(color="#0072B2", linestyle="-", marker="s"),
}
GRID = [index / 100 for index in range(101)]
REFERENCE = "https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/tree/main/backbonefolder/figures/complete_comparison"
START = "<!-- THREE_WAY_COMPARISON_START -->"
END = "<!-- THREE_WAY_COMPARISON_END -->"


def measurements():
    # Keep plotting-only imports independent of the server research environment.
    from run import evaluator
    result = {"schema_version": 2, "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reference_layout": REFERENCE, "split": "SoccerGMR Standard Test", "methods": list(METHODS),
        "gmiou_unit": "0–1", "mAP_unit": "percent; positive queries; no existence gate",
        "positive_decision": "pred_exist_score > threshold", "grid": GRID, "backbones": {}}
    for backbone in BACKBONES:
        evaluate, compute_gmiou, gate, normalize = evaluator(backbone)
        if backbone == "moment_detr":
            from metrics import compute_gmr_cls, get_existence_score
            gt_path = PROJECT / "soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/test.jsonl"
        else:
            from official_eval.metrics import compute_gmr_cls, get_existence_score
            gt_path = Path("/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/data/label/Standard/test.jsonl")
        gt = normalize(rows(gt_path))
        assert len(gt) == len({row["qid"] for row in gt}) == 1036
        positive = np.asarray([bool(row["relevant_windows"]) for row in gt])
        assert int(positive.sum()) == 544 and int((~positive).sum()) == 492
        series = {}
        for method in METHODS:
            baseline = method == "baseline"
            variant = method.removesuffix("_geq")
            directory = source_dir(backbone, "baseline") if baseline else OUTPUT / backbone / variant
            pred_path = original_prediction(backbone, "baseline") if baseline else directory / "test_after_predictions.jsonl"
            metric_path = original_metrics(backbone, "baseline") if baseline else directory / "test_after_metrics.json"
            expected = read(metric_path) if baseline else read(metric_path)["0.4"]
            predictions = rows(pred_path)
            by_id = {row["qid"]: row for row in predictions}
            assert len(predictions) == len(by_id) == 1036
            assert set(by_id) == {row["qid"] for row in gt}
            predictions = [by_id[row["qid"]] for row in gt]
            scores = np.asarray([get_existence_score(row)[0] for row in predictions])
            assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
            metrics = evaluate(predictions, gt, cls_thresholds=(0.4, 0.5, 0.6),
                               gmiou_cls_threshold=0.4, map_num_workers=0, verbose=False)
            for key, value in expected["brief"].items():
                if key in metrics["brief"]:
                    assert math.isclose(float(value), metrics["brief"][key], abs_tol=1e-6), (backbone, method, key)
            classification = compute_gmr_cls(predictions, gt, thresholds=tuple(GRID))
            grid = []
            for threshold in GRID:
                gated, _ = gate(predictions, cls_threshold=threshold, max_pred_windows=10)
                gmiou = compute_gmiou(gated, gt, k_list=(1, 3))
                cls = classification["per_threshold"][str(threshold)]
                grid.append(dict(threshold=threshold, gmiou_1=gmiou["G-mIoU@1"] / 100,
                    gmiou_3=gmiou["G-mIoU@3"] / 100, rejection_f1=cls["Rej-F1"] / 100,
                    negative_accuracy=cls["TN"] / 492, positive_recall=cls["TP"] / 544,
                    tp=cls["TP"], tn=cls["TN"], fp=cls["FP"], fn=cls["FN"]))
            if not baseline:
                with (directory / "test_after_sweep.csv").open(newline="") as stream:
                    previous = list(csv.DictReader(stream))
                assert len(previous) == 101
                for a, b in zip(grid, previous):
                    for key, old_key in (("gmiou_1", "G-mIoU@1"), ("gmiou_3", "G-mIoU@3")):
                        assert math.isclose(a[key], float(b[old_key]), abs_tol=1e-12), (backbone, method, key)
            empirical = []
            for threshold in np.unique(np.r_[0.0, scores, 1.0]):
                predicted = scores > threshold
                empirical.append(dict(threshold=float(threshold),
                    positive_recall=float(predicted[positive].mean()),
                    negative_accuracy=float((~predicted[~positive]).mean())))
            config = read(directory / "experiment_config.json")
            if method == "baseline_geq":
                assert config["source_prediction_sha256"] == sha(original_prediction(backbone, "baseline"))
                assert config["source_sha256"] == sha(source_dir(backbone, "baseline") / "best.ckpt")
            series[method] = dict(label=LABELS[method], seed=config.get("seed", 2023),
                metrics=metrics["brief"], grid=grid, empirical_tradeoff=empirical,
                source_predictions=str(pred_path), source_predictions_sha256=sha(pred_path),
                source_metrics=str(metric_path), source_metrics_sha256=sha(metric_path),
                source_config=str(directory / "experiment_config.json"),
                source_config_sha256=sha(directory / "experiment_config.json"),
                checks=dict(queries=1036, positive=544, negative=492,
                            full_metric_recomputation_matches=True, geq_sweep_matches_archive=not baseline))
            print("MEASURED", backbone, method, "mAP", metrics["brief"]["mAP"], flush=True)
        result["backbones"][backbone] = dict(ground_truth_sha256=sha(gt_path), series=series)
    validate(result)
    return result


def validate(snapshot):
    assert set(snapshot["backbones"]) == set(BACKBONES)
    assert snapshot["methods"] == list(METHODS)
    for backbone, data in snapshot["backbones"].items():
        assert set(data["series"]) == set(METHODS)
        assert data["series"]["baseline"]["metrics"]["mAP"] == data["series"]["baseline_geq"]["metrics"]["mAP"]
        for method, series in data["series"].items():
            grid = series["grid"]
            assert len(grid) == 101
            for index, point in enumerate(grid):
                assert math.isclose(point["threshold"], index / 100, abs_tol=1e-12)
                assert all(math.isfinite(point[key]) and 0 <= point[key] <= 1
                           for key in ("gmiou_1", "gmiou_3", "rejection_f1", "negative_accuracy", "positive_recall"))
                assert point["tp"] + point["fn"] == 544 and point["tn"] + point["fp"] == 492
            for key, metric in (("gmiou_1", "G-mIoU@1"), ("gmiou_3", "G-mIoU@3")):
                assert math.isclose(grid[40][key], series["metrics"][metric] / 100, abs_tol=1e-12), (backbone, method)
            assert math.isclose(grid[40]["rejection_f1"], series["metrics"]["Rej-F1@0.4"] / 100, abs_tol=1e-12)


def configure_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "gmr-geq-three-way-20261031", "savefig.facecolor": "white"})


def save(figure, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        path = destination.with_suffix("." + suffix)
        temporary = path.with_name(path.stem + ".tmp" + path.suffix)
        figure.savefig(temporary, dpi=220, bbox_inches="tight", pad_inches=0.12)
        if suffix == "svg":
            temporary.write_text("\n".join(line.rstrip() for line in temporary.read_text().splitlines()) + "\n")
        temporary.replace(path)
    plt.close(figure)


def axis(ax, title, xlabel, ylabel, bounded=True):
    ax.set_title(title, loc="left", pad=10)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(0, 1))
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", color="#DFE7F1", linewidth=0.7)
    ax.set_axisbelow(True)
    if bounded:
        ax.set_ylim(0, 1)


def line(ax, x, y, method, markers=True):
    style = STYLES[method].copy()
    if not markers:
        style.pop("marker")
    return ax.plot(x, y, **style, linewidth=1.9, markersize=3.8,
                   markevery=10 if markers else None, label=LABELS[method])[0]


def render(backbone, data, output):
    series = data["series"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.9))
    figure.subplots_adjust(left=0.065, right=0.986, bottom=0.11, top=0.855, hspace=0.4, wspace=0.29)
    g1, g3, delta, tradeoff, f1, fixed = axes.flat
    axis(g1, "(a) Threshold sensitivity", "Presence threshold", "G-mIoU@1 (0–1)")
    axis(g3, "(b) Top-3 retrieval", "Presence threshold", "G-mIoU@3 (0–1)")
    axis(delta, "(c) Relative to GMR baseline", "Presence threshold", "Δ G-mIoU@1 (0–1 units)", bounded=False)
    axis(tradeoff, "(d) Rejection / positive retention", "Positive recall (0–1)", "N-acc (0–1)")
    axis(f1, "(e) Negative-class F1", "Presence threshold", "Rejection F1 (0–1)")
    fixed.set(xlabel="Presence threshold", ylabel="G-mIoU@1 (0–1)", ylim=(0, 1))
    fixed.set_title("(f) Fixed operating points", loc="left", pad=10)
    fixed.grid(axis="y", color="#DFE7F1", linewidth=0.7)
    fixed.set_axisbelow(True)
    baseline = np.asarray([row["gmiou_1"] for row in series["baseline"]["grid"]])
    deltas = []
    for method in METHODS:
        rows_ = series[method]["grid"]
        values1 = np.asarray([row["gmiou_1"] for row in rows_])
        difference = values1 - baseline
        line(g1, GRID, values1, method)
        line(g3, GRID, [row["gmiou_3"] for row in rows_], method)
        line(delta, GRID, difference, method)
        deltas.extend(difference.tolist())
        empirical = series[method]["empirical_tradeoff"]
        line(tradeoff, [r["positive_recall"] for r in empirical], [r["negative_accuracy"] for r in empirical], method, markers=False)
        for threshold in (0.4, 0.5, 0.6):
            point = rows_[round(threshold * 100)]
            tradeoff.scatter(point["positive_recall"], point["negative_accuracy"],
                             color=STYLES[method]["color"], marker=STYLES[method]["marker"], s=29, zorder=5)
        line(f1, GRID, [row["rejection_f1"] for row in rows_], method)
        index = METHODS.index(method)
        width = 0.18
        positions = np.arange(3) + (index - (len(METHODS) - 1) / 2) * width
        values = [rows_[round(t * 100)]["gmiou_1"] for t in (0.4, 0.5, 0.6)]
        bars = fixed.bar(positions, values, width=width * 0.94, color=STYLES[method]["color"], zorder=3)
        fixed.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, rotation=90, fontsize=8.5)
    fixed.set_xticks(range(3), ("0.4", "0.5", "0.6"))
    for panel in (g1, g3, delta, f1):
        panel.axvline(0.4, color="#94A3B8", linestyle=":", linewidth=0.7, zorder=0)
    delta.axhline(0, color="#94A3B8", linewidth=0.7, zorder=0)
    low, high = min(deltas), max(deltas)
    span = max(0.05, high - low)
    delta.set_ylim(low - 0.10 * span, high + 0.14 * span)
    delta.yaxis.set_major_locator(MaxNLocator(6))
    handles = [Line2D([0], [0], **STYLES[m], linewidth=1.9, markersize=4.5,
                label=f"{LABELS[m]}  (mAP {series[m]['metrics']['mAP']:.2f}%)") for m in METHODS]
    figure.suptitle(NAMES[backbone] + " | SoccerGMR Standard Test", fontsize=14, fontweight="bold", y=0.99)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.948), ncol=4,
                  frameon=False, columnspacing=1.1, handlelength=2.4, fontsize=10)
    note = "Frozen Test predictions | GEQ fixed epoch 4 for baseline / multi / shared | 101 thresholds | mAP without presence gating"
    figure.text(0.5, 0.037, note, ha="center", va="bottom", fontsize=9.5)
    protocol = ("Baseline seed 2026, batch 32; GEQ seed 2023. Mixed training protocols; not a controlled single-factor comparison."
                if backbone == "eatr" else "Seed 2023; source training budgets / batch sizes differ; see README. No Test-based model or threshold selection.")
    figure.text(0.5, 0.014, protocol, ha="center", va="bottom", fontsize=9)
    save(figure, output / backbone / "gmr_multi_shared_geq_complete_comparison")
    # Large two-panel export isolates the two explicitly requested metrics.
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    figure.subplots_adjust(top=0.72, bottom=0.18, wspace=0.30, left=0.08, right=0.99)
    for ax, metric, title in zip(axes, ("gmiou_1", "gmiou_3"), ("G-mIoU@1", "G-mIoU@3")):
        axis(ax, title, "Presence threshold", title + " (0–1)")
        for method in METHODS:
            line(ax, GRID, [r[metric] for r in series[method]["grid"]], method)
        ax.axvline(0.4, color="#94A3B8", linestyle=":", linewidth=0.7)
    figure.suptitle(NAMES[backbone] + " | GMR, GMR + GEQ, multi + GEQ, shared + GEQ", fontsize=12, y=0.995)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.935), ncol=2, fontsize=9, frameon=False)
    figure.text(0.5, 0.005, "Frozen Standard Test predictions | 101 thresholds | all G-mIoU axes span 0–1", ha="center", fontsize=9)
    save(figure, output / backbone / "gmr_multi_shared_geq_gmiou_1_3")


def managed_block(path, text):
    original = path.read_text() if path.is_file() else "# GEQ evaluation\n"
    original = re.sub(re.escape(START) + r".*?" + re.escape(END) + r"\n*", "", original, flags=re.S)
    first, separator, rest = original.partition("\n")
    path.write_text(first + "\n\n" + START + "\n" + text + "\n" + END + "\n\n" + rest.lstrip())


def csv_write(path, data):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def publish(snapshot, output):
    table = ["| Backbone | GMR baseline mAP (%) | GMR baseline + GEQ mAP (%) | multi + GEQ mAP (%) | shared + GEQ mAP (%) |",
             "|---|---:|---:|---:|---:|"]
    rows_, curve_rows, fixed_rows = [], [], []
    for backbone in BACKBONES:
        data = snapshot["backbones"][backbone]
        series = data["series"]
        table.append("| " + NAMES[backbone] + " | " + " | ".join(f"{series[m]['metrics']['mAP']:.2f}" for m in METHODS) + " |")
        baseline = series["baseline"]["grid"]
        for method in METHODS:
            rows_.append(dict(backbone=backbone, method=LABELS[method], mAP=series[method]["metrics"]["mAP"],
                              AUROC=series[method]["metrics"]["AUROC"], seed=series[method]["seed"]))
            for index, point in enumerate(series[method]["grid"]):
                item = dict(backbone=backbone, method=LABELS[method], **point,
                            delta_gmiou_1=point["gmiou_1"] - baseline[index]["gmiou_1"],
                            delta_gmiou_3=point["gmiou_3"] - baseline[index]["gmiou_3"])
                curve_rows.append(item)
                if index in (40, 50, 60):
                    fixed_rows.append(item)
        dump(output / backbone / "four_way_comparison_data.json", data)
        csv_write(output / backbone / "four_way_threshold_sweep.csv", [r for r in curve_rows if r["backbone"] == backbone])
        fig = "gmr_multi_shared_geq_complete_comparison"
        detail = "gmr_multi_shared_geq_gmiou_1_3"
        base_g1 = series["baseline"]["grid"][40]["gmiou_1"]
        delta_multi = series["multi_geq"]["grid"][40]["gmiou_1"] - base_g1
        delta_shared = series["shared_geq"]["grid"][40]["gmiou_1"] - base_g1
        text = (f"## GMR baseline / GMR baseline + GEQ / multi + GEQ / shared + GEQ\n\n"
            f"固定阈值 0.4 下，相对 GMR baseline 的 G-mIoU@1 变化：multi + GEQ **{delta_multi:+.4f}**、"
            f"shared + GEQ **{delta_shared:+.4f}**（0–1 单位）。完整曲线展示这些差异随阈值的变化。\n\n"
            f"![六面板对照]({fig}.png)\n\n[PDF]({fig}.pdf) · [SVG]({fig}.svg) · "
            f"[@1 / @3 双面板 PDF]({detail}.pdf) · [101 点数据](four_way_threshold_sweep.csv)\n\n"
            f"Test mAP：GMR baseline **{series['baseline']['metrics']['mAP']:.2f}%**；"
            f"GMR baseline + GEQ **{series['baseline_geq']['metrics']['mAP']:.2f}%**；"
            f"multi + GEQ **{series['multi_geq']['metrics']['mAP']:.2f}%**；"
            f"shared + GEQ **{series['shared_geq']['metrics']['mAP']:.2f}%**。\n\n"
            "G-mIoU、N-acc、Positive recall、Rej-F1 均使用 0–1；差值允许为负。"
            "mAP 按官方正查询定位口径计算，不按存在性阈值门控。原始配对图保留在下方。")
        if backbone == "eatr":
            text += "\n\nEaTR baseline 复用历史 seed 2026、batch 32；GEQ 为 seed 2023，存在训练协议差异。"
        managed_block(output / backbone / "README.md", text)
    csv_write(output / "four_way_map.csv", rows_)
    csv_write(output / "four_way_threshold_sweep.csv", curve_rows)
    csv_write(output / "four_way_fixed_points.csv", fixed_rows)
    root_text = "## 四条曲线：GMR baseline、GMR baseline + GEQ、multi + GEQ、shared + GEQ\n\n" + "\n".join(table)
    root_text += ("\n\nmAP 单位为百分数，来自相同 SoccerGMR Standard Test 的正查询，不随存在性阈值变化。"
        "GMR baseline 是当前六路线实验中的 baseline。EaTR baseline 为历史 seed 2026，其他两组为 seed 2023；"
        "各来源的训练预算、batch 也可能不同，完整来源保存在快照中。"
        "新增的 GMR baseline + GEQ 严格从图中同一个 baseline 初始化，固定微调 4 epoch，仅训练 GEQ，"
        "其定位候选与 mAP 保持不变。EaTR 继续采用已完成的历史 baseline，不使用仍在训练的重训快照。"
        "四曲线反映完整方法的效果；GEQ 的独立影响采用各方法与其 +GEQ 的配对结果判断。\n\n"
        "六面板沿用[参考图版式](" + REFERENCE + ")："
        "(a) G-mIoU@1 随阈值；(b) G-mIoU@3 随阈值；(c) 同阈值相对 GMR baseline 的 ΔG-mIoU@1；"
        "(d) 负查询拒绝率与正查询保留率；(e) 拒绝 F1；(f) 阈值 0.4/0.5/0.6 的 G-mIoU@1。"
        "所有 G-mIoU 图轴均覆盖 0–1，差值图保留负值。四条曲线使用同样的颜色、线型与标记跨骨干对应。\n\n"
        "[mAP CSV](four_way_map.csv) · [全部 1,616 行阈值数据](four_way_threshold_sweep.csv) · "
        "[固定工作点](four_way_fixed_points.csv) · [离线快照](four_way_comparison_snapshot.json)\n\n")
    for backbone in BACKBONES:
        base = backbone + "/gmr_multi_shared_geq_complete_comparison"
        root_text += f"### {NAMES[backbone]}\n\n![{NAMES[backbone]} 六面板]({base}.png)\n\n[PDF]({base}.pdf) · [SVG]({base}.svg)\n\n"
    managed_block(output / "README.md", root_text.rstrip())
    dump(output / "four_way_comparison_snapshot.json", snapshot)
    dump(output / "FOUR_WAY_FIGURE_AUDIT.json", dict(status="PASSED", reference=REFERENCE,
         backbones=list(BACKBONES), methods=list(METHODS), measured_series=16,
         newly_finetuned_models=12, baseline_geq_controls=4, thresholds_per_series=101, curve_rows=len(curve_rows),
         fixed_point_rows=len(fixed_rows), gmiou_units="0–1; deltas signed", mAP_unit="percent",
         all_archived_metrics_reproduced=True, all_geq_101_point_sweeps_reproduced=True,
         empirical_tradeoff_uses_all_unique_scores=True, weights_loaded=False, training_launched=False))


def refresh(snapshot_path=None, output=FIGURES, force=False):
    source = Path(snapshot_path) if snapshot_path else OUTPUT / "four_way_comparison_snapshot.json"
    if force or not source.is_file():
        if not all(read(OUTPUT / b / v / "COMPLETE.json").get("status") == "COMPLETED"
                   for b in BACKBONES for v in ("baseline", "shared", "multi")):
            print("FOUR_WAY_WAITING_FOR_TWELVE_COMPLETE_MODELS", flush=True)
            return
        snapshot = measurements()
        dump(source, snapshot)
    else:
        snapshot = read(source)
    validate(snapshot)
    configure_style()
    for backbone in BACKBONES:
        render(backbone, snapshot["backbones"][backbone], output)
    publish(snapshot, output)
    print("FOUR_WAY_PLOTS_COMPLETE", output, "4 six-panel + 4 two-panel figures", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=FIGURES)
    parser.add_argument("--refresh", action="store_true", help="Recompute from saved predictions; never retrain")
    args = parser.parse_args()
    if args.snapshot and args.refresh:
        parser.error("--snapshot and --refresh are mutually exclusive")
    lock_root = args.output if args.snapshot else OUTPUT
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / "plot.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        refresh(args.snapshot, args.output, args.refresh)


if __name__ == "__main__":
    main()
