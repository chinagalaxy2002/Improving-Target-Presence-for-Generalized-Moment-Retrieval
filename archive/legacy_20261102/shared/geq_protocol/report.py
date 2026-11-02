#!/usr/bin/env python3
"""Publish only completed Test curves and keep incomplete studies clearly labeled."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from study_paths import BACKBONES, FIGURES, KIT, MODES, OUTPUT, PROJECT, VARIANTS, dump, sha, original_prediction
import plotting


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def rows(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]


def history(path):
    result = []
    if path.is_file():
        for row in path.read_text().splitlines():
            try:
                result.append(json.loads(row))
            except json.JSONDecodeError:
                pass
    return result


def measured(prediction_path, metric_path, destination):
    signature = dict(prediction_sha256=sha(prediction_path), metric_sha256=sha(metric_path))
    cached = read(destination)
    if cached.get("signature") == signature:
        return cached
    sys.path.insert(0, "/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared")
    from official_eval.metrics import (_clean_pred_windows, _compute_set_iou_score,
        get_existence_score, prepare_submission_for_gmiou, compute_G_mIoU)
    gt = rows(PROJECT / "soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/test.jsonl")
    predictions = rows(prediction_path)
    mapping = {p["qid"]: p for p in predictions}
    assert len(gt) == len(predictions) == len(mapping) == 1036
    assert set(mapping) == {g["qid"] for g in gt}
    predictions = [mapping[g["qid"]] for g in gt]
    y = np.asarray([bool(g["relevant_windows"]) for g in gt])
    scores = np.asarray([get_existence_score(p)[0] for p in predictions])
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    windows = [_clean_pred_windows(p.get("pred_relevant_windows", []), max_pred_windows=10) for p in predictions]
    raw = {k: [_compute_set_iou_score([w[:2] for w in p[:k]], g["relevant_windows"]) for p, g in zip(windows, gt)] for k in (1, 3)}
    grid = []
    for threshold in plotting.GRID:
        positive = scores > threshold
        tp = int((positive & y).sum()); tn = int((~positive & ~y).sum())
        fp = int((positive & ~y).sum()); fn = int((~positive & y).sum())
        point = dict(threshold=threshold, positive_recall=tp / int(y.sum()), negative_accuracy=tn / int((~y).sum()),
            rejection_f1=round(100 * 2 * tn / max(1, 2 * tn + fp + fn), 2) / 100, tp=tp, tn=tn, fp=fp, fn=fn)
        for k in (1, 3):
            values = [raw[k][i] if positive[i] else float(not y[i]) for i in range(len(gt))]
            point[f"gmiou_{k}"] = round(100 * sum(values) / len(values), 2) / 100
        grid.append(point)
    metrics = read(metric_path)
    reports = metrics if "0.4" in metrics else {"0.4": metrics}
    for tau, report in reports.items():
        if tau not in ("0.4", "0.5", "0.6"):
            continue
        gated, _ = prepare_submission_for_gmiou(predictions, cls_threshold=float(tau), max_pred_windows=10)
        official = compute_G_mIoU(gated, gt, k_list=(1, 3))
        for k in (1, 3):
            assert math.isclose(grid[round(float(tau)*100)][f"gmiou_{k}"], official[f"G-mIoU@{k}"] / 100, abs_tol=1e-12)
            assert official[f"G-mIoU@{k}"] == report["brief"][f"G-mIoU@{k}"], (prediction_path, tau, k)
    empirical = [dict(threshold=float(t), positive_recall=float((scores[y] > t).mean()),
                     negative_accuracy=float((scores[~y] <= t).mean())) for t in np.unique(np.r_[0., scores, 1.])]
    result = dict(signature=signature, metrics=reports["0.4"]["brief"], grid=grid, empirical_tradeoff=empirical,
        source_predictions=str(prediction_path), source_metrics=str(metric_path), queries=1036, thresholds=101,
        gmiou_units="0–1", fixed_points_verified_against_official=True)
    dump(destination, result)
    print("MEASURED", prediction_path, flush=True)
    return result


def csv_write(path, records):
    if not records:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)


def render_group(backbone, methods, labels, sources, folder, figure_name, caption):
    folder.mkdir(parents=True, exist_ok=True)
    series = {}
    for key in methods:
        pred, metrics = sources[key]
        series[key] = measured(pred, metrics, folder / (figure_name + "_" + key + "_data.json"))
    data = dict(series=series, protocol_note=caption,
        seed_note="GEQ seed 2023 | Val thresholds 0.4 / 0.5 / 0.6 | Test never used for checkpoint selection",
        figure_name=figure_name, subtitle=" / ".join(labels.values()))
    plotting.setup(methods, labels); plotting.configure_style()
    plotting.render(backbone, data, folder.parent)
    records = [dict(method=labels[m], **point) for m in methods for point in series[m]["grid"]]
    csv_write(folder / (figure_name + "_sweep.csv"), records)
    dump(folder / (figure_name + "_snapshot.json"), dict(backbone=backbone, methods=methods, labels=labels, data=data))
    return series


def baseline_study():
    root = PROJECT / "geq_epoch_study/20261031"
    figures = PROJECT / "newidea/figure/gmr_mr_detr_test_20261030/geq_finetune/epoch_study"
    methods = ("baseline", "full", "frozen")
    labels = dict(baseline="GMR baseline", full="GMR + GEQ (full)", frozen="GMR + GEQ (frozen)")
    lines = ["# main E1 同源初始化：全量与冻结 GEQ 对照", "",
        "四骨干八组均完成训练与完整 Test。Val G-mIoU@1 在0.4/0.5/0.6的均值选 best；连续超过5轮（6轮）无提升早停，上限20轮。Test不参与选模。", "",
        "本组使用 main E1 的历史原始 GMR 检查点；GEQ seed 2023。Moment-DETR按检查点恢复CLIP→SlowFast顺序。与当前多路线实验的初始化不同，分别展示，不能混作同一受控消融。", "",
        "| Backbone | 方法 | Best Epoch | mAP (%) | G1@0.4 | G1@0.5 | G1@0.6 | G1 三点均值 | G3 三点均值 | AUROC (%) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summaries = []
    for b in BACKBONES:
        folder = figures / b
        sources = {"baseline": (root/b/"full/baseline_test_predictions.jsonl", root/b/"full/baseline_test_metrics.json")}
        sources.update({m: (root/b/m/"test_predictions.jsonl", root/b/m/"test_metrics.json") for m in ("full", "frozen")})
        series = render_group(b, methods, labels, sources, folder, "full_frozen_comparison",
            "Val-selected best | native batch 8 / 32 | main E1 source checkpoints | mAP without presence gating")
        for m in methods:
            points = [series[m]["grid"][i] for i in (40,50,60)]
            metric = series[m]["metrics"]
            best = read(root/b/m/"COMPLETE.json").get("best_epoch") if m != "baseline" else None
            record = dict(backbone=b, method=labels[m], best_epoch=best, mAP=metric["mAP"],
                **{f"g1_{i}": series[m]["grid"][i]["gmiou_1"] for i in (40,50,60)},
                g1_mean=sum(p["gmiou_1"] for p in points)/3, g3_mean=sum(p["gmiou_3"] for p in points)/3, AUROC=metric["AUROC"])
            summaries.append(record)
            lines.append(f"| {plotting.NAMES[b]} | {labels[m]} | {best or '—'} | {metric['mAP']:.2f} | " + " | ".join(f"{p['gmiou_1']:.4f}" for p in points) + f" | {record['g1_mean']:.4f} | {record['g3_mean']:.4f} | {metric['AUROC']:.2f} |")
        local = f"# {plotting.NAMES[b]}：full / frozen\n\n![Test threshold comparison](full_frozen_comparison.png)\n\n![G-mIoU@1 and @3](full_frozen_comparison_gmiou_1_3.png)\n\n![Val selection](validation_selection.png)\n\n[101点CSV](full_frozen_comparison_sweep.csv) · [PDF](full_frozen_comparison.pdf)\n"
        (folder/"README.md").write_text(local)
    lines += ["", "全量EaTR/QD的Test三点均值优于冻结；Moment/CG冻结更高。全量Moment的mAP下降，AUROC多数略降，不能将固定阈值的G-mIoU提升解释为所有能力均提升。", ""]
    for b in BACKBONES:
        lines += [f"## {plotting.NAMES[b]}", "", f"![{b} Test]({b}/full_frozen_comparison.png)", "", f"![{b} Val]({b}/validation_selection.png)", ""]
    (figures/"README.md").write_text("\n".join(lines))
    csv_write(figures/"test_summary.csv", summaries)
    dump(figures/"TEST_FIGURE_AUDIT.json", dict(status="PASSED", backbones=4, methods=3, thresholds_per_series=101,
        full_domain_0_1=True, fixed_points_match_official=True, no_test_selection=True))


def protocol_study():
    FIGURES.mkdir(parents=True, exist_ok=True)
    queue = read(OUTPUT/"QUEUE.json")
    lines = ["# Baseline / Multi / Shared：统一 Val 选模协议", "",
        "Multi与Shared各运行全量和冻结两种方案，共16组。另补同源Baseline的两种微调方案8组，合计24组，用于统一四曲线对照。", "",
        "各方案从对应已完成的原路线best独立初始化，GEQ seed2023。Val G-mIoU@1在0.4/0.5/0.6的均值选best，连续超过5轮（6轮）无提升早停，上限20轮；Test仅评估选定best。full保留原GMR+O2M损失并加GEQ BCE，原模型lr1e-5、GEQ lr3e-5；frozen仅GEQ BCE，原模型保持eval。", "",
        "每任务num_workers=2，Moment batch8，其余batch32，无梯度累积，AMP关闭。每GPU至多两个CUDA任务，计入已有训练。旧固定4轮结果单独保留，不冒充新选模结果。", "",
        "全量和冻结成对初始化相同。每骨干四条曲线分别为原GMR、GMR+GEQ、Multi+GEQ、Shared+GEQ；两种训练方式各一套。原GMR与路线的既有预训练预算仍存在差异，完整方法比较不等于单因素因果消融。", "",
        f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}", "",
        "| Backbone | 路线 | 模式 | 状态 | 完成Epoch | Best Epoch | Best Val均值 | Test mAP (%) | Test G1均值 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|"]
    summary = []
    for b in BACKBONES:
        folder = FIGURES/b; folder.mkdir(parents=True, exist_ok=True)
        for v in VARIANTS:
            for m in MODES:
                key=f"{b}/{v}/{m}"; root=OUTPUT/key
                complete=read(root/"COMPLETE.json"); items=history(root/"history.jsonl")
                state=queue.get("jobs",{}).get(key,{})
                status=complete.get("status",state.get("status","QUEUED"))
                best=complete.get("selection",items[-1]["selection"] if items else {})
                test=read(root/"test_metrics.json") if complete else {}
                map_value=test.get("0.4",{}).get("brief",{}).get("mAP")
                mean=sum(test[t]["brief"]["G-mIoU@1"]/100 for t in ("0.4","0.5","0.6"))/3 if test else None
                record=dict(backbone=b,variant=v,mode=m,status=status,epoch=items[-1]["epoch"] if items else None,
                    best_epoch=best.get("best_epoch"),best_val_mean=best.get("best_mean"),mAP=map_value,test_g1_mean=mean)
                summary.append(record)
                fmt=lambda x,p=4: "—" if x is None else f"{x:.{p}f}"
                lines.append(f"| {plotting.NAMES[b]} | {v} | {m} | {status} | {record['epoch'] or '—'} | {record['best_epoch'] or '—'} | {fmt(record['best_val_mean'])} | {fmt(map_value,2)} | {fmt(mean)} |")
        # Render completed modes only; never fill a missing curve with an old model.
        per_bone = [f"# {plotting.NAMES[b]}：统一 Val 选模", ""]
        for variant in VARIANTS:
            roots = {m: OUTPUT/b/variant/m for m in MODES}
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7.5,4.5), layout="constrained")
            plotted = False
            baseline_drawn = False
            for mode,color,marker in (("full","#0072B2","o"),("frozen","#D55E00","s")):
                items=history(roots[mode]/"history.jsonl")
                if not items:
                    continue
                plotted=True
                ax.plot([i["epoch"] for i in items],[i["selection"]["mean"] for i in items],
                        label=mode,color=color,marker=marker,markersize=3)
                if not baseline_drawn:
                    ax.axhline(items[0]["selection"]["baseline_mean"],label="Original route",color="#64748B",linestyle="--")
                    baseline_drawn=True
            ax.set(xlabel="Fine-tuning epoch",ylabel="Val mean G-mIoU@1 (0–1)",ylim=(0,1),
                   title=plotting.NAMES[b]+" · "+variant+" · Val checkpoint selection")
            ax.grid(alpha=.2)
            if plotted:
                ax.legend()
                plotting.save(fig,folder/(variant+"_validation_selection"))
                per_bone += [f"## {variant}：Val 选模曲线","",f"![{variant} Val]({variant}_validation_selection.png)",""]
            else:
                plt.close(fig)
            if variant == "baseline" or not all(read(p/"COMPLETE.json").get("status")=="COMPLETED" for p in roots.values()):
                continue
            methods=("baseline","full","frozen")
            labels=dict(baseline=variant+" checkpoint",full=variant+" + GEQ (full)",frozen=variant+" + GEQ (frozen)")
            sources={"baseline":(roots["full"]/"baseline_test_predictions.jsonl",roots["full"]/"baseline_test_metrics.json")}
            sources.update({m:(roots[m]/"test_predictions.jsonl",roots[m]/"test_metrics.json") for m in MODES})
            name=variant+"_full_frozen"
            render_group(b,methods,labels,sources,folder,name,"Val-selected best | same source route | full vs frozen GEQ")
            for dest in (lines,per_bone):
                dest += ["",f"## {plotting.NAMES[b]} · {variant} · full / frozen","",f"![{variant} comparison]({b+'/' if dest is lines else ''}{name}.png)",""]
        for mode in MODES:
            roots={v: OUTPUT/b/v/mode for v in VARIANTS}
            if not all(read(p/"COMPLETE.json").get("status")=="COMPLETED" for p in roots.values()):
                lines += ["",f"{plotting.NAMES[b]} {mode}：四曲线等待同源Baseline、Multi、Shared全部完成Test。"]
                continue
            methods=("baseline","baseline_geq","multi_geq","shared_geq")
            labels=dict(baseline="GMR baseline",baseline_geq="GMR + GEQ",multi_geq="multi + GEQ",shared_geq="shared + GEQ")
            sources={"baseline": (roots["baseline"]/"baseline_test_predictions.jsonl",roots["baseline"]/"baseline_test_metrics.json")}
            sources.update({v+"_geq": (roots[v]/"test_predictions.jsonl",roots[v]/"test_metrics.json") for v in VARIANTS})
            render_group(b,methods,labels,sources,folder,mode+"_comparison",f"{mode} GEQ | Val-selected best | source route checkpoints | mAP without presence gating")
            for dest in (lines,per_bone):
                dest += ["",f"## {plotting.NAMES[b]} · {mode}","",f"![{b} {mode}]({b+'/' if dest is lines else ''}{mode}_comparison.png)",""]
        (folder/"README.md").write_text("\n".join(per_bone)+"\n")
    (FIGURES/"README.md").write_text("\n".join(lines)+"\n")
    dump(OUTPUT/"summary.json",summary);csv_write(OUTPUT/"summary.csv",summary)
    dump(FIGURES/"summary.json",summary);csv_write(FIGURES/"summary.csv",summary)
    print("PROTOCOL_REPORT",sum(s["status"]=="COMPLETED" for s in summary),"/",len(summary),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--baseline-study",action="store_true")
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/"report.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        baseline_study() if args.baseline_study else protocol_study()
