#!/usr/bin/env python3
"""Recompute the current six-backbone retention comparison from portable predictions.

Uses the historical report's threshold candidates and strict score > threshold
semantics. No training, checkpoint reselection, or Test threshold optimization.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

NAMES = ('moment_detr', 'sim_detr', 'bm_detr', 'uvcom', 'taskweave', 'ld_detr')
LABELS = ('Moment-DETR', 'Sim-DETR', 'BM-DETR', 'UVCOM', 'TaskWeave', 'LD-DETR')
TICKS = ('Moment-\nDETR', 'Sim-\nDETR', 'BM-\nDETR', 'UVCOM', 'Task-\nWeave', 'LD-\nDETR')
SEARCHES = ('grid_101', 'val_breakpoints')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_evidence(root, meta):
    path = root / meta['path']
    assert sha(path) == meta['sha256'], path
    raw = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == meta['uncompressed_sha256']
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == meta['rows']
    return rows


def ordered(predictions, truth):
    indexed = {p['qid']: p for p in predictions}
    assert len(indexed) == len(predictions) == len(truth)
    assert set(indexed) == {g['qid'] for g in truth}
    rows = [indexed[g['qid']] for g in truth]
    assert all(p['vid'] == g['vid'] and p.get('query', g['query']) == g['query'] for p, g in zip(rows, truth))
    return rows


def select_threshold(scores, positive, search):
    # Match report_high_retention.py exactly: "exact" means all observed Val
    # score breakpoints (plus 0/1), not a continuous real-valued optimization.
    thresholds = np.arange(101) / 100 if search == 'grid_101' else np.unique(np.r_[0., scores, 1.])
    rates = (scores[positive, None] > thresholds[None, :]).mean(0)
    feasible = np.flatnonzero(rates >= .95 - 1e-12)
    assert len(feasible), '95% positive retention is infeasible on Val'
    i = feasible[-1]
    return float(thresholds[i]), float(100 * rates[i])


def compute(root):
    from soccergmr.evaluation.official.metrics import (
        _clean_pred_windows, _compute_set_iou_score, prepare_submission_for_gmiou,
        compute_G_mIoU, compute_mAP,
    )
    evidence = json.loads((root / 'EVIDENCE.json').read_text())
    assert set(evidence['models']) == set(NAMES)
    truth = {s: read_evidence(root, meta) for s, meta in evidence['ground_truth'].items()}
    assert [len(truth[s]) for s in ('val', 'standard', 'full')] == [465, 1036, 2970]
    positive = {s: np.array([bool(g['relevant_windows']) for g in rows]) for s, rows in truth.items()}
    pairs = {s: {(g['vid'], g['query']) for g in rows} for s, rows in truth.items()}
    assert not pairs['val'] & pairs['standard'] and not pairs['val'] & pairs['full']
    predictions, selected = {}, {}
    for name in NAMES:
        for stage in ('gmr', 'geq_full'):
            meta = evidence['models'][name]['methods'][stage]
            for split, info in meta['predictions'].items():
                assert info['checkpoint_sha256'] == meta['checkpoint_sha256']
                pred = ordered(read_evidence(root, info), truth[split])
                scores = np.array([p['pred_exist_score'] for p in pred])
                assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
                predictions[name, stage, split] = pred
            val_scores = np.array([p['pred_exist_score'] for p in predictions[name, stage, 'val']])
            for search in SEARCHES:
                selected[name, stage, search] = select_threshold(val_scores, positive['val'], search)
    # All thresholds are now fixed; only then compute Test outcomes.
    rows, checks = [], 0
    mAP = {}
    for name in NAMES:
        for stage in ('gmr', 'geq_full'):
            for split in ('standard', 'full'):
                mAP[name, stage, split] = compute_mAP(predictions[name, stage, split], truth[split], num_workers=1)['mAP']
    for search in SEARCHES:
        for split in ('standard', 'full'):
            y = positive[split]
            for name in NAMES:
                row = dict(search=search, split=split, backbone=name)
                for stage in ('gmr', 'geq_full'):
                    pred = predictions[name, stage, split]
                    threshold, val_tpr = selected[name, stage, search]
                    scores = np.array([p['pred_exist_score'] for p in pred])
                    accept = scores > threshold
                    gated, _ = prepare_submission_for_gmiou(pred, cls_threshold=threshold)
                    official = compute_G_mIoU(gated, truth[split], k_list=(1, 3))
                    for k in (1, 3):
                        raw = np.array([_compute_set_iou_score(
                            [w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'], max_pred_windows=10)[:k]],
                            g['relevant_windows']) for p, g in zip(pred, truth[split])])
                        metric = 100 * float(np.where(accept, raw, (~y).astype(float)).mean())
                        assert abs(round(metric, 2) - official[f'G-mIoU@{k}']) < 1e-9
                        row[f'{stage}_G{k}'] = metric
                        checks += 1
                    row.update({stage + '_threshold': threshold, stage + '_val_TPR': val_tpr,
                                stage + '_test_TPR': 100 * float(accept[y].mean()),
                                stage + '_test_FPR': 100 * float(accept[~y].mean()),
                                stage + '_mAP': mAP[name, stage, split]})
                row['delta_pp'] = row['geq_full_G1'] - row['gmr_G1']
                row['relative_gain_percent'] = 100 * row['delta_pp'] / row['gmr_G1']
                rows.append(row)
    summary = []
    for search in SEARCHES:
        for split in ('standard', 'full'):
            group = [r for r in rows if r['search'] == search and r['split'] == split]
            a, b = [float(np.mean([r[stage + '_G1'] for r in group])) for stage in ('gmr', 'geq_full')]
            record = dict(search=search, split=split,mean_GMR=a, mean_GEQ=b, delta_pp=b-a,
                          relative_gain_percent=100*(b-a)/a, positive_backbones=sum(r['delta_pp'] > 0 for r in group))
            for stage in ('gmr', 'geq_full'):
                for metric in ('test_TPR', 'test_FPR'):
                    record[f'mean_{stage}_{metric}'] = float(np.mean([r[f'{stage}_{metric}'] for r in group]))
            summary.append(record)
    # Also cross-check ungated mAP against the independently audited 11-model report.
    with (root.parent / 'e3_eleven/METRICS.csv').open() as stream:
        original = list(csv.DictReader(stream))
    for (name, stage, split), score in mAP.items():
        ref = next(r for r in original if (r['backbone'], r['stage'], r['split']) == (name, stage, 'test_' + split))
        assert score == float(ref['mAP'])
    audit = dict(passed=True, comparison_rows=len(rows), official_G1_G3_checks=checks,
                 mAP_checks=len(mAP), prediction_files=36, backbones=list(NAMES),
                 thresholds_selected_on='Standard Val only', test_used_for_selection=False,
                 subset_fixed_from_historical_reference=True, subset_refiltered_on_current_test=False,
                 new_training=False, new_checkpoint_inference=False, checkpoint_selection='Existing finalized bests',
                 script_sha256=sha(__file__), evidence_sha256=sha(root/'EVIDENCE.json'),
                 standard_full_overlap_video_query_pairs=len(pairs['standard'] & pairs['full']),
                 standard_full_overlap_video_ids=len({g['vid'] for g in truth['standard']} & {g['vid'] for g in truth['full']}))
    return rows, summary, audit


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def plot(root, rows, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    upper = max(max(r['gmr_G1'], r['geq_full_G1']) for r in rows)
    ymax = 5 * np.ceil((upper + 5) / 5)
    for search, stem in (('grid_101', 'PRIMARY'), ('val_breakpoints', 'EXACT_THRESHOLDS')):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True)
        for ax, split, title in zip(axes, ('standard', 'full'), ('Standard Test (1,036)', 'Full-Decodable Test (2,970)')):
            group = [next(r for r in rows if (r['search'], r['split'], r['backbone']) == (search, split, n)) for n in NAMES]
            x = np.arange(6)
            a, b = [np.array([r[stage+'_G1'] for r in group]) for stage in ('gmr', 'geq_full')]
            ax.bar(x-.18, a, width=.34, color='#a6a6a6', edgecolor='#555555', linewidth=.6, label='GMR baseline')
            ax.bar(x+.18, b, width=.34, color='#0072b2', edgecolor='#003f63', linewidth=.6, label='GEQ + aux')
            ax.scatter(x-.18, a, marker='s', s=13, color='#444444', zorder=3)
            ax.scatter(x+.18, b, marker='o', s=13, color='#003f63', zorder=3)
            for i, r in enumerate(group):
                ax.text(i, max(a[i], b[i])+.6, f"{r['delta_pp']:+.2f}", ha='center', fontsize=8)
            ax.set_xticks(x, TICKS)
            ax.set_ylim(0, ymax)
            ax.set_ylabel('G-mIoU@1 (0–100)')
            ax.set_title(title)
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
            mean = next(r for r in summary if (r['search'], r['split']) == (search, split))
            ax.text(.02, .97, f"Mean gain: {mean['delta_pp']:+.2f} pp ({mean['relative_gain_percent']:+.2f}%)\nImproved: {mean['positive_backbones']}/6", transform=ax.transAxes, va='top', fontsize=8.5)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.005))
        fig.text(.5, .065, 'Each method: largest Val threshold retaining ≥95% of positive queries; fixed on both Tests.', ha='center', fontsize=8)
        fig.text(.5, .023, 'Current auxiliary-trained Stage 1 vs GEQ+aux · same six reference backbones · ' + ('0.01 threshold grid' if search == 'grid_101' else 'all Val score breakpoints'), ha='center', fontsize=8)
        fig.tight_layout(rect=[0, .11, 1, .94])
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(root / (stem+'.'+ext), dpi=200)
        plt.close(fig)


def document(root, rows, summary):
    lines = ['# 高事件保留率场景：本轮六骨干配对评测', '',
             '比较本轮 **GMR baseline → GEQ+aux**。GMR baseline 按用户要求命名，实际为带 Balanced BCE + 0.2 Dice 辅助监督的 Stage 1；不等同于历史无辅助 GMR。Moment-DETR 复用已完成实验，其余模型使用本轮固定源码。没有重新训练或重选 checkpoint。', '',
             '骨干范围事先沿用参考分支的 Moment-DETR、Sim-DETR、BM-DETR、UVCOM、TaskWeave、LD-DETR，未依据本次 Test 增益删选。原六骨干集合来自历史探索性分析，不能视为所有11骨干的普遍结论。', '',
             '## 阈值规则', '',
             '每个骨干、每种方法分别在 Standard Val（465条，255正/210空）选择正查询保留率 ≥95% 的最大候选阈值，然后固定用于两套 Test。严格采用 `pred_exist_score > threshold`。主分析搜索0–1、步长0.01；精确复核候选为全部 Val 分数断点及0/1，与参考脚本一致（不是连续实数域优化）。Test 不参与阈值或模型选择。', '',
             '保留率只表示事件查询通过存在性门控，不代表定位正确率。mAP 不使用门控，两种阈值搜索的 mAP 完全相同。', '',
             '## 两张图应该怎么读', '',
             '**两张图比较同一批模型与 checkpoint，每张都是左侧 Standard、右侧 Full。唯一改变的是在 Val 上选择存在性阈值时，允许使用哪些候选值。**', '',
             '| 项目 | 图 A：0.01 网格主结果 | 图 B：Val 分数断点复核 |',
             '|---|---|---|',
             '| 候选阈值 | 0.00、0.01、0.02…1.00 | Val 中实际出现的全部分数，加0/1 |',
             '| 选阈值目标 | Val 正查询保留率≥95%的最大候选阈值 | 相同 |',
             '| 模型、数据及评分规则 | 同一批固定 checkpoint、两套 Test、严格 score > τ | 相同 |',
             '| 用途 | 与原分支保持相同网格的主结果 | 检查结论是否依赖0.01这一网格粒度 |', '',
             '例如，Moment-DETR 的 GMR baseline 在图 A 使用 τ=0.42（Val 保留率100%），图 B 使用 τ≈0.428048（Val 保留率95.29%）。图 B 可以在0.42和0.43之间选择，因而可能拒绝更多空查询，也可能多拒绝部分正查询，最终 Test G1 会变化。更细的搜索不保证 Test G1 更高；这里优化的是 Val 保留率约束下的阈值，并非 Test G1。完整精度见 DECISIONS.csv。', '',
             '正文的常规11骨干汇总使用固定0.4/0.5/0.6三阈值均值，本节采用“Val 保留率≥95%”选出的阈值，回答不同的使用场景，分数不应直接混为同一评测设置。', '',
             '## 图 A：0.01 网格主结果', '',
             '主分析中 Standard / Full 都有5/6骨干提高，LD-DETR下降。每个方法的阈值只在 Val 上确定。', '', '![图A：0.01网格主结果](PRIMARY.png)', '',
             '[PDF](PRIMARY.pdf) · [SVG](PRIMARY.svg)', '',
             '| 骨干 | Standard GMR→GEQ+aux | Δ百分点 | Full GMR→GEQ+aux | Δ百分点 |',
             '|---|---:|---:|---:|---:|']
    for name, label in zip(NAMES, LABELS):
        cells = []
        for split in ('standard', 'full'):
            r = next(r for r in rows if (r['search'], r['split'], r['backbone']) == ('grid_101', split, name))
            cells += [f"{r['gmr_G1']:.3f}→{r['geq_full_G1']:.3f}", f"{r['delta_pp']:+.3f}"]
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 图 B：Val 分数断点复核（同一模型，更细的阈值候选）', '',
              '复核中两套 Test 仍为5/6骨干提高，但平均增益缩小，LD-DETR下降更明显。这张图用于展示对阈值粒度的敏感性，不是一轮新训练。', '', '![图B：Val分数断点复核](EXACT_THRESHOLDS.png)', '',
              '[PDF](EXACT_THRESHOLDS.pdf) · [SVG](EXACT_THRESHOLDS.svg)', '',
              '| 阈值搜索 | Test | GMR平均 | GEQ+aux平均 | Δ百分点 | 相对增益 | 提高骨干数 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['search']} | {r['split']} | {r['mean_GMR']:.3f} | {r['mean_GEQ']:.3f} | {r['delta_pp']:+.3f} | {r['relative_gain_percent']:+.2f}% | {r['positive_backbones']}/6 |")
    lines += ['', '汇总先等权平均六骨干原始 G1，再计算相对增益。图和表保留下降结果，不沿用旧分支“六个均提高”的结论。', '',
              '## Test 实际保留率及空查询误返回率', '',
              '95%只约束 Val，Test 保留率允许低于95%。以下为0.01网格下六骨干等权平均，完整逐模型值及阈值见 DECISIONS.csv。', '',
              '| Test | GMR正查询保留率 | GEQ+aux正查询保留率 | GMR空查询误返回率 | GEQ+aux空查询误返回率 |',
              '|---|---:|---:|---:|---:|']
    for r in summary:
        if r['search'] == 'grid_101':
            lines.append(f"| {r['split']} | {r['mean_gmr_test_TPR']:.2f}% | {r['mean_geq_full_test_TPR']:.2f}% | {r['mean_gmr_test_FPR']:.2f}% | {r['mean_geq_full_test_FPR']:.2f}% |")
    lines += ['', 'Standard Test 1036条，Full-Decodable 2970/2986条；两个 Test 有重叠（185个视频ID、186个视频—文本输入），不能当作独立重复。单 seed=2023，不作统计显著性声明。主图关注高保留率场景，不能由此推断默认0.4/0.5/0.6门控或全部场景均改善。', '',
              '## 数据与复算', '',
              '[逐模型 mAP/G1/G3](METRICS.csv) · [汇总](SUMMARY.csv) · [阈值/保留率](DECISIONS.csv) · [来源与哈希](EVIDENCE.json) · [数值审计](AUDIT.json)', '',
              '36份预测由12份本目录 Val 压缩预测及24份 ../e3_eleven 中 Test 压缩预测组成，同时提供 GT 快照。不需模型权重、特征或 GPU。逐文件核验哈希、覆盖与 checkpoint 绑定；96个 G1/G3 值与官方实现核验，24个 mAP 与11骨干报告独立交叉核验。', '',
              '```bash', 'PYTHONPATH=src python scripts/report_e3_high_retention.py',
              'PYTHONPATH=src python scripts/report_e3_high_retention.py --verify-only', '```', '']
    (root/'README.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/e3_high_retention'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    rows, summary, audit = compute(args.root)
    if args.verify_only:
        with (args.root/'METRICS.csv').open() as stream:
            expected = list(csv.DictReader(stream))
        assert len(expected) == len(rows)
        for row, ref in zip(rows, expected):
            for key, value in row.items():
                assert value == ref[key] if isinstance(value, str) else abs(value-float(ref[key])) < 1e-10
        print('VERIFIED', json.dumps(audit))
        return
    write_csv(args.root/'METRICS.csv', rows)
    write_csv(args.root/'SUMMARY.csv', summary)
    keys = ['search', 'split', 'backbone', 'gmr_threshold', 'geq_full_threshold', 'gmr_val_TPR',
            'geq_full_val_TPR', 'gmr_test_TPR', 'geq_full_test_TPR', 'gmr_test_FPR', 'geq_full_test_FPR']
    write_csv(args.root/'DECISIONS.csv', [{k: r[k] for k in keys} for r in rows])
    plot(args.root, rows, summary)
    document(args.root, rows, summary)
    audit['figures'] = {p.name: sha(p) for p in args.root.iterdir() if p.suffix in ('.png', '.svg', '.pdf')}
    (args.root/'AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
