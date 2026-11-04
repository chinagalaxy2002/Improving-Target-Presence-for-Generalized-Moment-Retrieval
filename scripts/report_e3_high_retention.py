#!/usr/bin/env python3
"""Recompute the full 11-backbone retention comparison from portable predictions.

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

NAMES = ('moment_detr', 'eatr', 'qd_detr', 'cg_detr', 'sim_detr', 'bm_detr',
         'bam_detr', 'uvcom', 'tr_detr', 'taskweave', 'ld_detr')
LABELS = ('Moment-DETR', 'EaTR', 'QD-DETR', 'CG-DETR', 'Sim-DETR', 'BM-DETR',
          'BAM-DETR', 'UVCOM', 'TR-DETR', 'TaskWeave', 'LD-DETR')
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
            assert len(group) == len(NAMES)
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
    assert set(NAMES) == {r['backbone'] for r in original}
    for (name, stage, split), score in mAP.items():
        ref = next(r for r in original if (r['backbone'], r['stage'], r['split']) == (name, stage, 'test_' + split))
        assert score == float(ref['mAP'])
    audit = dict(passed=True, comparison_rows=len(rows), official_G1_G3_checks=checks,
                 mAP_checks=len(mAP), prediction_files=6*len(NAMES), backbones=list(NAMES),
                 thresholds_selected_on='Standard Val only', test_used_for_selection=False,
                 all_published_backbones_included=True, subset_refiltered_on_current_test=False,
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
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    upper = max(max(r['gmr_G1'], r['geq_full_G1']) for r in rows)
    limit = 5 * np.ceil((upper + 8) / 5)
    labels = [label + ('*' if name == 'moment_detr' else '') for name, label in zip(NAMES, LABELS)]
    plotted = []
    for search, stem, subtitle in (
        ('grid_101', 'PRIMARY', 'A | 0.01 threshold grid'),
        ('val_breakpoints', 'EXACT_THRESHOLDS', 'B | All Val score breakpoints'),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 9), sharey=True)
        fig.subplots_adjust(left=.15, right=.98, top=.80, bottom=.18, wspace=.22)
        fig.suptitle('High event retention | All 11 backbones', fontsize=17, fontweight='bold', y=.975)
        fig.text(.56, .924, subtitle, ha='center', fontsize=12)
        handles = [Line2D([], [], color=c, marker=marker, linestyle='none', markersize=8, label=label)
                   for c, marker, label in [('#a6a6a6', 's', 'GMR baseline'), ('#0072b2', 'o', 'GEQ + aux')]]
        fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.56, .904))
        for ax, split, title in zip(axes, ('standard', 'full'), ('Standard Test (1,036)', 'Full-Decodable Test (2,970)')):
            group = [next(r for r in rows if (r['search'], r['split'], r['backbone']) == (search, split, n)) for n in NAMES]
            y = np.arange(len(NAMES))
            for i, stage in enumerate(('gmr', 'geq_full')):
                values = np.array([r[stage+'_G1'] for r in group])
                centers = y + (-.18 if i == 0 else .18)
                color, edge, marker = ('#a6a6a6', '#555555', 's') if i == 0 else ('#0072b2', '#003f63', 'o')
                bars = ax.barh(centers, values, height=.30, color=color, edgecolor=edge, linewidth=.5, zorder=2)
                assert np.array_equal(np.array([bar.get_width() for bar in bars]), values)
                ax.scatter(values, centers, marker=marker, s=14, color=edge, zorder=3)
                for n, value, center in zip(NAMES, values, centers):
                    ax.text(value + limit*.015, center, f'{value:.2f}', va='center', fontsize=9.5, color='#333333')
                    plotted.append(dict(search=search, split=split, backbone=n, stage=stage,
                                        G1=float(value), label=f'{value:.2f}'))
            ax.set_yticks(y, labels)
            ax.set_xlim(0, limit)
            ax.set_xlabel('G-mIoU@1 (0–100; higher is better)', fontsize=10)
            ax.set_title(title, fontsize=12, pad=30)
            ax.set_axisbelow(True)
            ax.grid(axis='x', alpha=.18)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0)
            mean = next(r for r in summary if (r['search'], r['split']) == (search, split))
            ax.text(.5, 1.015, f"Mean gain {mean['delta_pp']:+.2f} pp | Improved {mean['positive_backbones']}/{len(NAMES)}",
                    transform=ax.transAxes, ha='center', fontsize=9.5, color='#475569')
        axes[0].set_ylim(len(NAMES)-.48, -.55)
        footer = fig.text(.56, .044, 'Each method: largest Val threshold retaining ≥95% of positive queries; fixed on both Tests.\n'
                         '* Moment-DETR results reused. GMR baseline is auxiliary-trained Stage 1. All backbones retained.',
                         ha='center', fontsize=8.8, linespacing=1.6, color='#475569')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            for label in list(ax.texts) + list(ax.get_yticklabels()) + [ax.title, ax.xaxis.label]:
                if label.get_visible():
                    bounds = label.get_window_extent(renderer)
                    assert bounds.x0 >= fig.bbox.x0 and bounds.x1 <= fig.bbox.x1
                    assert bounds.y0 >= fig.bbox.y0 and bounds.y1 <= fig.bbox.y1
            assert not ax.xaxis.label.get_window_extent(renderer).overlaps(footer.get_window_extent(renderer))
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(root / (stem+'.'+ext), dpi=200)
        plt.close(fig)
    assert len(plotted) == 8*len(NAMES)
    (root/'PLOT_DATA.json').write_text(json.dumps(plotted, indent=2)+'\n')


def document(root, rows, summary):
    count = len(NAMES)
    def selected(search, split):
        return next(r for r in summary if (r['search'], r['split']) == (search, split))
    lines = ['# 高事件保留率场景：全部11骨干配对评测', '',
             '比较本轮 **GMR baseline → GEQ+aux**。GMR baseline 是带 Balanced BCE + 0.2 Dice 辅助监督的 Stage 1；GEQ+aux 是同一骨干的 Stage 2。Moment-DETR复用已完成实验，全部模型保持既有 Val best，没有重新训练或重选 checkpoint。', '',
             '本报告包含 Moment-DETR、EaTR、QD-DETR、CG-DETR、Sim-DETR、BM-DETR、BAM-DETR、UVCOM、TR-DETR、TaskWeave、LD-DETR **全部11个骨干**，每个都报告 Standard 和 Full。范围由本轮实验清单确定，保留所有提高和下降结果。早期六骨干版本保留在 Git 历史，当前主结果不再限定该子集。', '',
             '## 阈值规则与柱状图 A/B 的区别', '',
             '对每个骨干、每种方法，分别在 Standard Val（465条，255正/210空）选择正查询保留率≥95%的最大候选阈值，再固定用于 Standard / Full Test。严格采用 `score > threshold`。95%只约束 Val，不保证 Test 同样达到95%。', '',
             '**图 A 与图 B 使用相同模型，每张都是左侧 Standard、右侧 Full；唯一变化是 Val 上允许使用的阈值候选。**', '',
             '| 项目 | 图 A：0.01网格主结果 | 图 B：Val分数断点复核 |',
             '|---|---|---|',
             '| 候选阈值 | 0.00、0.01、0.02…1.00 | 全部 Val 分数，加0/1 |',
             '| 选阈值目标 | 正查询保留率≥95%的最大候选阈值 | 相同 |',
             '| 用途 | 沿用参考分析的网格规则 | 检查对0.01阈值粒度的敏感性 |', '',
             '例如 Moment-DETR 的 GMR baseline 在图 A 使用0.42（Val保留率100%），图 B 使用约0.428048（95.29%）。更细的阈值候选会改变保留与拒绝的查询，但不保证 Test G1 更高。精确复核遍历完整 Val 分数断点，并非连续实数域优化。Test 不参与阈值选择。', '',
             '常规11骨干汇总的 G1 使用固定0.4/0.5/0.6均值，本节使用 Val 选出的高保留率阈值，属于不同评测设置。mAP不使用存在性门控，在图 A、B中相同。', '']
    for search, stem, title in [('grid_101', 'PRIMARY', '图 A：0.01网格主结果'),
                                ('val_breakpoints', 'EXACT_THRESHOLDS', '图 B：Val分数断点复核')]:
        a, b = selected(search, 'standard'), selected(search, 'full')
        lines += ['## '+title, '',
                  f"Standard / Full 的11骨干等权平均 G1 分别变化 **{a['delta_pp']:+.3f} / {b['delta_pp']:+.3f}个百分点**，提高骨干数为 **{a['positive_backbones']}/{count} / {b['positive_backbones']}/{count}**。", '',
                  f'![{title}：全部11骨干]({stem}.png)', '',
                  f'[PDF]({stem}.pdf) · [SVG]({stem}.svg)', '',
                  '| 骨干 | Standard G1：baseline→GEQ+aux | Δ百分点 | Full G1：baseline→GEQ+aux | Δ百分点 |',
                  '|---|---:|---:|---:|---:|']
        for name, label in zip(NAMES, LABELS):
            cells = []
            for split in ('standard', 'full'):
                row = next(r for r in rows if (r['search'], r['split'], r['backbone']) == (search, split, name))
                cells += [f"{row['gmr_G1']:.3f}→{row['geq_full_G1']:.3f}", f"{row['delta_pp']:+.3f}"]
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## 汇总', '',
              '| 阈值规则 | Test | baseline平均G1 | GEQ+aux平均G1 | Δ百分点 | 相对增益 | 提高骨干数 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in summary:
        search = '0.01网格' if row['search'] == 'grid_101' else 'Val分数断点'
        lines.append(f"| {search} | {row['split']} | {row['mean_GMR']:.3f} | {row['mean_GEQ']:.3f} | {row['delta_pp']:+.3f} | {row['relative_gain_percent']:+.2f}% | {row['positive_backbones']}/{count} |")
    lines += ['', '先等权平均全部11个骨干的原始 G1，再计算相对增益；没有用 Test 结果筛选骨干。', '',
              '## 全骨干 mAP / G1 / G3（图 A 的固定阈值）', '',
              '各项为 GMR baseline → GEQ+aux，单位0–100；mAP在两种阈值规则下相同，图 B 的 G1/G3 完整值见 METRICS.csv。', '',
              '| 骨干 | Test | mAP | G1 | G3 |', '|---|---|---:|---:|---:|']
    for name, label in zip(NAMES, LABELS):
        for split in ('standard', 'full'):
            row = next(r for r in rows if (r['search'], r['split'], r['backbone']) == ('grid_101', split, name))
            cells = [f"{row['gmr_'+metric]:.2f}→{row['geq_full_'+metric]:.2f}" for metric in ('mAP', 'G1', 'G3')]
            lines.append('| '+label+' | '+split+' | '+' | '.join(cells)+' |')
    lines += ['', '## 实际保留率与空查询误返回率', '',
              '以下为图 A 下全部11骨干等权平均；逐骨干、逐方法、两种规则的阈值与实际保留率见 DECISIONS.csv。高保留率不代表低空查询误返回率，也不等于定位正确率。', '',
              '| Test | baseline正查询保留率 | GEQ+aux正查询保留率 | baseline空查询误返回率 | GEQ+aux空查询误返回率 |',
              '|---|---:|---:|---:|---:|']
    for row in summary:
        if row['search'] == 'grid_101':
            lines.append(f"| {row['split']} | {row['mean_gmr_test_TPR']:.2f}% | {row['mean_geq_full_test_TPR']:.2f}% | {row['mean_gmr_test_FPR']:.2f}% | {row['mean_geq_full_test_FPR']:.2f}% |")
    lines += ['', 'Standard Test为1036条，Full-Decodable为2970/2986条；两套Test有185个视频ID、186个视频—文本输入重叠。单 seed=2023，不作独立重复验证或多seed稳定性声明。', '',
              '## 数据与复算', '',
              '[逐模型指标](METRICS.csv) · [汇总](SUMMARY.csv) · [阈值与保留率](DECISIONS.csv) · [检查点与预测哈希](EVIDENCE.json) · [数值审计](AUDIT.json) · [图中原值](PLOT_DATA.json)', '',
              '本包包含22份 Val 压缩预测，并引用 ../e3_eleven 中44份 Test 压缩预测，合计66份；同时提供GT快照。无需权重、特征或GPU，即可核验完整查询覆盖、checkpoint绑定及文件哈希。176个G1/G3值与官方实现核验；44个mAP与既有11骨干报告交叉核验。', '',
              '```bash', 'PYTHONPATH=src python scripts/report_e3_high_retention.py',
              'PYTHONPATH=src python scripts/report_e3_high_retention.py --verify-only', '```', '']
    if (root/'STANDARD_TEST.png').exists() and (root/'FULL_TEST.png').exists():
        position = lines.index('## 阈值规则与柱状图 A/B 的区别')
        lines[position:position] = [
            '## 按 Test 集区分的参考样式曲线图', '',
            '每张曲线图包含全部11骨干，左列G1、右列G3；菱形为0.01网格下的Val高保留率工作点。行下方ΔG1/G3对应各自工作点，并非0.4/0.5/0.6均值。', '',
            '[Standard Test 图](STANDARD_TEST.png) · [Full-Decodable Test 图](FULL_TEST.png) · [两图PDF](BOTH_TESTS.pdf)', '',
            '[Standard每骨干单页PDF](STANDARD_BY_BACKBONE.pdf) · [Full每骨干单页PDF](FULL_BY_BACKBONE.pdf) · [完整曲线与工作点数据](CURVE_PLOT_DATA.json)', '',
            'Oracle虚线用GT存在性及各方法固定候选计算，只作分析参考。曲线纵轴为0–1单位；下文柱状图A/B为0–100单位。', '',
            '[Oracle原值](ORACLE.csv) · [曲线图审计](CURVE_FIGURE_AUDIT.json)', '',
        ]
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
    audit['plot_data_sha256'] = sha(args.root/'PLOT_DATA.json')
    audit['bars_and_labels_verified'] = 8*len(NAMES)
    (args.root/'AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
