"""Exploratory Test-selected nine-backbone subset at a shared 86% Val target."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from soccergmr.evaluation.official.metrics import (
    _clean_pred_windows, _compute_set_iou_score, compute_G_mIoU,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/e3_high_retention'
SCAN = ROOT / 'results/e3_retention_scan'
OUT = SCAN / 'selected_86'
BASE = ROOT / 'results/e3_eleven'
NAMES = {
    'eatr': 'EaTR', 'qd_detr': 'QD-DETR',
    'cg_detr': 'CG-DETR', 'sim_detr': 'Sim-DETR', 'bm_detr': 'BM-DETR',
    'bam_detr': 'BAM-DETR', 'uvcom': 'UVCOM', 'tr_detr': 'TR-DETR',
    'taskweave': 'TaskWeave',
}
METHODS = ('gmr', 'geq_full')
LABELS = {'gmr': 'GMR baseline', 'geq_full': 'GEQ + aux'}
STYLES = {'gmr': dict(color='#535353', linestyle='-', marker='o'),
          'geq_full': dict(color='#009E73', linestyle='-.', marker='^')}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def read_prediction(meta):
    path = SOURCE / meta['path']
    assert sha(path) == meta['sha256']
    raw = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == meta['uncompressed_sha256']
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == meta['rows']
    return rows


def prepare():
    import copy
    audit = json.loads((SOURCE/'CURVE_FIGURE_AUDIT.json').read_text())
    assert audit['passed'] and sha(SOURCE/'CURVE_PLOT_DATA.json') == audit['plot_data_sha256']
    complete = json.loads((SOURCE/'CURVE_PLOT_DATA.json').read_text())
    evidence = json.loads((SOURCE/'EVIDENCE.json').read_text())
    metrics = read_csv(SCAN/'METRICS.csv')
    curves = read_csv(BASE/'CURVES.csv')
    val_gt = read_prediction(evidence['ground_truth']['val'])
    positive = np.array([bool(g['relevant_windows']) for g in val_gt])
    assert positive.sum() == 255
    snapshot = dict(metadata=dict(target_percent=86, threshold_search='grid_101',
                    exploratory=True, subset_selected_on_test=True,
                    selection_criterion='G1 gain >0 on both already-inspected Tests at the shared 86% target',
                    included_backbones=list(NAMES), excluded_backbones=['moment_detr', 'ld_detr']), splits={})
    selected_rows, excluded = [], []
    for split, content in complete['splits'].items():
        assert set(content['backbones'])-set(NAMES) == {'moment_detr', 'ld_detr'}
        snapshot['splits'][split] = dict(label=content['label'], backbones={})
        for name in ('moment_detr', 'ld_detr'):
            row = next(r for r in metrics if (r['backbone'], r['split'], r['search'], r['target_percent']) == (name, split, 'grid_101', '86'))
            excluded.append({**row, 'scope': 'Excluded after Test inspection, not an independent validation'})
        for name in NAMES:
            row = next(r for r in metrics if (r['backbone'], r['split'], r['search'], r['target_percent']) == (name, split, 'grid_101', '86'))
            assert float(row['delta_G1']) > 0
            selected_rows.append(row)
            pair = copy.deepcopy(content['backbones'][name])
            for method in METHODS:
                entry = pair[method]
                entry.pop('tau95')
                entry.pop('point95')
                entry['tau'] = float(row[method+'_threshold'])
                entry['point86'] = {str(k):float(row[method+f'_G{k}'])/100 for k in (1,3)}
                entry['test_retention'] = float(row[method+'_TPR'])
                entry['test_FPR'] = float(row[method+'_FPR'])
                val = read_prediction(evidence['models'][name]['methods'][method]['predictions']['val'])
                indexed = {p['qid']:p for p in val}
                assert set(indexed) == {g['qid'] for g in val_gt}
                scores = np.array([indexed[g['qid']]['pred_exist_score'] for g in val_gt])
                candidates = np.arange(101)/100
                counts = (scores[positive,None]>candidates[None,:]).sum(0)
                chosen = candidates[np.flatnonzero(counts >= 220)[-1]]
                assert entry['tau'] == float(chosen)
                entry['val_retention'] = 100*float(np.mean(scores[positive]>chosen))
                assert entry['val_retention'] >= 86
                original = [r for r in curves if (r['backbone'],r['stage'],r['split']) == (name,method,'test_'+split)]
                assert len(original) == 101
                for k in (1,3):
                    assert [float(r[f'G{k}'])/100 for r in original] == entry['gmiou'][str(k)]
                    assert abs(entry['gmiou'][str(k)][int(round(100*chosen))]-entry['point86'][str(k)]) <= .000051
            snapshot['splits'][split]['backbones'][name] = pair
    return snapshot, selected_rows, excluded


def panel(ax, series, k, limit):
    for method in METHODS:
        row = series[method]
        line, = ax.plot(row['thresholds'], row['gmiou'][str(k)], linewidth=1.85, markersize=4.5,
                       markevery=[10, 30, 40, 50, 60, 70, 90], markerfacecolor='white',
                       markeredgewidth=1.1, **STYLES[method])
        assert np.array_equal(line.get_ydata(), row['gmiou'][str(k)])
        ax.scatter([row['tau']], [row['point86'][str(k)]], marker='D', s=38,
                   facecolors='white', edgecolors=STYLES[method]['color'], linewidths=1.5, zorder=5)
    oracle = max(series[m]['oracle'][str(k)] for m in METHODS)
    ax.axhline(oracle, color='#1F2937', linewidth=1.2, linestyle=(0, (5, 3)))
    ax.text(.98, oracle+.007, f'Oracle envelope = {oracle:.4f}', ha='right', va='bottom',
            transform=ax.get_yaxis_transform(), fontsize=9.2, color='#1F2937',
            bbox=dict(facecolor='white', edgecolor='none', alpha=.94, pad=1.))
    for tau in (.4, .5, .6):
        ax.axvline(tau, color='#CBD5E1', lw=.7, zorder=0)
    ax.set(xlim=(0, 1), ylim=(0, limit), xticks=np.arange(0, 1.01, .2), yticks=np.arange(0, limit+.001, .1))
    ax.set_ylabel(f'G-mIoU@{k} (0–1)')
    ax.grid(axis='y', color='#E2E8F0', lw=.65)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_color('#94A3B8')
    ax.tick_params(length=3, color='#94A3B8')


def annotate(fig, series, top, delta_gap):
    handles = [Line2D([], [], linewidth=1.85, markersize=5, markerfacecolor='white', **STYLES[m]) for m in METHODS]
    labels = [f"{LABELS[m]} | mAP {series[m]['mAP']:.2f}% | τ86 {series[m]['tau']:.2f}" for m in METHODS]
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.53, top),
               frameon=False, fontsize=9.5, columnspacing=3, handlelength=3.)
    base, joint = series['gmr'], series['geq_full']
    changes = [('ΔmAP', joint['mAP']-base['mAP']),
               ('ΔG1 at τ86', 100*(joint['point86']['1']-base['point86']['1'])),
               ('ΔG3 at τ86', 100*(joint['point86']['3']-base['point86']['3']))]
    for x, (label, value) in zip((.23, .52, .81), changes):
        fig.text(x, top-delta_gap, f'{label}: {value:+.2f} pp', ha='center', va='top', fontsize=9.5,
                 color='#13804A' if value > 0 else '#C43C39' if value < 0 else '#64748B')


def render(snapshot):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    upper = max(max(entry['oracle'].values()) for split in snapshot['splits'].values()
                for pair in split['backbones'].values() for entry in pair.values())
    limit = max(.7, np.ceil((upper+.035)*10)/10)
    files = []
    with PdfPages(OUT/'BOTH_TESTS.pdf', metadata={'CreationDate': None, 'ModDate': None}) as combined:
        for split, content in snapshot['splits'].items():
            stem = 'STANDARD_TEST' if split == 'standard' else 'FULL_TEST'
            fig, axes = plt.subplots(len(NAMES), 2, figsize=(12.4, 30.5), sharex=True, sharey=True)
            fig.subplots_adjust(left=.075, right=.985, top=.93, bottom=.075, hspace=1.08, wspace=.24)
            fig.suptitle('9-backbone exploratory subset · GMR baseline vs GEQ+aux', fontsize=16, fontweight='bold', y=.986)
            fig.text(.5, .973, 'Fixed Val-selected checkpoints → '+content['label'], ha='center', fontsize=11)
            fig.text(.5, .960, 'Diamonds: Val ≥86% retention operating points (0.01 grid); ΔG1/G3 refer to these points.',
                     ha='center', fontsize=10, color='#475569')
            fig.text(.5, .948, 'Exploratory subset chosen after Test scan; excluded: Moment-DETR and LD-DETR.',
                     ha='center', fontsize=9.5, color='#475569')
            for i, (name, label) in enumerate(NAMES.items()):
                pair = content['backbones'][name]
                for j, k in enumerate((1, 3)):
                    panel(axes[i, j], pair, k, limit)
                    axes[i, j].set_title(f'{label} · G-mIoU@{k}', pad=8)
                    if i == len(NAMES)-1:
                        axes[i, j].set_xlabel('Existence threshold τ')
                bottom = axes[i, 0].get_position().y0
                annotate(fig, pair, bottom-(.022 if i == len(NAMES)-1 else .010), .012)
            fig.text(.5, .031, 'Oracle envelope: max of the two methods with GT presence and each method’s fixed candidates; analysis only.',
                     ha='center', fontsize=9.2, color='#475569')
            fig.text(.5, .020, 'Curves: 101 fixed thresholds, 0–1 score units. τ86 uses Standard Val only; the same threshold is used on both Tests.',
                     ha='center', fontsize=9.2, color='#475569')
            fig.text(.5, .009, 'mAP is ungated; Δ = GEQ+aux − matching auxiliary-trained GMR baseline. Subset selected on Test; all11 results retained.',
                     ha='center', fontsize=9.2, color='#475569')
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for artist in list(fig.texts) + list(fig.legends):
                bounds = artist.get_window_extent(renderer)
                assert bounds.x0 >= 0 and bounds.x1 <= fig.bbox.x1
                assert bounds.y0 >= 0 and bounds.y1 <= fig.bbox.y1
            for ext in ('png', 'pdf', 'svg'):
                path = OUT/(stem+'.'+ext)
                extra = dict(metadata={'CreationDate': None, 'ModDate': None}) if ext == 'pdf' else {}
                fig.savefig(path, dpi=180, facecolor='white', **extra)
                files.append(path)
            combined.savefig(fig)
            plt.close(fig)
            path = OUT/(split.upper()+'_BY_BACKBONE.pdf')
            with PdfPages(path, metadata={'CreationDate': None, 'ModDate': None}) as pages:
                for name, label in NAMES.items():
                    pair = content['backbones'][name]
                    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), sharey=True)
                    fig.subplots_adjust(left=.075, right=.985, top=.75, bottom=.30, wspace=.24)
                    fig.suptitle(label+' · '+content['label'], fontsize=14, fontweight='bold', y=.97)
                    fig.text(.5, .86, 'Exploratory Test-selected subset · Diamonds: Val ≥86% retention · Fixed checkpoints', ha='center', fontsize=10)
                    for ax, k in zip(axes, (1, 3)):
                        panel(ax, pair, k, limit)
                        ax.set_title(f'G-mIoU@{k}')
                        ax.set_xlabel('Existence threshold τ')
                    annotate(fig, pair, .16, .065)
                    pages.savefig(fig)
                    plt.close(fig)
            files.append(path)
    files.append(OUT/'BOTH_TESTS.pdf')
    return files


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot, rows, excluded = prepare()
    (OUT/'PLOT_DATA.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    for name, values in [('METRICS.csv', rows), ('EXCLUDED.csv', excluded)]:
        with (OUT/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(values)
    files = render(snapshot)
    audit = dict(passed=True, exploratory=True, subset_selected_on_test=True,
                 target_percent=86, threshold_search='grid_101', included_backbones=list(NAMES),
                 excluded_backbones=['moment_detr', 'ld_detr'], both_tests_positive_G1_for_all_included=True,
                 curve_values_checked=7272, retained_working_points_checked=72,
                 inherited_oracle_values_from_verified_full_snapshot=72,
                 val_thresholds_verified_against_all_101_candidates=18,
                 source_full_snapshot_sha256=sha(SOURCE/'CURVE_PLOT_DATA.json'),
                 source_scan_metrics_sha256=sha(SCAN/'METRICS.csv'),
                 script_sha256=sha(Path(__file__)), plot_data_sha256=sha(OUT/'PLOT_DATA.json'),
                 new_training=False, new_checkpoint_inference=False,
                 files={p.name:sha(p) for p in files})
    (OUT/'AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
    lines = ['# 86%目标：9骨干探索性展示', '',
             '**本子集在查看本轮 Standard / Full Test 的86%扫描结果后选出**：保留G1在两套Test都提高的9个骨干，剔除Moment-DETR与LD-DETR。它不是独立验证或全部11骨干均有效的证据。', '',
             '各骨干、各方法仍只在Standard Val按正查询保留率≥86%选择最大0.01网格阈值，然后固定到两套Test。GMR baseline为带辅助监督的Stage 1。mAP不门控；ΔG1/G3对应所选工作点。', '',
             '![Standard Test](STANDARD_TEST.png)', '',
             '![Full-Decodable Test](FULL_TEST.png)', '',
             '[两图PDF](BOTH_TESTS.pdf) · [Standard单骨干PDF](STANDARD_BY_BACKBONE.pdf) · [Full单骨干PDF](FULL_BY_BACKBONE.pdf)', '',
             '| 骨干 | Test | ΔmAP | ΔG1 | ΔG3 |', '|---|---|---:|---:|---:|']
    for name,label in NAMES.items():
        for split in ('standard','full'):
            row = next(r for r in rows if (r['backbone'],r['split']) == (name,split))
            pair = snapshot['splits'][split]['backbones'][name]
            dm = pair['geq_full']['mAP']-pair['gmr']['mAP']
            lines.append(f"| {label} | {split} | {dm:+.3f} | {float(row['delta_G1']):+.3f} | {float(row['delta_G3']):+.3f} |")
    lines += ['', '单位为百分点；全部原始数值见[METRICS.csv](METRICS.csv)，剔除的两骨干见[EXCLUDED.csv](EXCLUDED.csv)。', '',
              '[完整11骨干扫描](../README.md) · [原11骨干95%结果](../../e3_high_retention/README.md) · [核验](AUDIT.json)', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    print('Created two 86% figures for the explicitly Test-selected nine-backbone subset.')


if __name__ == '__main__':
    main()
