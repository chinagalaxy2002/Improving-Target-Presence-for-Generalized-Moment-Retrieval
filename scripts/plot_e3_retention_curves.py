"""Reference-style Standard / Full curves with Val >=95% retention operating points."""
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
OUT = ROOT / 'results/e3_high_retention'
BASE = ROOT / 'results/e3_eleven'
NAMES = {
    'moment_detr': 'Moment-DETR*', 'eatr': 'EaTR', 'qd_detr': 'QD-DETR',
    'cg_detr': 'CG-DETR', 'sim_detr': 'Sim-DETR', 'bm_detr': 'BM-DETR',
    'bam_detr': 'BAM-DETR', 'uvcom': 'UVCOM', 'tr_detr': 'TR-DETR',
    'taskweave': 'TaskWeave', 'ld_detr': 'LD-DETR',
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
    path = OUT / meta['path']
    assert sha(path) == meta['sha256']
    raw = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest() == meta['uncompressed_sha256']
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) == meta['rows']
    return rows


def prepare():
    audit = json.loads((OUT/'AUDIT.json').read_text())
    assert audit['passed'] and audit['all_published_backbones_included']
    evidence = json.loads((OUT/'EVIDENCE.json').read_text())
    assert set(evidence['models']) == set(NAMES)
    curves = read_csv(BASE/'CURVES.csv')
    metrics = read_csv(BASE/'METRICS.csv')
    retention = read_csv(OUT/'METRICS.csv')
    assert len(curves) == 4444 and len(metrics) == len(retention) == 44
    snapshot = dict(splits={})
    oracle_rows, checks, points = [], 0, 0
    for split, label in [('standard', 'Standard Test (1,036 queries)'),
                         ('full', 'Full-Decodable Test (2,970 / 2,986 queries)')]:
        gt = read_prediction(evidence['ground_truth'][split])
        positive = np.array([bool(g['relevant_windows']) for g in gt])
        content = dict(label=label, backbones={})
        for name in NAMES:
            content['backbones'][name] = {}
            retained = next(r for r in retention if (r['backbone'], r['split'], r['search']) == (name, split, 'grid_101'))
            for method in METHODS:
                info = evidence['models'][name]['methods'][method]
                assert info['predictions'][split]['checkpoint_sha256'] == info['checkpoint_sha256']
                pred = read_prediction(info['predictions'][split])
                byid = {p['qid']: p for p in pred}
                assert len(byid) == len(pred) == len(gt) and set(byid) == {g['qid'] for g in gt}
                pred = [byid[g['qid']] for g in gt]
                assert all(p['vid'] == g['vid'] for p, g in zip(pred, gt))
                scores = np.array([p['pred_exist_score'] for p in pred])
                metric = next(r for r in metrics if (r['backbone'], r['stage'], r['split']) == (name, method, 'test_'+split))
                series = [r for r in curves if (r['backbone'], r['stage'], r['split']) == (name, method, 'test_'+split)]
                thresholds = np.array([i/100 for i in range(101)])
                assert [float(r['threshold']) for r in series] == list(thresholds)
                tau = float(retained[method+'_threshold'])
                entry = dict(mAP=float(metric['mAP']), checkpoint_sha256=info['checkpoint_sha256'],
                             tau95=tau, val_retention=float(retained[method+'_val_TPR']),
                             test_retention=float(retained[method+'_test_TPR']),
                             thresholds=list(thresholds), gmiou={}, oracle={}, point95={})
                assert entry['mAP'] == float(retained[method+'_mAP']) and entry['val_retention'] >= 95
                perfect_gate = [{**p, 'pred_relevant_windows': p['pred_relevant_windows'] if g['relevant_windows'] else []}
                                for p, g in zip(pred, gt)]
                official = compute_G_mIoU(perfect_gate, gt, k_list=(1, 3))
                for k in (1, 3):
                    raw = np.array([_compute_set_iou_score(
                        [w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'], max_pred_windows=10)[:k]],
                        g['relevant_windows']) for p, g in zip(pred, gt)])
                    oracle = float(np.where(positive, raw, 1.).mean())
                    assert round(oracle*100, 2) == official[f'G-mIoU@{k}']
                    exact = 100*np.where(scores[:, None] > thresholds[None, :], raw[:, None], (~positive)[:, None]).mean(0)
                    plotted = [float(r[f'G{k}'])/100 for r in series]
                    assert all(round(float(x), 2) == float(r[f'G{k}']) for x, r in zip(exact, series))
                    point = float(np.where(scores > tau, raw, ~positive).mean())
                    assert abs(point*100-float(retained[f'{method}_G{k}'])) < 1e-10
                    assert max(plotted) <= oracle+0.000051
                    entry['gmiou'][str(k)] = plotted
                    entry['oracle'][str(k)] = oracle
                    entry['point95'][str(k)] = point
                    checks += 1
                    points += 101
                content['backbones'][name][method] = entry
                oracle_rows.append(dict(split=split, backbone=name, method=method,
                                        oracle_G1=entry['oracle']['1'], oracle_G3=entry['oracle']['3']))
            print('PREPARED', name, split, flush=True)
        snapshot['splits'][split] = content
    return snapshot, oracle_rows, checks, points


def panel(ax, series, k, limit):
    for method in METHODS:
        row = series[method]
        line, = ax.plot(row['thresholds'], row['gmiou'][str(k)], linewidth=1.85, markersize=4.5,
                       markevery=[10, 30, 40, 50, 60, 70, 90], markerfacecolor='white',
                       markeredgewidth=1.1, **STYLES[method])
        assert np.array_equal(line.get_ydata(), row['gmiou'][str(k)])
        ax.scatter([row['tau95']], [row['point95'][str(k)]], marker='D', s=38,
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
    labels = [f"{LABELS[m]} | mAP {series[m]['mAP']:.2f}% | τ95 {series[m]['tau95']:.2f}" for m in METHODS]
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.53, top),
               frameon=False, fontsize=9.5, columnspacing=3, handlelength=3.)
    base, joint = series['gmr'], series['geq_full']
    changes = [('ΔmAP', joint['mAP']-base['mAP']),
               ('ΔG1 at τ95', 100*(joint['point95']['1']-base['point95']['1'])),
               ('ΔG3 at τ95', 100*(joint['point95']['3']-base['point95']['3']))]
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
            fig, axes = plt.subplots(11, 2, figsize=(12.4, 36.8), sharex=True, sharey=True)
            fig.subplots_adjust(left=.075, right=.985, top=.93, bottom=.075, hspace=1.08, wspace=.24)
            fig.suptitle('11 backbones · GMR baseline vs GEQ+aux', fontsize=16, fontweight='bold', y=.986)
            fig.text(.5, .973, 'Fixed Val-selected checkpoints → '+content['label'], ha='center', fontsize=11)
            fig.text(.5, .960, 'Diamonds: Val ≥95% retention operating points (0.01 grid); ΔG1/G3 refer to these points.',
                     ha='center', fontsize=10, color='#475569')
            fig.text(.5, .948, 'Seed 2023 · 128 video tokens · GMR patience 20 / GEQ patience 10 · * Moment-DETR reused',
                     ha='center', fontsize=9.5, color='#475569')
            for i, (name, label) in enumerate(NAMES.items()):
                pair = content['backbones'][name]
                for j, k in enumerate((1, 3)):
                    panel(axes[i, j], pair, k, limit)
                    axes[i, j].set_title(f'{label} · G-mIoU@{k}', pad=8)
                    if i == 10:
                        axes[i, j].set_xlabel('Existence threshold τ')
                bottom = axes[i, 0].get_position().y0
                annotate(fig, pair, bottom-(.022 if i == 10 else .010), .012)
            fig.text(.5, .031, 'Oracle envelope: max of the two methods with GT presence and each method’s fixed candidates; analysis only.',
                     ha='center', fontsize=9.2, color='#475569')
            fig.text(.5, .020, 'Curves: 101 fixed thresholds, 0–1 score units. τ95 uses Standard Val only; the same threshold is used on both Tests.',
                     ha='center', fontsize=9.2, color='#475569')
            fig.text(.5, .009, 'mAP is ungated; Δ = GEQ+aux − matching GMR baseline (auxiliary-trained Stage 1). No Test tuning or model reselection.',
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
                    fig.text(.5, .86, 'Diamonds: Val ≥95% retention; fixed checkpoints and native localization candidates', ha='center', fontsize=10)
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
    snapshot, oracle_rows, official_checks, points = prepare()
    (OUT/'CURVE_PLOT_DATA.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    with (OUT/'ORACLE.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(oracle_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(oracle_rows)
    files = render(snapshot)
    audit = dict(passed=True, backbones=11, splits=['standard', 'full'], curves=88,
                 curve_values_checked=points, oracle_G1_G3_official_checks=official_checks,
                 retention_operating_points_checked=88, prediction_files_checked=44,
                 threshold_source='Standard Val >=95% positive retention; grid_101',
                 deltas='GEQ+aux minus matching GMR at each method own Val-selected threshold',
                 source_curves_sha256=sha(BASE/'CURVES.csv'), source_metrics_sha256=sha(BASE/'METRICS.csv'),
                 source_retention_metrics_sha256=sha(OUT/'METRICS.csv'),
                 evidence_sha256=sha(OUT/'EVIDENCE.json'), script_sha256=sha(Path(__file__)),
                 plot_data_sha256=sha(OUT/'CURVE_PLOT_DATA.json'), oracle_csv_sha256=sha(OUT/'ORACLE.csv'),
                 reference='https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-e3-aux-11/results/test_thresholds_11_backbones/STANDARD_TEST.png',
                 new_training=False, new_checkpoint_inference=False, test_used_for_selection=False,
                 files={p.name: sha(p) for p in files})
    (OUT/'CURVE_FIGURE_AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('Created Standard / Full reference-style figures: 88 curves, 88 oracle checks and 88 Val-retention points.')


if __name__ == '__main__':
    main()
