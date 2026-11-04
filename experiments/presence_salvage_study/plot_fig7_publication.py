#!/usr/bin/env python3
"""Generate publication 6-panel threshold sweep figures and Oracle Presence analysis.
Follows the exact visual style of backbonefolder/figures/complete_comparison/build_complete_comparison.py.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from matplotlib.lines import Line2D
import numpy as np
import json, gzip
from pathlib import Path
import sys

ROOT = Path('/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11')
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from soccergmr.evaluation.official.metrics import _clean_pred_windows, _compute_set_iou_score

CORE_5 = ['eatr', 'qd_detr', 'cg_detr', 'sim_detr', 'bm_detr']
NAMES = {'eatr':'EaTR', 'qd_detr':'QD-DETR', 'cg_detr':'CG-DETR', 'sim_detr':'Sim-DETR', 'bm_detr':'BM-DETR'}
GRID = np.arange(101, dtype=float) / 100.0

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.facecolor': 'white'
})

def load_gz(p):
    return [json.loads(line) for line in gzip.decompress(open(p, 'rb').read()).splitlines() if line.strip()]

def order_predictions(preds, truth):
    idx = {p['qid']: p for p in preds}
    return [idx[t['qid']] for t in truth]

def get_test_path(b, m, split):
    if m == 1:
        d = f'{ROOT}/results/test_thresholds/runs' if b in ['moment_detr', 'eatr', 'qd_detr', 'cg_detr', 'sim_detr'] else f'{ROOT}/results/test_thresholds_11_backbones/runs'
        return f'{d}/{b}/{split}/gmr/predictions.jsonl.gz'
    elif m == 3:
        d = f'{ROOT}/results/test_thresholds/runs' if b in ['moment_detr', 'eatr', 'qd_detr', 'cg_detr', 'sim_detr'] else f'{ROOT}/results/test_thresholds_11_backbones/runs'
        return f'{d}/{b}/{split}/geq_full/predictions.jsonl.gz'
    elif m == 4:
        return f'{ROOT}/results/e3_eleven/predictions/{b}/geq_full/test_{split}/predictions.jsonl.gz'

ev_root = ROOT / 'results/e3_high_retention'
ev = json.load(open(ev_root / 'EVIDENCE.json'))

STYLES = {
    'baseline': {'color': '#94A3B8', 'linestyle': '--', 'marker': 's'},
    'geq':      {'color': '#0072B2', 'linestyle': '-',  'marker': '^'},
    'both':     {'color': '#D55E00', 'linestyle': '-',  'marker': 'o'},
}
LEGEND_LABELS = {
    'baseline': 'GMR Baseline',
    'geq':      'GEQ only (Macro Gate)',
    'both':     'BMP: GEQ + Aux (Peak)'
}

def axis(ax, title, xlabel, ylabel, xmax=1.0, ymax=None):
    ax.set_title(title, loc='left', pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(0.2 if xmax == 1.0 else 20))
    ax.grid(axis='y', color='#DFE7F1', linewidth=0.7)
    ax.set_axisbelow(True)
    if ymax is not None:
        ax.set_ylim(0, ymax)

def compute_grid_metrics(preds_dict, gt):
    pos = np.array([bool(t['relevant_windows']) for t in gt])
    series = {}
    for key, preds in preds_dict.items():
        scores = np.array([p['pred_exist_score'] for p in preds], dtype=float)
        raw_iou1 = np.array([
            _compute_set_iou_score([w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'], 10)[:1]], t['relevant_windows'])
            if bool(t['relevant_windows']) else 0.0 for p, t in zip(preds, gt)
        ])
        raw_iou3 = np.array([
            _compute_set_iou_score([w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'], 10)[:3]], t['relevant_windows'])
            if bool(t['relevant_windows']) else 0.0 for p, t in zip(preds, gt)
        ])
        
        grid = []
        for th in GRID:
            acc = scores > th
            g1 = 100.0 * np.where(pos, np.where(acc, raw_iou1, 0.0), np.where(acc, 0.0, 1.0)).mean()
            g3 = 100.0 * np.where(pos, np.where(acc, raw_iou3, 0.0), np.where(acc, 0.0, 1.0)).mean()
            tp = int((acc & pos).sum())
            tn = int((~acc & ~pos).sum())
            fp = int((acc & ~pos).sum())
            fn = int((~acc & pos).sum())
            rec = tp / pos.sum() if pos.sum() > 0 else 0.0
            n_acc = tn / (~pos).sum() if (~pos).sum() > 0 else 0.0
            rej_f1 = (2 * tn) / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0.0
            grid.append({
                'threshold': float(th), 'G-mIoU@1': g1, 'G-mIoU@3': g3,
                'positive_recall': 100.0 * rec, 'N_acc': 100.0 * n_acc,
                'Rej-F1': 100.0 * rej_f1
            })
            
        oracle_g1 = 100.0 * np.where(pos, raw_iou1, 1.0).mean()
        oracle_g3 = 100.0 * np.where(pos, raw_iou3, 1.0).mean()
        series[key] = {'grid': grid, 'oracle_g1': oracle_g1, 'oracle_g3': oracle_g3}
    return series

def render_6panel(title_str, series, oracle_val, out_png, out_pdf):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    fig.subplots_adjust(left=0.065, right=0.986, bottom=0.095, top=0.865, hspace=0.36, wspace=0.26)
    ax1, ax3, delta, tradeoff, f1, fixed = axes.flat

    axis(ax1, '(a) Threshold sensitivity', 'Presence threshold $\\tau$', 'G-mIoU@1 (%)', ymax=70)
    axis(ax3, '(b) Top-3 retrieval', 'Presence threshold $\\tau$', 'G-mIoU@3 (%)', ymax=70)
    axis(delta, '(c) Relative to GMR baseline', 'Presence threshold $\\tau$', '$\\Delta$ G-mIoU@1 (pp)')
    axis(tradeoff, '(d) Rejection / positive retention', 'Positive recall (%)', 'N-acc (%)', xmax=100, ymax=105)
    axis(f1, '(e) Negative-class F1', 'Presence threshold $\\tau$', 'Rejection F1 (%)', ymax=100)
    
    fixed.set_title('(f) Fixed operating points vs. Oracle', loc='left', pad=10)
    fixed.set_xlabel('Presence threshold $\\tau$')
    fixed.set_ylabel('G-mIoU@1 (%)')
    fixed.set_ylim(0, 70)
    fixed.grid(axis='y', color='#DFE7F1', linewidth=0.7)
    fixed.set_axisbelow(True)

    # Plot Oracle line on ax1 and ax3
    ax1.axhline(oracle_val, color='#D97706', linestyle=':', linewidth=1.8, label=f'Oracle Presence ({oracle_val:.1f}%)')
    ax3.axhline(oracle_val, color='#D97706', linestyle=':', linewidth=1.8, label=f'Oracle Presence ({oracle_val:.1f}%)')
    
    # Plot series
    base_grid = np.array([r['G-mIoU@1'] for r in series['baseline']['grid']])
    deltas = []
    
    for key in ['baseline', 'geq', 'both']:
        d = series[key]
        rows = d['grid']
        x = np.array([r['threshold'] for r in rows])
        g1 = np.array([r['G-mIoU@1'] for r in rows])
        g3 = np.array([r['G-mIoU@3'] for r in rows])
        rec = np.array([r['positive_recall'] for r in rows])
        nacc = np.array([r['N_acc'] for r in rows])
        rf1 = np.array([r['Rej-F1'] for r in rows])
        
        style = STYLES[key]
        lbl = LEGEND_LABELS[key]
        
        # ax1
        ax1.plot(x, g1, color=style['color'], linestyle=style['linestyle'], marker=style['marker'],
                 markevery=10, markersize=3.8, linewidth=2.0 if key=='both' else 1.8, label=lbl)
        # ax3
        ax3.plot(x, g3, color=style['color'], linestyle=style['linestyle'], marker=style['marker'],
                 markevery=10, markersize=3.8, linewidth=2.0 if key=='both' else 1.8)
        # delta
        diff = g1 - base_grid
        delta.plot(x, diff, color=style['color'], linestyle=style['linestyle'], marker=style['marker'],
                   markevery=10, markersize=3.8, linewidth=2.0 if key=='both' else 1.8)
        deltas.extend(diff.tolist())
        # tradeoff
        tradeoff.plot(rec, nacc, color=style['color'], linestyle=style['linestyle'], linewidth=1.8)
        # scatter fixed points 0.4, 0.5, 0.6
        for th in (0.4, 0.5, 0.6):
            pt = rows[int(round(th * 100))]
            tradeoff.scatter(pt['positive_recall'], pt['N_acc'], color=style['color'], marker=style['marker'], s=30, zorder=5)
        # f1
        f1.plot(x, rf1, color=style['color'], linestyle=style['linestyle'], marker=style['marker'],
                markevery=10, markersize=3.8, linewidth=2.0 if key=='both' else 1.8)

    # Markers and lines
    for ax in (ax1, ax3, delta, f1):
        ax.axvline(0.5, color='#94A3B8', linestyle=':', linewidth=1.0, zorder=0)
    delta.axhline(0, color='#94A3B8', linewidth=0.8, zorder=0)

    # Bar chart in (f)
    positions = np.arange(3)
    width = 0.25
    for number, key in enumerate(['baseline', 'geq', 'both']):
        d = series[key]
        vals = [d['grid'][int(round(t * 100))]['G-mIoU@1'] for t in (0.4, 0.5, 0.6)]
        offset = (number - 1) * width
        bars = fixed.bar(positions + offset, vals, width=width * 0.92, color=STYLES[key]['color'], zorder=3)
        fixed.bar_label(bars, labels=[f'{v:.1f}' for v in vals], padding=3, rotation=90, fontsize=8.5)
    
    fixed.axhline(oracle_val, color='#D97706', linestyle=':', linewidth=1.8)
    fixed.text(2.35, oracle_val + 1.2, f'Oracle: {oracle_val:.1f}%', color='#B45309', fontsize=9, fontweight='bold', ha='right')
    fixed.set_xticks(positions, ['$\\tau=0.4$', '$\\tau=0.5$', '$\\tau=0.6$'])

    handles = [Line2D([0], [0], **STYLES[key], linewidth=2.0, markersize=5, label=LEGEND_LABELS[key]) for key in ['baseline', 'geq', 'both']]
    handles.append(Line2D([0], [0], color='#D97706', linestyle=':', linewidth=2.0, label=f'Oracle Presence Ceiling ({oracle_val:.1f}%)'))
    
    fig.suptitle(title_str, fontsize=14, fontweight='bold', y=0.991)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.952), ncol=len(handles), frameon=False, columnspacing=1.8, handlelength=2.5)

    footer = f'EVALUATION | Core-5 DETR Models | All 101 thresholds evaluated without post-hoc cherry-picking | Oracle Presence Ceiling = {oracle_val:.2f}%'
    fig.text(0.5, 0.016, footer, ha='center', va='bottom', fontsize=9.5, color='#475569')

    plt.savefig(out_png, dpi=240, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f'Successfully generated {out_png}')

def main():
    Path('docs/images').mkdir(parents=True, exist_ok=True)
    
    # 1. Standard Test: Compute Core-5 Average series
    gt_std = load_gz(ev_root / ev['ground_truth']['standard']['path'])
    
    # Aggregate grids across Core-5
    std_models_data = {}
    for b in CORE_5:
        p1 = order_predictions(load_gz(get_test_path(b, 1, 'standard')), gt_std)
        p3 = order_predictions(load_gz(get_test_path(b, 3, 'standard')), gt_std)
        p4 = order_predictions(load_gz(get_test_path(b, 4, 'standard')), gt_std)
        std_models_data[b] = compute_grid_metrics({'baseline': p1, 'geq': p3, 'both': p4}, gt_std)

    # Average series for Core-5
    mean_series_std = {'baseline': {'grid': []}, 'geq': {'grid': []}, 'both': {'grid': []}}
    for key in ['baseline', 'geq', 'both']:
        for i in range(101):
            row = {'threshold': GRID[i]}
            for m in ['G-mIoU@1', 'G-mIoU@3', 'positive_recall', 'N_acc', 'Rej-F1']:
                row[m] = np.mean([std_models_data[b][key]['grid'][i][m] for b in CORE_5])
            mean_series_std[key]['grid'].append(row)
            
    oracle_std = float(np.mean([std_models_data[b]['baseline']['oracle_g1'] for b in CORE_5]))
    
    # Render Standard 6-panel
    render_6panel(
        'Core-5 DETR Family Average | Soccer-GMR Standard Test',
        mean_series_std, oracle_std,
        'docs/images/fig7_core5_standard_test_threshold_sweep.png',
        'docs/images/fig7_core5_standard_test_threshold_sweep.pdf'
    )

    # 2. Render EaTR and CG-DETR 6-panels directly
    for b in ['eatr', 'cg_detr']:
        b_name = NAMES[b]
        b_oracle = std_models_data[b]['baseline']['oracle_g1']
        render_6panel(
            f'{b_name} | Soccer-GMR Standard Test',
            std_models_data[b], b_oracle,
            f'docs/images/{b}_fig7_test_threshold_sweep.png',
            f'docs/images/{b}_fig7_test_threshold_sweep.pdf'
        )

    # 3. Oracle Presence Salvage Ceiling Comparison Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=240)
    
    labels = ['EaTR', 'QD-DETR', 'CG-DETR', 'Sim-DETR', 'BM-DETR', 'Core-5 Mean']
    base_vals = [std_models_data[b]['baseline']['grid'][50]['G-mIoU@1'] for b in CORE_5] + [mean_series_std['baseline']['grid'][50]['G-mIoU@1']]
    bmp_vals  = [std_models_data[b]['both']['grid'][50]['G-mIoU@1'] for b in CORE_5] + [mean_series_std['both']['grid'][50]['G-mIoU@1']]
    orc_vals  = [std_models_data[b]['baseline']['oracle_g1'] for b in CORE_5] + [oracle_std]
    
    x = np.arange(len(labels))
    w = 0.26
    
    ax1.bar(x - w, base_vals, w, label='Baseline G-mIoU', color='#94A3B8', edgecolor='black', linewidth=0.6)
    ax1.bar(x,     bmp_vals,  w, label='BMP (Ours, $\\tau=0.50$)', color='#059669', edgecolor='black', linewidth=0.6)
    ax1.bar(x + w, orc_vals,  w, label='Oracle Presence Ceiling (100% Accurate)', color='#D97706', edgecolor='black', linewidth=0.6, alpha=0.85)
    
    for i in range(len(labels)):
        ax1.annotate(f'{bmp_vals[i]:.1f}%', (x[i], bmp_vals[i] + 0.8), ha='center', fontsize=8.5, fontweight='bold', color='#065F46')
        ax1.annotate(f'{orc_vals[i]:.1f}%', (x[i] + w, orc_vals[i] + 0.8), ha='center', fontsize=8.5, fontweight='bold', color='#B45309')

    ax1.set_title('(a) G-mIoU@1: Baseline vs. Ours vs. Oracle Ceiling', fontsize=11, fontweight='bold')
    ax1.set_ylabel('G-mIoU@1 (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax1.set_ylim(25, 68)
    ax1.grid(axis='y', color='#DFE7F1', linewidth=0.7)
    ax1.legend(loc='lower right', frameon=False, fontsize=9.5)
    
    # Right panel: Salvage percentage = (Ours - Base) / (Oracle - Base)
    gaps = [orc_vals[i] - base_vals[i] for i in range(len(labels))]
    recovered = [bmp_vals[i] - base_vals[i] for i in range(len(labels))]
    ratios = [100.0 * recovered[i] / gaps[i] if gaps[i] > 0 else 0.0 for i in range(len(labels))]
    
    colors_bar = ['#0284C7', '#0284C7', '#0284C7', '#64748B', '#64748B', '#D97706']
    bars = ax2.bar(labels, ratios, color=colors_bar, width=0.45, edgecolor='black', linewidth=0.6)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f'{h:.1f}%', (bar.get_x() + bar.get_width() / 2, h + 1.2), ha='center', fontsize=9.5, fontweight='bold', color='#0F172A')

    ax2.set_title('(b) Presence Gap Recovery Ratio: $(G_{\\text{BMP}} - G_{\\text{Base}}) / (G_{\\text{Oracle}} - G_{\\text{Base}})$', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Presence Gap Recovered (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9.5, fontweight='bold')
    ax2.set_ylim(0, 85)
    ax2.grid(axis='y', color='#DFE7F1', linewidth=0.7)
    
    plt.suptitle('Oracle Presence Salvage Ceiling Analysis (Soccer-GMR Standard Test)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('docs/images/figure_oracle_presence_salvage.png', dpi=240, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print('Successfully generated docs/images/figure_oracle_presence_salvage.png')

if __name__ == '__main__':
    main()
