"""Render the complete exploratory target scan, keeping both search rules and losses."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/e3_retention_scan'
NAMES = ('moment_detr', 'eatr', 'qd_detr', 'cg_detr', 'sim_detr', 'bm_detr',
         'bam_detr', 'uvcom', 'tr_detr', 'taskweave', 'ld_detr')
LABELS = ('Moment-DETR', 'EaTR', 'QD-DETR', 'CG-DETR', 'Sim-DETR', 'BM-DETR',
          'BAM-DETR', 'UVCOM', 'TR-DETR', 'TaskWeave', 'LD-DETR')
SEARCHES = ('grid_101', 'val_breakpoints')
STYLES = {'grid_101': dict(color='#0072B2', linestyle='-', marker='o'),
          'val_breakpoints': dict(color='#D55E00', linestyle='--', marker='s')}
SEARCH_LABELS = {'grid_101': '0.01 threshold grid', 'val_breakpoints': 'All Val score breakpoints'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    audit = json.loads((OUT/'AUDIT.json').read_text())
    assert audit['passed'] and audit['exploratory']
    rows = read_csv(OUT/'METRICS.csv')
    summary = read_csv(OUT/'TEST_SUMMARY.csv')
    selected = json.loads((OUT/'VAL_SELECTION_FROZEN.json').read_text())['selected']
    grid_target = int(selected['grid_101']['target_percent'])
    plt.rcParams.update({'font.size': 10.5, 'axes.labelsize': 10.5, 'axes.titlesize': 12,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9.5,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    files, plotted = [], []

    def save(fig, stem):
        for ext in ('png', 'svg', 'pdf'):
            path = OUT/(stem+'.'+ext)
            options = dict(metadata={'CreationDate': None, 'ModDate': None}) if ext == 'pdf' else {}
            fig.savefig(path, dpi=200, facecolor='white', **options)
            files.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.1), sharex=True, sharey='row')
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.17, hspace=.30, wspace=.20)
    fig.suptitle('Exploratory sensitivity to the Val positive-retention target', fontsize=15, fontweight='bold', y=.98)
    fig.text(.5, .929, 'All 11 backbones; per-target thresholds fixed on Val, then applied to both Tests', ha='center', fontsize=10)
    for column, split in enumerate(('standard', 'full')):
        for search in SEARCHES:
            group = sorted([r for r in summary if (r['search'], r['split']) == (search, split)], key=lambda r:int(r['target_percent']))
            assert [int(r['target_percent']) for r in group] == list(range(101))
            for row, metric in enumerate(('mean_delta_G1', 'positive_G1_backbones')):
                x = np.array([int(r['target_percent']) for r in group])
                y = np.array([float(r[metric]) for r in group])
                line, = axes[row, column].plot(x, y, markevery=5, markersize=3.2, linewidth=1.6, **STYLES[search])
                assert np.array_equal(line.get_ydata(), y)
                plotted.extend(dict(figure='TARGET_SWEEP', split=split, search=search, metric=metric,
                                    target_percent=int(t), value=float(v)) for t,v in zip(x,y))
        axes[0, column].set_title('Standard Test' if split == 'standard' else 'Full-Decodable Test')
        axes[0, column].axhline(0, color='#64748B', linewidth=.8)
        axes[1, column].axhline(11, color='#64748B', linewidth=.8, linestyle=':')
        axes[1, column].set_ylim(-.3, 11.7)
        axes[1, column].set_yticks([0, 2, 4, 6, 8, 10, 11])
        axes[1, column].set_xlabel('Minimum positive retention required on Val (%)')
        for ax in axes[:, column]:
            ax.axvline(95, color='#111827', linewidth=1, linestyle='--')
            ax.axvline(grid_target, color='#8E6C8A', linewidth=1, linestyle=':')
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 20, 40, 60, 80, 95, 100])
            ax.grid(alpha=.18)
            ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].set_ylabel('Mean G1 gain (GEQ+aux − GMR; pp)')
    axes[1, 0].set_ylabel('Backbones with G1 gain > 0 (of 11)')
    handles = [Line2D([], [], label=SEARCH_LABELS[s], linewidth=1.6, markersize=4, **STYLES[s]) for s in SEARCHES]
    handles += [Line2D([], [], color='#111827', linestyle='--', label='Original 95% target'),
                Line2D([], [], color='#8E6C8A', linestyle=':', label=f'{grid_target}%: Val-only grid reference')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .902), ncol=2, frameon=False, fontsize=9.5)
    fig.text(.5, .068, 'Exploratory scan after Test inspection; full 95% results are preserved alongside the selected-subset display.\n'
             'Lower retention changes the operating scenario. Mean gains can increase while some backbones still lose.',
             ha='center', fontsize=9, linespacing=1.6, color='#475569')
    save(fig, 'TARGET_SWEEP')
    plt.close(fig)

    high_rows = [r for r in rows if r['split'] in ('standard','full') and 80 <= int(r['target_percent']) <= 100]
    limit = max(abs(float(r['delta_G1'])) for r in high_rows)
    limit = max(1, float(np.ceil(limit)))
    for split, stem, title in [('standard', 'STANDARD_G1_GAIN', 'Standard Test'), ('full', 'FULL_G1_GAIN', 'Full-Decodable Test')]:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 7.2), sharey=True)
        fig.subplots_adjust(left=.14, right=.98, top=.81, bottom=.23, wspace=.10)
        fig.suptitle(title+' | per-backbone G1 gain, targets 80–100%', fontsize=14, fontweight='bold', y=.97)
        fig.text(.55, .918, 'Exploratory; all 11 backbones retained; a shared target for both methods and all backbones', ha='center', fontsize=9.5)
        for ax, search in zip(axes, SEARCHES):
            matrix = np.array([[float(next(r for r in high_rows if (r['search'], r['split'], r['backbone'], int(r['target_percent'])) == (search, split, n, target))['delta_G1'])
                                for target in range(80, 101)] for n in NAMES])
            assert matrix.shape == (11, 21)
            im = ax.imshow(matrix, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit, interpolation='nearest')
            ax.set_title(SEARCH_LABELS[search], pad=10)
            ax.set_yticks(range(11), LABELS)
            ax.set_xticks([0, 5, 10, 15, 20], [80, 85, 90, 95, 100])
            ax.set_xlabel('Required Val positive retention (%)')
            ax.add_patch(Rectangle((14.5, -.5), 1, 11, fill=False, linewidth=1.4, edgecolor='#111827'))
            if search == 'grid_101':
                ax.add_patch(Rectangle((grid_target-80-.5, -.5), 1, 11, fill=False, linewidth=1.3, edgecolor='#8E6C8A'))
            for i, name in enumerate(NAMES):
                for j, target in enumerate(range(80, 101)):
                    plotted.append(dict(figure=stem, split=split, search=search, metric='delta_G1', backbone=name,
                                        target_percent=target, value=float(matrix[i,j])))
                    if target in (80, 85, 90, 95, 100):
                        align = 'left' if j == 0 else 'right' if j == 20 else 'center'
                        ax.text(j-.30 if j == 0 else j+.30 if j == 20 else j, i, f'{matrix[i,j]:+.2f}',
                                ha=align, va='center', fontsize=8,
                                color='white' if abs(matrix[i,j]) > .60*limit else '#111827')
        cax = fig.add_axes([.28, .12, .56, .025])
        fig.colorbar(im, cax=cax, orientation='horizontal', label='G1 gain in percentage points (blue: gain; red: loss)')
        fig.text(.55, .025, 'Black outline: original 95%. Purple outline: Val-selected grid target. Same color scale across both Tests.\n'
                 'Full 0–100% scan and exact values are provided in CSV; this panel zooms the high-retention range.',
                 ha='center', fontsize=8.5, linespacing=1.5, color='#475569')
        save(fig, stem)
        plt.close(fig)
    (OUT/'PLOT_DATA.json').write_text(json.dumps(plotted, indent=2)+'\n')
    (OUT/'PLOT_AUDIT.json').write_text(json.dumps(dict(passed=True, exploratory=True,
         source_metrics_sha256=sha(OUT/'METRICS.csv'), source_summary_sha256=sha(OUT/'TEST_SUMMARY.csv'),
         script_sha256=sha(Path(__file__)), plotted_values=len(plotted), all_11_backbones_included=True,
         both_threshold_searches_displayed=True, heatmap_shared_symmetric_limit=limit,
         plot_data_sha256=sha(OUT/'PLOT_DATA.json'), files={p.name:sha(p) for p in files}), indent=2)+'\n')
    print('Created full-range target scan and two all-backbone gain heatmaps.')


if __name__ == '__main__':
    main()
