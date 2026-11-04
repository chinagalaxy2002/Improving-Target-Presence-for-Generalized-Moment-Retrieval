"""Plot the README's fixed Standard / Full Test summary from audited CSV rows."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/e3_eleven'
NAMES = {
    'moment_detr': 'Moment-DETR*', 'eatr': 'EaTR', 'qd_detr': 'QD-DETR',
    'cg_detr': 'CG-DETR', 'sim_detr': 'Sim-DETR', 'bm_detr': 'BM-DETR',
    'bam_detr': 'BAM-DETR', 'uvcom': 'UVCOM', 'tr_detr': 'TR-DETR',
    'taskweave': 'TaskWeave', 'ld_detr': 'LD-DETR',
}
STAGES = ('gmr', 'geq_full')
LABELS = ('GMR baseline', 'GEQ + aux')
COLORS = ('#a6a6a6', '#0072b2')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_audit = json.loads((OUT / 'AUDIT.json').read_text())
    assert source_audit['passed'] and len(source_audit['evaluations']) == 44
    with (OUT / 'METRICS.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 44
    data = {(r['backbone'], r['stage'], r['split']): r for r in rows}
    assert len(data) == 44
    expected = {(name, stage, split) for name in NAMES for stage in STAGES
                for split in ('test_standard', 'test_full')}
    assert set(data) == expected
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
                         'xtick.labelsize': 10, 'ytick.labelsize': 11,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    plotted, figure_files = [], []
    with PdfPages(OUT / 'TEST_SUMMARY_BOTH.pdf') as pages:
        for split, stem, title in (
            ('test_standard', 'STANDARD_SUMMARY', 'Standard Test | 1,036 queries'),
            ('test_full', 'FULL_SUMMARY', 'Full-Decodable Test | 2,970 / 2,986 queries'),
        ):
            fig, axes = plt.subplots(1, 2, figsize=(10.4, 8.8), sharey=True)
            fig.subplots_adjust(left=.14, right=.98, top=.80, bottom=.18, wspace=.26)
            fig.suptitle(title, fontsize=17, fontweight='bold', y=.976)
            handles = [Line2D([], [], color=COLORS[i], marker=('s', 'o')[i], linestyle='none',
                              markersize=8, label=LABELS[i]) for i in range(2)]
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .943),
                       ncol=2, frameon=False, fontsize=12)
            y = np.arange(len(NAMES))
            fig.text(.55, .879, 'Native task losses retained; auxiliary-trained Stage 1 vs GEQ+aux',
                     ha='center', fontsize=10, color='#475569')
            for ax, metric, title_metric, limit, ticks in zip(
                axes, ('mAP', 'G1_mean'), ('mAP | ungated positive queries', 'G-mIoU@1 | mean at 0.4 / 0.5 / 0.6'),
                (29, 66), ([0, 5, 10, 15, 20, 25], [0, 10, 20, 30, 40, 50, 60]),
            ):
                stage_values = []
                for i, stage in enumerate(STAGES):
                    values = np.array([float(data[name, stage, split][metric]) for name in NAMES])
                    assert np.isfinite(values).all() and ((0 <= values) & (values <= 100)).all()
                    stage_values.append(values)
                    centers = y + (-.18 if i == 0 else .18)
                    bars = ax.barh(centers, values, height=.30, color=COLORS[i],
                                   edgecolor=('#555555', '#003f63')[i], linewidth=.5, zorder=2)
                    ax.scatter(values, centers, marker=('s', 'o')[i], s=14,
                               color=('#555555', '#003f63')[i], zorder=3)
                    assert np.array_equal(np.array([b.get_width() for b in bars]), values)
                    for name, value, center in zip(NAMES, values, centers):
                        ax.text(value + limit * .014, center, f'{value:.2f}', va='center',
                                fontsize=9.5, color='#333333')
                        plotted.append(dict(backbone=name, split=split, stage=stage,
                                            metric=metric, value=float(value), label=f'{value:.2f}'))
                gains = int(np.sum(stage_values[1] > stage_values[0]))
                ax.set_title(title_metric, fontsize=11, pad=27)
                ax.text(.5, 1.015, f'GEQ+aux improves {gains}/11 backbones',
                        transform=ax.transAxes, ha='center', fontsize=9.5, color='#475569')
                ax.set_yticks(y, list(NAMES.values()))
                ax.set_xlim(0, limit)
                ax.set_xticks(ticks)
                ax.set_xlabel('Score (0–100; higher is better)')
                ax.set_axisbelow(True)
                ax.grid(axis='x', alpha=.17)
                ax.spines[['top', 'right', 'left']].set_visible(False)
                ax.tick_params(axis='y', length=0)
            axes[0].set_ylim(len(NAMES)-.48, -.55)
            footer = fig.text(.55, .044, '* Moment-DETR results reused. GMR baseline denotes auxiliary-trained Stage 1.\n'
                     'Fixed Val-selected checkpoints; seed 2023. G1 uses fixed thresholds, not 95% retention calibration.',
                     ha='center', fontsize=9, linespacing=1.6, color='#475569')
            # Every label must lie within the canvas; numeric values and bar
            # widths were taken from the same original, unrounded CSV values.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            canvas = fig.bbox
            for ax in axes:
                for label in list(ax.texts) + list(ax.get_yticklabels()) + [ax.title, ax.xaxis.label]:
                    if label.get_visible():
                        bounds = label.get_window_extent(renderer)
                        assert bounds.x0 >= canvas.x0 and bounds.x1 <= canvas.x1
                        assert bounds.y0 >= canvas.y0 and bounds.y1 <= canvas.y1
                assert not ax.xaxis.label.get_window_extent(renderer).overlaps(footer.get_window_extent(renderer))
            for ext in ('png', 'pdf', 'svg'):
                path = OUT / f'{stem}.{ext}'
                fig.savefig(path, dpi=200, facecolor='white')
                figure_files.append(path)
            pages.savefig(fig)
            plt.close(fig)
    figure_files.append(OUT / 'TEST_SUMMARY_BOTH.pdf')
    assert len(plotted) == 88
    (OUT / 'SUMMARY_PLOT_DATA.json').write_text(json.dumps(plotted, indent=2) + '\n')
    audit = dict(passed=True, source_metrics_sha256=sha(OUT/'METRICS.csv'),
                 source_evaluation_audit_sha256=sha(OUT/'AUDIT.json'),
                 script_sha256=sha(Path(__file__)), fixed_metric='mean G1 at 0.4/0.5/0.6; ungated mAP',
                 metric_rows=44, panels=4, bars_checked_against_CSV=88, numeric_labels=88,
                 label_canvas_bounds_checked=True, checkpoint_reselection=False,
                 all_backbones_in_original_order=True, new_training=False,
                 plot_data_sha256=sha(OUT/'SUMMARY_PLOT_DATA.json'),
                 files={p.name: sha(p) for p in figure_files})
    (OUT / 'SUMMARY_PLOT_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Rendered Standard / Full summaries: 4 panels, 88 bars and labels validated against original CSV.')


if __name__ == '__main__':
    main()
