"""Plot the README's 86%-retention three-way Test summaries."""
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
NAMES = (
    'eatr', 'qd_detr', 'cg_detr', 'sim_detr', 'bm_detr', 'bam_detr',
    'uvcom', 'tr_detr', 'taskweave',
)
LABELS = {
    'eatr': 'EaTR', 'qd_detr': 'QD-DETR', 'cg_detr': 'CG-DETR',
    'sim_detr': 'Sim-DETR', 'bm_detr': 'BM-DETR', 'bam_detr': 'BAM-DETR',
    'uvcom': 'UVCOM', 'tr_detr': 'TR-DETR', 'taskweave': 'TaskWeave',
}
SERIES = ('gmr', 'geq_only_reference', 'geq_full')
SERIES_LABELS = ('GMR(aux)', 'GEQ only (reference)', 'GEQ + aux')
COLORS = ('#a6a6a6', '#e69f00', '#0072b2')
EDGES = ('#555555', '#8c5f00', '#003f63')
MARKERS = ('s', '^', 'o')
REFERENCE_COMMIT = '682db34ab0511ce0ebfd69e9b966150cac6473a5'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_audit = json.loads((OUT / 'RETENTION86_AUDIT.json').read_text())
    assert source_audit['passed'] and source_audit['exploratory']
    assert source_audit['target_percent'] == 86
    assert source_audit['required_val_positives'] == 220
    assert tuple(source_audit['displayed_backbones']) == NAMES
    assert source_audit['excluded_from_display'] == ['moment_detr', 'ld_detr']
    with (OUT / 'RETENTION86_METRICS.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 22
    data = {(row['backbone'], row['split']): row for row in rows}
    assert len(data) == 22
    selected = {
        row['backbone'] for row in rows
        if row['selected_both_test_G1_gain'] == 'True'
    }
    assert selected == set(NAMES)
    reference_audit = json.loads(
        (OUT / 'GEQ_ONLY_RETENTION86_AUDIT.json').read_text())
    assert reference_audit['passed']
    assert reference_audit['reference_commit'] == REFERENCE_COMMIT
    assert reference_audit['unmatched_parent_and_protocol']
    assert reference_audit['same_threshold_used_for_standard_and_full']
    assert tuple(reference_audit['backbones']) == NAMES
    with (OUT / 'GEQ_ONLY_RETENTION86_METRICS.csv').open() as stream:
        reference_rows = list(csv.DictReader(stream))
    assert len(reference_rows) == 18
    reference = {(row['backbone'], row['split']): row for row in reference_rows}
    assert len(reference) == 18
    for name in NAMES:
        standard = data[name, 'standard']
        full = data[name, 'full']
        for stage in ('gmr', 'geq_full'):
            assert standard[f'{stage}_threshold'] == full[f'{stage}_threshold']
        assert reference[name, 'standard']['threshold'] == reference[name, 'full']['threshold']
        assert float(standard['delta_G1']) > 0 and float(full['delta_G1']) > 0

    def metric_value(name, split, series):
        if series == 'geq_only_reference':
            return float(reference[name, split]['G1'])
        return float(data[name, split][f'{series}_G1'])

    def threshold(name, split, series):
        if series == 'geq_only_reference':
            return float(reference[name, split]['threshold'])
        return float(data[name, split][f'{series}_threshold'])

    max_value = max(metric_value(name, split, series)
                    for name in NAMES for split in ('standard', 'full')
                    for series in SERIES)
    limit = float(5 * np.ceil((max_value + 8) / 5))
    assert limit == 45
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 11,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none',
        'svg.hashsalt': 'soccergmr-retention86-summary',
    })
    plotted = []
    figure_files = []
    curve_files = [
        OUT / f'{split}_TEST.{ext}'
        for split in ('STANDARD', 'FULL') for ext in ('png', 'pdf', 'svg')
    ]
    curve_hashes = {path.name: sha(path) for path in curve_files}

    with PdfPages(OUT / 'TEST_SUMMARY_BOTH.pdf', metadata={
        'Title': '86% Val-retention G-mIoU@1 summaries',
        'CreationDate': None,
        'ModDate': None,
    }) as pages:
        for split, stem, title in (
            ('standard', 'STANDARD_SUMMARY', 'Standard Test | 1,036 queries'),
            ('full', 'FULL_SUMMARY', 'Full-Decodable Test | 2,970 / 2,986 queries'),
        ):
            fig, ax = plt.subplots(figsize=(10.8, 7.9))
            fig.subplots_adjust(left=.20, right=.97, top=.75, bottom=.22)
            fig.suptitle(title, fontsize=17, fontweight='bold', y=.976)
            fig.text(
                .585, .918,
                'G-mIoU@1 at each method\'s independently Val-selected existence gate',
                ha='center', fontsize=11.5,
            )
            fig.text(
                .585, .875,
                '86% Val positive-retention target (≥220/255) | selected 9/11 exploratory subset',
                ha='center', fontsize=10, color='#475569',
            )
            handles = [
                Line2D([], [], color=COLORS[index], marker=MARKERS[index],
                       markeredgecolor=EDGES[index], linestyle='none', markersize=8,
                       label=SERIES_LABELS[index])
                for index in range(3)
            ]
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.585, .848),
                       ncol=3, frameon=False, fontsize=11.2)
            y = np.arange(len(NAMES))
            values_by_series = {}
            for index, series in enumerate(SERIES):
                values = np.array([metric_value(name, split, series) for name in NAMES])
                values_by_series[series] = values
                centers = y + (-.24, 0, .24)[index]
                bars = ax.barh(
                    centers, values, height=.19, color=COLORS[index],
                    edgecolor=EDGES[index], linewidth=.55, zorder=2,
                )
                ax.scatter(values, centers, marker=MARKERS[index], s=16,
                           color=EDGES[index], zorder=3)
                assert np.array_equal(np.array([bar.get_width() for bar in bars]), values)
                for name, value, center in zip(NAMES, values, centers):
                    baseline_delta = float(data[name, split]['delta_G1'])
                    reference_delta = (
                        float(data[name, split]['geq_full_G1'])
                        - float(reference[name, split]['G1']))
                    label = (f'{value:.2f}  (vs ref {reference_delta:+.2f})'
                             if series == 'geq_full' else f'{value:.2f}')
                    ax.text(value + .38, center, label, va='center', fontsize=8.8,
                            color='#333333')
                    plotted.append({
                        'backbone': name,
                        'split': split,
                        'series': series,
                        'metric': 'G-mIoU@1',
                        'value': float(value),
                        'threshold': threshold(name, split, series),
                        'geq_aux_minus_gmr': baseline_delta,
                        'geq_aux_minus_geq_only_reference': reference_delta,
                        'label': label,
                    })
            baseline_mean = float(values_by_series['gmr'].mean())
            reference_mean = float(values_by_series['geq_only_reference'].mean())
            auxiliary_mean = float(values_by_series['geq_full'].mean())
            ax.set_title(
                f'9-backbone mean: GMR(aux) {baseline_mean:.2f} | GEQ ref {reference_mean:.2f} | '
                f'GEQ+aux {auxiliary_mean:.2f} (aux−ref {auxiliary_mean - reference_mean:+.2f} pp)',
                fontsize=11, pad=12, color='#334155',
            )
            ax.set_yticks(y, [LABELS[name] for name in NAMES])
            ax.set_ylim(len(NAMES) - .48, -.55)
            ax.set_xlim(0, limit)
            ax.set_xticks([0, 10, 20, 30, 40])
            ax.set_xlabel('G-mIoU@1 (0–100; higher is better)')
            ax.set_axisbelow(True)
            ax.grid(axis='x', alpha=.18)
            ax.spines[['top', 'right', 'left']].set_visible(False)
            ax.tick_params(axis='y', length=0)
            footer = fig.text(
                .585, .075,
                'Each backbone/method selects its own largest 0.01-grid threshold on Standard Val; '
                'the same threshold is used on both Tests.\n'
                'Subset is fixed by current GEQ+aux vs current GMR(aux) gains, not by the GEQ-only reference. '
                'Moment-DETR and LD-DETR are omitted.\n'
                'GEQ-only is an outcome-level reference with a different no-aux GMR parent/protocol; '
                'aux−ref is not a controlled causal ablation.',
                ha='center', fontsize=8.3, linespacing=1.5, color='#475569',
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for label in list(ax.texts) + list(ax.get_yticklabels()) + [ax.title, ax.xaxis.label]:
                if label.get_visible():
                    bounds = label.get_window_extent(renderer)
                    assert bounds.x0 >= fig.bbox.x0 and bounds.x1 <= fig.bbox.x1
                    assert bounds.y0 >= fig.bbox.y0 and bounds.y1 <= fig.bbox.y1
            assert not ax.xaxis.label.get_window_extent(renderer).overlaps(
                footer.get_window_extent(renderer))
            for ext in ('png', 'pdf', 'svg'):
                path = OUT / f'{stem}.{ext}'
                if ext == 'pdf':
                    options = {'metadata': {
                        'Title': title, 'CreationDate': None, 'ModDate': None}}
                elif ext == 'svg':
                    options = {'metadata': {'Date': None}}
                else:
                    options = {}
                fig.savefig(path, dpi=200, facecolor='white', **options)
                figure_files.append(path)
            pages.savefig(fig)
            plt.close(fig)

    figure_files.append(OUT / 'TEST_SUMMARY_BOTH.pdf')
    assert len(plotted) == 54
    plot_data_path = OUT / 'SUMMARY_PLOT_DATA.json'
    plot_data_path.write_text(json.dumps(plotted, indent=2) + '\n')
    audit = {
        'passed': True,
        'exploratory': True,
        'source_metrics_sha256': sha(OUT / 'RETENTION86_METRICS.csv'),
        'source_evaluation_audit_sha256': sha(OUT / 'RETENTION86_AUDIT.json'),
        'reference_metrics_sha256': sha(OUT / 'GEQ_ONLY_RETENTION86_METRICS.csv'),
        'reference_evaluation_audit_sha256': sha(OUT / 'GEQ_ONLY_RETENTION86_AUDIT.json'),
        'reference_commit': REFERENCE_COMMIT,
        'reference_is_unmatched_parent_and_protocol': True,
        'script_sha256': sha(Path(__file__)),
        'metric': 'G-mIoU@1 at per-method Standard-Val-selected threshold',
        'target_percent': 86,
        'required_val_positives': 220,
        'all_backbones_evaluated': 11,
        'displayed_backbones': list(NAMES),
        'excluded_backbones': ['moment_detr', 'ld_detr'],
        'selection_rule': 'positive GEQ+aux minus GMR(aux) delta G1 on both Standard and Full Test',
        'selected_after_test_inspection': True,
        'panels': 2,
        'bars_checked_against_CSV': 54,
        'numeric_value_labels': 54,
        'aux_minus_reference_delta_labels': 18,
        'same_threshold_across_both_tests_checked': True,
        'label_canvas_bounds_checked': True,
        'checkpoint_reselection': False,
        'new_training': False,
        'plot_data_sha256': sha(plot_data_path),
        'full_curve_files_unchanged': curve_hashes,
        'files': {path.name: sha(path) for path in figure_files},
    }
    (OUT / 'SUMMARY_PLOT_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Rendered two 86%-retention summaries: 9 backbones, 54 bars, all labels validated.')


if __name__ == '__main__':
    main()
