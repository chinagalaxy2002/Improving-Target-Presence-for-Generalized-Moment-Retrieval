"""Plot the README's 86%-retention four-way Test summaries."""
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
SERIES = ('gmr_baseline', 'gmr', 'geq_only_reference', 'geq_full')
SERIES_LABELS = ('GMR baseline', 'GMR(aux)', 'GEQ only (reference)', 'GEQ+aux')
COLORS = ('#b0b0b0', '#882255', '#e69f00', '#0072b2')
EDGES = ('#555555', '#551133', '#8c5f00', '#003f63')
MARKERS = ('s', 'd', '^', 'o')
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

    gmr_baseline_audit = json.loads(
        (OUT / 'GMR_BASELINE_RETENTION86_AUDIT.json').read_text())
    assert gmr_baseline_audit['passed']
    assert gmr_baseline_audit['reference_commit'] == REFERENCE_COMMIT
    assert gmr_baseline_audit['same_threshold_used_for_standard_and_full']
    assert tuple(gmr_baseline_audit['backbones']) == NAMES
    with (OUT / 'GMR_BASELINE_RETENTION86_METRICS.csv').open() as stream:
        gmr_baseline_rows = list(csv.DictReader(stream))
    assert len(gmr_baseline_rows) == 18
    gmr_baseline = {(row['backbone'], row['split']): row for row in gmr_baseline_rows}
    assert len(gmr_baseline) == 18

    for name in NAMES:
        standard = data[name, 'standard']
        full = data[name, 'full']
        for stage in ('gmr', 'geq_full'):
            assert standard[f'{stage}_threshold'] == full[f'{stage}_threshold']
        assert reference[name, 'standard']['threshold'] == reference[name, 'full']['threshold']
        assert gmr_baseline[name, 'standard']['threshold'] == gmr_baseline[name, 'full']['threshold']
        assert float(standard['delta_G1']) > 0 and float(full['delta_G1']) > 0

    def metric_value(name, split, series):
        if series == 'gmr_baseline':
            return float(gmr_baseline[name, split]['G1'])
        if series == 'geq_only_reference':
            return float(reference[name, split]['G1'])
        return float(data[name, split][f'{series}_G1'])

    def threshold(name, split, series):
        if series == 'gmr_baseline':
            return float(gmr_baseline[name, split]['threshold'])
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
        'axes.titlesize': 12,
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
        'Title': '86% Val-retention G-mIoU@1 summaries (4 methods)',
        'CreationDate': None,
        'ModDate': None,
    }) as pages:
        for split, stem, title in (
            ('standard', 'STANDARD_SUMMARY', 'Standard Test | 1,036 queries'),
            ('full', 'FULL_SUMMARY', 'Full-Decodable Test | 2,970 / 2,986 queries'),
        ):
            fig, ax = plt.subplots(figsize=(11.5, 9.6))
            fig.subplots_adjust(left=.18, right=.97, top=.78, bottom=.17)
            fig.suptitle(title, fontsize=17, fontweight='bold', y=.976)
            fig.text(
                .585, .925,
                'G-mIoU@1 at each method\'s independently Val-selected existence gate',
                ha='center', fontsize=11.5,
            )
            fig.text(
                .585, .885,
                '86% Val positive-retention target (≥220/255) | selected 9/11 exploratory subset',
                ha='center', fontsize=10, color='#475569',
            )
            handles = [
                Line2D([], [], color=COLORS[index], marker=MARKERS[index],
                       markeredgecolor=EDGES[index], linestyle='none', markersize=8,
                       label=SERIES_LABELS[index])
                for index in range(4)
            ]
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.585, .855),
                       ncol=4, frameon=False, fontsize=10.5)
            y = np.arange(len(NAMES))
            values_by_series = {}
            offsets = [-0.285, -0.095, 0.095, 0.285]
            for index, series in enumerate(SERIES):
                values = np.array([metric_value(name, split, series) for name in NAMES])
                values_by_series[series] = values
                centers = y + offsets[index]
                bars = ax.barh(
                    centers, values, height=.165, color=COLORS[index],
                    edgecolor=EDGES[index], linewidth=.55, zorder=2,
                )
                ax.scatter(values, centers, marker=MARKERS[index], s=16,
                           color=EDGES[index], zorder=3)
                assert np.array_equal(np.array([bar.get_width() for bar in bars]), values)
                for name, value, center in zip(NAMES, values, centers):
                    base_val = metric_value(name, split, 'gmr_baseline')
                    aux_val = metric_value(name, split, 'gmr')
                    ref_val = metric_value(name, split, 'geq_only_reference')
                    geq_val = metric_value(name, split, 'geq_full')

                    delta_aux = aux_val - base_val
                    delta_geq_noaux = ref_val - base_val
                    delta_geq_aux = geq_val - aux_val

                    if series == 'gmr_baseline':
                        label = f'{value:.2f}'
                    elif series == 'gmr':
                        label = f'{value:.2f}'
                    elif series == 'geq_only_reference':
                        label = f'{value:.2f}  (Δ {delta_geq_noaux:+.2f})'
                    elif series == 'geq_full':
                        label = f'{value:.2f}  (Δ {delta_geq_aux:+.2f})'

                    ax.text(value + .38, center, label, va='center', fontsize=7.8,
                            color='#333333')
                    plotted.append({
                        'backbone': name,
                        'split': split,
                        'series': series,
                        'metric': 'G-mIoU@1',
                        'value': float(value),
                        'threshold': threshold(name, split, series),
                        'delta_aux': delta_aux,
                        'delta_geq_noaux': delta_geq_noaux,
                        'delta_geq_aux': delta_geq_aux,
                        'label': label,
                    })
            baseline_mean = float(values_by_series['gmr_baseline'].mean())
            aux_mean = float(values_by_series['gmr'].mean())
            reference_mean = float(values_by_series['geq_only_reference'].mean())
            full_mean = float(values_by_series['geq_full'].mean())
            d_aux_mean = aux_mean - baseline_mean
            d_geq_noaux_mean = reference_mean - baseline_mean
            d_geq_aux_mean = full_mean - aux_mean

            ax.set_title(
                f'9-backbone mean: GMR base {baseline_mean:.2f} | GMR(aux) {aux_mean:.2f} | '
                f'GEQ ref {reference_mean:.2f} (ΔGEQ {d_geq_noaux_mean:+.2f}) | '
                f'GEQ+aux {full_mean:.2f} (ΔGEQ {d_geq_aux_mean:+.2f})',
                fontsize=10.2, pad=12, color='#334155',
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
                .585, .068,
                'Each backbone/method selects its own largest 0.01-grid threshold on Standard Val; '
                'the same threshold is used on both Tests.\n'
                'Subset is fixed by prior exploratory gains; Moment-DETR and LD-DETR are omitted from display.\n'
                'Paired comparisons: GMR baseline → GEQ only (ΔGEQ no-aux), GMR(aux) → GEQ+aux (ΔGEQ aux). '
                'ΔAux is descriptive reference across different parent/protocol baselines.',
                ha='center', fontsize=8.2, linespacing=1.45, color='#475569',
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
    assert len(plotted) == 72
    plot_data_path = OUT / 'SUMMARY_PLOT_DATA.json'
    plot_data_path.write_text(json.dumps(plotted, indent=2) + '\n')
    audit = {
        'passed': True,
        'exploratory': True,
        'methods_count': 4,
        'methods': list(SERIES),
        'method_labels': list(SERIES_LABELS),
        'source_metrics_sha256': sha(OUT / 'RETENTION86_METRICS.csv'),
        'source_evaluation_audit_sha256': sha(OUT / 'RETENTION86_AUDIT.json'),
        'reference_metrics_sha256': sha(OUT / 'GEQ_ONLY_RETENTION86_METRICS.csv'),
        'reference_evaluation_audit_sha256': sha(OUT / 'GEQ_ONLY_RETENTION86_AUDIT.json'),
        'gmr_baseline_metrics_sha256': sha(OUT / 'GMR_BASELINE_RETENTION86_METRICS.csv'),
        'gmr_baseline_evaluation_audit_sha256': sha(OUT / 'GMR_BASELINE_RETENTION86_AUDIT.json'),
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
        'bars_checked_against_CSV': 72,
        'numeric_value_labels': 72,
        'delta_geq_noaux_labels': 18,
        'delta_geq_aux_labels': 18,
        'same_threshold_across_both_tests_checked': True,
        'label_canvas_bounds_checked': True,
        'checkpoint_reselection': False,
        'new_training': False,
        'plot_data_sha256': sha(plot_data_path),
        'full_curve_files_unchanged': curve_hashes,
        'files': {path.name: sha(path) for path in figure_files},
    }
    (OUT / 'SUMMARY_PLOT_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Rendered two 86%-retention summaries: 9 backbones, 72 bars, all labels validated.')


if __name__ == '__main__':
    main()
