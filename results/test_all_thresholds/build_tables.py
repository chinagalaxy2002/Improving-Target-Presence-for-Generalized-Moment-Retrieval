#!/usr/bin/env python3
"""Build Test-only method tables from frozen exports, with no model imports."""
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BRANCH = 'gmr-mr-detr-multiroute-20261030'
SOURCE_COMMIT = '22a882d45eb9cdc15f343923b41d876c5f069092'
REFERENCE = 'https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/main/ablations/component_ablation_20261023/tables/ABLATION_TEST_ALL_THRESHOLDS.tex'
BACKBONES = ('moment_detr', 'eatr', 'qd_detr', 'cg_detr')
NAMES = dict(moment_detr='Moment-DETR', eatr='EaTR', qd_detr='QD-DETR', cg_detr='CG-DETR', flashvtg='FlashVTG')
MAIN = ('gmr', 'gmr_geq_frozen', 'gmr_geq_full', 'multi', 'multi_geq_frozen', 'multi_geq_full')
LABELS = dict(gmr='GMR', multi='GMR + Multi', gmr_geq_frozen='GMR + GEQ Frozen',
              gmr_geq_full='GMR + GEQ Full', multi_geq_frozen='GMR + Multi + GEQ Frozen',
              multi_geq_full='GMR + Multi + GEQ Full', shared='GMR + Shared',
              shared_geq_frozen='GMR + Shared + GEQ Frozen', shared_geq_full='GMR + Shared + GEQ Full',
              sepffn='GMR + SepFFN', instruction='GMR + Instruction', adaptive='GMR + Adaptive',
              multi_positive_only='GMR + Multi (positive-only)')
COMMON = ('AUROC', 'mAP', 'mR@1', 'mR@5', 'mR+@5')
THRESHOLD_METRICS = ('N-acc', 'Rej-F1', 'G-mIoU@1', 'G-mIoU@3')
TAUS = (0.4, 0.5, 0.6)
INPUTS = {}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    p = Path(p)
    INPUTS[str(p.relative_to(REPO))] = sha(p)
    return json.loads(p.read_text())


def write_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(p, records):
    with p.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)


def make_row(dataset, split, protocol, backbone, method, source, data, parent=None, label=None, charades=False):
    if parent is None:
        parent = 'multi' if method.startswith('multi_geq') else 'shared' if method.startswith('shared_geq') else 'gmr'
    metrics = data['official']['brief'] if charades else data['metrics']
    common = {k: (None if charades and k == 'mR+@5' else metrics[k]) for k in COMMON}
    points = {}
    for tau in TAUS:
        point = next(g for g in data['grid'] if math.isclose(g['threshold'], tau, abs_tol=1e-10))
        points[str(tau)] = {'N-acc': 100*point['negative_rejection' if charades else 'negative_accuracy'],
                            'Rej-F1': 100*point['rejection_f1'],
                            'G-mIoU@1': point['gmiou_1'], 'G-mIoU@3': point['gmiou_3']}
    if label is None:
        label = LABELS[method]
        if backbone == 'flashvtg':
            label = label.replace('Multi', 'Multi-TQAux')
    return dict(dataset=dataset, split=split, protocol=protocol, backbone=backbone,
                method=method, label=label, parent=parent, common=common, thresholds=points, source=source)


def soccer_current(methods):
    output = []
    for b in BACKBONES:
        for m in methods:
            base = REPO / 'results/geq_val_selection' / b
            if m == 'gmr':
                p = base / 'full_comparison_baseline_data.json'
            elif m in ('gmr_geq_frozen', 'gmr_geq_full'):
                p = base / (('frozen' if m.endswith('frozen') else 'full') + '_comparison_baseline_geq_data.json')
            elif m in ('multi', 'shared'):
                p = base / (m + '_full_frozen_baseline_data.json')
            elif m.startswith(('multi_geq_', 'shared_geq_')):
                route, _, mode = m.split('_')
                p = base / (route + '_full_frozen_' + mode + '_data.json')
            else:
                p = HERE / 'supplemental/soccergmr' / b / m / 'data.json'
            data = read(p)
            output.append(make_row('SoccerGMR', 'Standard Test', 'current_val_selection', b, m,
                                   str(p.relative_to(REPO)), data))
    return output


def charades(split):
    output = []
    for b in ('moment_detr', 'eatr', 'qd_detr', 'flashvtg'):
        for m in MAIN + (('multi_positive_only',) if b == 'eatr' else ()):
            root = HERE / 'supplemental/charades' if m == 'multi_positive_only' else REPO / 'results/charades_multi_geq/run_metadata'
            p = root / b / m / split / 'metrics.json'
            data = read(p)
            marker = read(p.with_name('COMPLETE.json'))
            assert marker['metrics_sha256'] == sha(p)
            assert marker['rows'] == data['coverage']['rows'] == 7440
            output.append(make_row('Charades', split, 'current_val_selection', b, m,
                                   str(p.relative_to(REPO)), data, parent='multi' if m == 'multi_positive_only' else None, charades=True))
    return output


def soccer_history():
    e1, fixed = [], []
    for b in BACKBONES:
        for m, filename in (('gmr', 'baseline'), ('gmr_geq_frozen', 'frozen'), ('gmr_geq_full', 'full')):
            p = REPO / 'results/geq_epoch_study' / b / ('full_frozen_comparison_' + filename + '_data.json')
            e1.append(make_row('SoccerGMR', 'Standard Test', 'historical_main_e1', b, m,
                               str(p.relative_to(REPO)), read(p)))
        p = REPO / 'results/geq_finetune' / b / 'four_way_comparison_data.json'
        data = read(p)
        for key, m in (('baseline', 'gmr'), ('baseline_geq', 'gmr_geq_frozen'),
                       ('multi_geq', 'multi_geq_frozen'), ('shared_geq', 'shared_geq_frozen')):
            fixed.append(make_row('SoccerGMR', 'Standard Test', 'historical_fixed_four_epochs', b, m,
                                  str(p.relative_to(REPO)) + '#series/' + key, data['series'][key]))
    return e1, fixed


def fmt(value, metric):
    if value is None:
        return 'N/A'
    return f'{value:.4f}' if metric.startswith('G-mIoU') else f'{value:.2f}'


def cells(row):
    label = row['label'] + (' (baseline)' if row['method'] == 'gmr' else '')
    return [NAMES[row['backbone']], label] + [fmt(row['common'][m], m) for m in COMMON] + [
        fmt(row['thresholds'][str(t)][m], m) for t in TAUS for m in THRESHOLD_METRICS]


def metric_values(row):
    return [(m, row['common'][m]) for m in COMMON] + [
        (m, row['thresholds'][str(t)][m]) for t in TAUS for m in THRESHOLD_METRICS]


def baseline_for(row, index):
    return index[(row['dataset'], row['split'], row['protocol'], row['backbone'], 'gmr')]


def delta(value, baseline, metric):
    if value is None or baseline is None:
        return None
    return round(value - baseline, 4 if metric.startswith('G-mIoU') else 2) + 0.0


def delta_label(value, metric):
    if value is None:
        return ''
    precision = 4 if metric.startswith('G-mIoU') else 2
    return '(' + (f'{value:+.{precision}f}' if value else f'{value:.{precision}f}') + ')'


def display_rows(rows, index):
    """Show the actual GMR reference even in the other-methods appendix."""
    output = []
    for b in dict.fromkeys(r['backbone'] for r in rows):
        group = [r for r in rows if r['backbone'] == b]
        if not any(r['method'] == 'gmr' for r in group):
            output.append(baseline_for(group[0], index))
        output.extend(group)
    return output


def tex_escape(s):
    return s.replace('&', r'\&').replace('_', r'\_').replace('%', r'\%')


def tex_table(title, rows, note, index):
    out = [r'\noindent\textbf{' + tex_escape(title) + r'}\par\medskip',
           r'\noindent\resizebox{\textwidth}{!}{\begin{tabular}{ll' + 'r'*17 + r'}\toprule',
           r'\multicolumn{2}{l}{} & \multicolumn{5}{c}{Threshold-independent (\%)} & ' +
           ' & '.join(r'\multicolumn{4}{c}{$\tau=' + str(t) + '$}' for t in TAUS) + r' \\',
           r'\cmidrule(lr){3-7}\cmidrule(lr){8-11}\cmidrule(lr){12-15}\cmidrule(lr){16-19}',
           'Backbone & Method & ' + ' & '.join(tex_escape(m) for m in COMMON) + ' & ' +
           ' & '.join(tex_escape(m) for _ in TAUS for m in THRESHOLD_METRICS) + r' \\\midrule']
    previous = None
    for row in rows:
        if previous is not None and previous != row['backbone']:
            out.append(r'\midrule')
        values = cells(row)
        if previous == row['backbone']:
            values[0] = ''
        formatted = [tex_escape(v) for v in values[:2]]
        baseline = baseline_for(row, index)
        for j, ((metric, value), (_, reference)) in enumerate(zip(metric_values(row), metric_values(baseline))):
            absolute = tex_escape(values[j+2])
            if row['method'] == 'gmr':
                formatted.append(r'\textbf{' + absolute + '}')
            elif value is None:
                formatted.append(absolute)
            else:
                change = delta(value, reference, metric)
                color = 'GMRGain' if change > 0 else 'GMRLoss' if change < 0 else 'GMRZero'
                formatted.append(r'\shortstack{' + absolute + r' \\ {\scriptsize\textcolor{' + color + '}{' + delta_label(change, metric) + '}}}')
        out.append(' & '.join(formatted) + r' \\')
        previous = row['backbone']
    out += [r'\bottomrule\end{tabular}}\par\medskip',
            r'\noindent G-mIoU@1/@3: 0--1. All other numeric metrics: percent. N/A: not applicable.\par',
            r'\noindent Parentheses: method minus GMR baseline in the same dataset, Test split, protocol and backbone. '
            r'\textcolor{GMRGain}{Green: gain}; \textcolor{GMRLoss}{red: decrease}; gray: unchanged.\par',
            r'\noindent ' + tex_escape(note) + r'\par']
    return '\n'.join(out)


def render(title, rows, note, index):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
    widths = [118, 258] + [75]*5 + [77]*12
    width = sum(widths)
    row_height = 34
    header = 100
    height = header + len(rows)*row_height + 126
    fig = plt.figure(figsize=((width+48)/72, height/72), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-24, width+24)
    ax.set_ylim(height, 0)
    ax.axis('off')
    ax.text(0, 23, title, fontsize=14, fontweight='bold', va='center', color='#172b3a')
    ax.text(0, 43, 'Test value with (delta vs GMR baseline) below  |  G-mIoU: 0-1  |  Other values / deltas: % / percentage points', fontsize=9, color='#475569')
    edges = [0]
    for w in widths:
        edges.append(edges[-1] + w)
    ax.plot([0, width], [57, 57], color='#172b3a', lw=1.2)
    groups = [(2, 7, 'Threshold-independent (%)')] + [(7+i*4, 11+i*4, f'Threshold = {t:.1f}') for i, t in enumerate(TAUS)]
    for start, end, label in groups:
        ax.text((edges[start]+edges[end])/2, 70, label, ha='center', va='center', fontsize=10, fontweight='bold')
        ax.plot([edges[start]+4, edges[end]-4], [80, 80], color='#94a3b8', lw=0.65)
    headers = ['Backbone', 'Method'] + list(COMMON) + list(THRESHOLD_METRICS)*3
    for j, label in enumerate(headers):
        ax.text(edges[j]+4 if j < 2 else (edges[j]+edges[j+1])/2, 91, label,
                ha='left' if j < 2 else 'center', va='center', fontsize=8.5, fontweight='bold')
    ax.plot([0, width], [header, header], color='#172b3a', lw=1)
    last = None
    group = -1
    for i, row in enumerate(rows):
        y = header + i*row_height
        if row['backbone'] != last:
            group += 1
            if last is not None:
                ax.plot([0, width], [y, y], color='#64748b', lw=0.75)
        if group % 2 == 1:
            ax.add_patch(plt.Rectangle((0, y), width, row_height, facecolor='#f1f5f9', edgecolor='none'))
        if row['method'] == 'gmr':
            ax.add_patch(plt.Rectangle((0, y), width, row_height, facecolor='#e2edf8', edgecolor='none'))
        values = cells(row)
        if last == row['backbone']:
            values[0] = ''
        for j, value in enumerate(values):
            ax.text(edges[j]+4 if j < 2 else edges[j+1]-8, y+row_height/2 if j < 2 or row['method'] == 'gmr' or value == 'N/A' else y+10, value,
                    ha='left' if j < 2 else 'right', va='center', fontsize=8.7,
                    fontweight='bold' if row['method'] == 'gmr' or j == 0 and row['backbone'] != last else 'normal', color='#172b3a')
        if row['method'] != 'gmr':
            baseline = baseline_for(row, index)
            for j, ((metric, value), (_, reference)) in enumerate(zip(metric_values(row), metric_values(baseline)), 2):
                change = delta(value, reference, metric)
                color = '#15803d' if change is not None and change > 0 else '#b91c1c' if change is not None and change < 0 else '#64748b'
                ax.text(edges[j+1]-8, y+25, delta_label(change, metric), ha='right', va='center',
                        fontsize=7.8, color=color, fontweight='bold' if change else 'normal')
        last = row['backbone']
    bottom = header + len(rows)*row_height
    ax.plot([0, width], [bottom, bottom], color='#172b3a', lw=1.2)
    for j in (2, 7, 11, 15):
        ax.plot([edges[j], edges[j]], [82, bottom], color='#cbd5e1', lw=0.5)
    ax.text(0, bottom+20, 'Delta = method - GMR baseline in the SAME dataset / Test split / protocol / backbone.', fontsize=9, va='top', color='#172b3a', fontweight='bold')
    ax.text(0, bottom+39, 'Green: gain', fontsize=9, va='top', color='#15803d', fontweight='bold')
    ax.text(105, bottom+39, 'Red: decrease', fontsize=9, va='top', color='#b91c1c')
    ax.text(225, bottom+39, 'Gray: unchanged. Deltas use source precision, then round for display. N/A: not applicable.', fontsize=9, va='top', color='#64748b')
    ax.text(0, bottom+60, note, fontsize=9, va='top', color='#475569')
    ax.text(0, bottom+81, 'N-acc = TN / negative queries; Rej-F1 treats rejection as the target class. Threshold = existence-score threshold.', fontsize=9, va='top', color='#475569')
    ax.text(0, bottom+102, f'Source branch: {BRANCH} | Base: {SOURCE_COMMIT[:12]} + verified supplemental Test exports', fontsize=8, va='top', color='#64748b')
    return fig


def main():
    soccer_main = soccer_current(MAIN)
    soccer_other = soccer_current(('shared', 'shared_geq_frozen', 'shared_geq_full', 'sepffn', 'instruction', 'adaptive'))
    ci, co = charades('test_id'), charades('test_ood')
    e1, fixed = soccer_history()
    groups = [
        ('SOCCERGMR_TEST', 'SoccerGMR Standard Test - GMR / Multi methods', soccer_main, 'GEQ checkpoints selected by Val G-mIoU@1 mean at 0.4 / 0.5 / 0.6. Native parent training budgets differ.'),
        ('CHARADES_TEST_ID', 'Charades Test-ID - existing methods', ci, '24 primary methods plus the previously trained EaTR positive-only variant. Single-GT protocol: mR+@5 is N/A.'),
        ('CHARADES_TEST_OOD', 'Charades Test-OOD - existing methods', co, '7440 weighted query rows (3720 positive / 3720 negative). Same positive pool as Test-ID; presence results differ.'),
        ('SOCCERGMR_OTHER_METHODS_TEST', 'SoccerGMR Standard Test - other existing methods', soccer_other, 'Includes the completed EaTR Adaptive Test omitted from the old branch snapshot. Missing threshold reports supplemented from formal predictions.'),
        ('SOCCERGMR_HISTORICAL_E1_TEST', 'SoccerGMR Standard Test - historical main E1 initialization', e1, 'Historical parent checkpoints. Compare within this table; do not mix their GMR values with the current protocol.'),
        ('SOCCERGMR_HISTORICAL_FIXED4_TEST', 'SoccerGMR Standard Test - historical fixed-four-epoch GEQ', fixed, 'Historical frozen GEQ after four fixed epochs, without the newer Val checkpoint-selection protocol.'),
    ]
    all_rows = [r for _, _, rr, _ in groups for r in rr]
    index = {(r['dataset'], r['split'], r['protocol'], r['backbone'], r['method']): r for r in all_rows}
    assert len(all_rows) == 126
    assert len(soccer_main+soccer_other) == 48
    assert len(ci) == len(co) == 25
    long = []
    wide = []
    for r in all_rows:
        info = {k: r[k] for k in ('dataset', 'split', 'protocol', 'backbone', 'method', 'label', 'source')}
        wide_row = dict(info, **r['common'])
        for t in TAUS:
            point = r['thresholds'][str(t)]
            for metric, value in point.items():
                assert value is None or (math.isfinite(value) and 0 <= value <= (1 if metric.startswith('G-mIoU') else 100))
                wide_row[f'{metric}@tau={t}'] = value
            long.append(dict(info, threshold=t, **r['common'], **point))
        wide.append(wide_row)
    write_json(HERE / 'TEST_ALL_THRESHOLDS.json', all_rows)
    write_csv(HERE / 'TEST_ALL_THRESHOLDS.csv', wide)
    write_csv(HERE / 'TEST_ALL_THRESHOLDS_LONG.csv', long)
    preamble = '\n'.join([r'\documentclass{article}', r'\usepackage[paperwidth=1830pt,paperheight=940pt,margin=24pt]{geometry}',
                          r'\usepackage[T1]{fontenc}', r'\usepackage{booktabs,graphicx,xcolor}',
                          r'\definecolor{GMRGain}{HTML}{15803D}\definecolor{GMRLoss}{HTML}{B91C1C}\definecolor{GMRZero}{HTML}{64748B}',
                          r'\pagestyle{empty}', r'\begin{document}'])
    tex_tables = []
    with PdfPages(HERE / 'TEST_ALL_THRESHOLDS.pdf', metadata={'Title': 'Two-dataset Test method comparison', 'CreationDate': None, 'ModDate': None}) as pdf:
        for stem, title, rr, note in groups:
            displayed = display_rows(rr, index)
            table = tex_table(title, displayed, note, index)
            tex_tables.append(table)
            (HERE / (stem + '.tex')).write_text(preamble + '\n' + table + '\n' + r'\end{document}' + '\n')
            fig = render(title, displayed, note, index)
            fig.savefig(HERE / (stem+'.pdf'), metadata={'CreationDate': None, 'ModDate': None})
            fig.savefig(HERE / (stem+'.png'), dpi=150)
            pdf.savefig(fig)
            plt.close(fig)
    (HERE / 'TEST_ALL_THRESHOLDS.tex').write_text(preamble + '\n' + ('\n' + r'\newpage' + '\n').join(tex_tables) + '\n' + r'\end{document}' + '\n')

    deltas = []
    metric_deltas = []
    for r in all_rows:
        prefix = (r['dataset'], r['split'], r['protocol'], r['backbone'])
        baseline = baseline_for(r, index)
        parent = index.get(prefix+(r['parent'],))
        if parent is None and r['protocol'] == 'historical_fixed_four_epochs':
            parent = index[(r['dataset'], r['split'], 'current_val_selection', r['backbone'], r['parent'])]
        assert parent is not None
        delta_parent = round(r['common']['mAP']-parent['common']['mAP'], 8)
        if r['method'].endswith('_frozen'):
            assert delta_parent == 0, (r['dataset'], r['protocol'], r['backbone'], r['method'], delta_parent)
        deltas.append(dict(dataset=r['dataset'], split=r['split'], protocol=r['protocol'], backbone=r['backbone'],
                           method=r['label'], mAP_percent=r['common']['mAP'], GMR_mAP_percent=baseline['common']['mAP'],
                           delta_vs_GMR_pp=round(r['common']['mAP']-baseline['common']['mAP'], 8)))
        for tau in TAUS:
            for metric, value in dict(r['common'], **r['thresholds'][str(tau)]).items():
                reference = baseline['common'][metric] if metric in COMMON else baseline['thresholds'][str(tau)][metric]
                # Threshold-independent metrics have one delta record per method.
                if metric in COMMON and tau != TAUS[0]:
                    continue
                metric_deltas.append(dict(dataset=r['dataset'], split=r['split'], protocol=r['protocol'],
                    backbone=r['backbone'], method=r['method'], metric=metric,
                    threshold='' if metric in COMMON else tau, test_value=value, GMR_baseline=reference,
                    delta_vs_GMR=None if value is None else value-reference,
                    display_delta=delta(value, reference, metric),
                    unit='0-1' if metric.startswith('G-mIoU') else 'percentage points',
                    baseline_source=baseline['source']))
    write_csv(HERE / 'MAP_DELTAS.csv', deltas)
    write_csv(HERE / 'TEST_DELTAS_VS_GMR.csv', metric_deltas)
    manifest = dict(source_branch=BRANCH, source_commit=SOURCE_COMMIT, reference_url=REFERENCE,
                    reference_git_blob_sha='1a3f44c4aabce95fa7eca4140ec9dc23d5e81545',
                    method_results=len(all_rows), current_soccer_results=48, current_charades_results=50,
                    historical_results=28, fixed_thresholds=list(TAUS), long_rows=len(long),
                    unit_policy='G-mIoU 0-1; other metrics percent; deltas percentage points',
                    all_results_are_test=True, missing_formal_test=0, test_used_for_selection=False,
                    comparison_reference='GMR baseline in the same dataset, Test split, protocol and backbone',
                    delta_colors=dict(positive='#15803d', negative='#b91c1c', zero='#64748b'),
                    inputs=INPUTS)
    write_json(HERE / 'PROVENANCE.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'inputs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
