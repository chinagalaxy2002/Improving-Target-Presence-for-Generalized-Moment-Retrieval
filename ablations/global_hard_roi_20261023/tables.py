"""Export all four cells, nine metrics and displayed GMR-relative gains."""
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize import BACKBONES, SETTINGS, METRICS, delta, RESULTS

ROOT = Path(__file__).resolve().parent
LABELS = {'gmr': 'GMR', 'global_only': '+ Global Query', 'hard_only': '+ Hard-RoI', 'global_hard': '+ Global Query + Hard-RoI'}


def main():
    out = ROOT / 'tables'
    out.mkdir(exist_ok=True)
    rows = []
    for backbone in BACKBONES:
        for setting in SETTINGS:
            path = RESULTS / backbone / setting
            assert (path / 'COMPLETE.json').exists()
            for split in ['test', 'val']:
                for tau, report in json.loads((path / f'{split}_metrics.json').read_text()).items():
                    brief = report['brief']
                    rows.append(dict(backbone=backbone, setting=setting, split=split, tau=tau,
                                     **{k: brief['Rej-F1@' + tau if k == 'Rej-F1' else k] for k in METRICS}))
    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})
    for split in ['test', 'val']:
        for tau in ['0.4', '0.5', '0.6']:
            selected = [r for r in rows if r['split'] == split and r['tau'] == tau]
            baseline = {r['backbone']: r for r in selected if r['setting'] == 'gmr'}
            fig, ax = plt.subplots(figsize=(18, 12))
            ax.axis('off')
            columns = ['Backbone', 'Configuration'] + METRICS
            body = []
            gains = []
            for row in selected:
                d = delta(row, baseline[row['backbone']])
                cell = lambda k: f'{row[k]:.2f}' + (f'\n({d[k]:+.2f})' if row['setting'] != 'gmr' else '')
                body.append([BACKBONES[row['backbone']] if row['setting'] == 'gmr' else '', LABELS[row['setting']]] + [cell(k) for k in METRICS])
                gains.append(d)
            table = ax.table(cellText=body, colLabels=columns, cellLoc='center',
                             colWidths=[.105, .23] + [.0738] * 9, bbox=[0, .075, 1, .86])
            table.auto_set_font_size(False)
            table.set_fontsize(10.5)
            for (i, j), cell in table.get_celld().items():
                cell.set_edgecolor('#d7dfe5')
                cell.set_linewidth(.4)
                if i == 0:
                    cell.set_facecolor('#e3ecf2')
                    cell.set_text_props(weight='bold')
                elif (i - 1) % 4 == 3:
                    cell.set_facecolor('#edf5f9')
                if j < 2:
                    cell.set_text_props(ha='left')
            for bi, backbone in enumerate(BACKBONES):
                chunk = selected[bi * 4:bi * 4 + 4]
                for mi, metric in enumerate(METRICS, 2):
                    best = max(round(r[metric], 2) for r in chunk)
                    for ri, row in enumerate(chunk, bi * 4 + 1):
                        if round(row[metric], 2) == best:
                            table[ri, mi].set_text_props(weight='bold')
            ax.text(0, .98, f'Global Event Query + Hard-RoI | {split.title()} | tau = {tau}', transform=ax.transAxes, fontsize=18, weight='bold', va='top')
            ax.text(0, .945, '5 backbones x 4 cells; fixed epoch 4; seed 2026; original GMR parameters frozen', transform=ax.transAxes, fontsize=11)
            note = ('All values: %. Parentheses: gain versus same-backbone GMR (pp), subtracting displayed values. Bold: within-backbone best.\n'
                    'LoRA / Gaussian-RoI / 6D Saliency / candidate presence fusion: OFF. Baseline and Global-only are verified reused cells.\n'
                    '10 new Hard-only / Global+Hard runs; baseline: 0 new epochs. Standard data: 4,138 train / 465 val / 1,036 test.\n'
                    'Exploratory single-seed comparisons; no test checkpoint/threshold selection. Native FlashVTG score calibration and NMS retained.')
            ax.text(0, .052, note, transform=ax.transAxes, fontsize=10, va='top', linespacing=1.6)
            fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.025)
            stem = out / f'ABLATION_{split.upper()}_tau_{tau}'
            for ext in ['pdf', 'png', 'svg']:
                fig.savefig(str(stem) + '.' + ext, dpi=180, bbox_inches='tight')
            plt.close(fig)
            header = '<tr>' + ''.join('<th>' + html.escape(c) + '</th>' for c in columns) + '</tr>'
            content = '\n'.join('<tr>' + ''.join('<td>' + html.escape(c).replace('\n', '<br>') + '</td>' for c in row) + '</tr>' for row in body)
            Path(str(stem) + '.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:14px sans-serif}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccc;text-align:center}</style><h1>' + html.escape(f'{split.title()} tau={tau}') + '</h1><table>' + header + content + '</table><p>' + html.escape(note).replace('\n', '<br>') + '</p>')
    print('Exported six nine-metric tables as PDF / PNG / SVG / HTML')


if __name__ == '__main__':
    main()
