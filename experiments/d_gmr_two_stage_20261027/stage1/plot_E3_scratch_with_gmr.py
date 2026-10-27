#!/usr/bin/env python3
"""Analysis-only Test threshold sweeps: real checkpoints for FlashVTG Scratch (E3) + Pretrained GMR baseline."""
import csv, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from independent_metrics import gmiou

DEST = ROOT / 'figures/FlashVTG_E3_scratch_curves'
DEST.mkdir(parents=True, exist_ok=True)

def read(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

gt = read(ROOT / 'data/test.jsonl')
ids = [g['qid'] for g in gt]
positive = np.array([bool(g['relevant_windows']) for g in gt])
assert len(gt) == 1036 and positive.sum() == 544

paths = {
    'GMR': ROOT / 'reused_baselines/flashvtg',
    26: ROOT / 'runs/flashvtg/E3',
    63: ROOT / 'runs/flashvtg/E3_extra50/epoch_063',
    70: ROOT / 'runs/flashvtg/E3_extra50/epoch_070',
    76: ROOT / 'runs/flashvtg/E3_extra50'
}

labels = {
    'GMR': 'Pretrained GMR (E0)',
    26: 'E3 epoch 26 (Scratch base)',
    63: 'E3 epoch 63 (Best Val Total)',
    70: 'E3 epoch 70 (Best Val Loc)',
    76: 'E3 epoch 76 (Scratch final)'
}

pred = {e: {p['qid']: p for p in read(d / 'test_predictions.jsonl')} for e, d in paths.items()}
assert all(set(p) == set(ids) for p in pred.values())

tau = np.linspace(0, 1, 501)
scores = {e: np.array([pred[e][i]['pred_exist_score'] for i in ids]) for e in paths}

local = {}
for e in paths:
    local[e] = np.array([
        gmiou([dict(pred[e][g['qid']], pred_exist_score=1.)], [g], 0.)['G-mIoU@1'] / 100.
        if g['relevant_windows'] else 0.
        for g in gt
    ])
    for t in (.4, .5, .6):
        value = 100 * np.mean(np.where(scores[e] > t, local[e], (~positive)))
        official = json.loads((paths[e] / 'test_metrics.json').read_text())[str(t)]['brief']['G-mIoU@1']
        assert abs(value - official) <= .011, (e, t, value, official)

styles = {
    'GMR': ('#000000', ':', 'x'),
    26: ('#4C72B0', '-', 'o'),
    63: ('#55A868', '--', 's'),
    70: ('#C44E52', '-.', 'D'),
    76: ('#8172B3', '-', '^')
}

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 15, 'axes.labelsize': 13,
    'legend.fontsize': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'
})

all_records = []
name = 'checkpoint_specific_localizers'
curves = {
    e: 100 * np.mean(np.where(scores[e][None, :] > tau[:, None], local[e][None, :], (~positive)[None, :]), axis=1)
    for e in paths
}
oracle = {e: 100 * np.mean(np.where(positive, local[e], 1.)) for e in paths}

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
a, b = axes

for e in ['GMR', 26, 63, 70, 76]:
    color, ls, marker = styles[e]
    lw = 2.0 if e == 'GMR' else 2.4
    a.plot(tau, curves[e], color=color, ls=ls, lw=lw, marker=marker, markevery=50, ms=5, label=labels[e])
    if e != 'GMR':
        b.plot(tau, curves[e] - curves['GMR'], color=color, ls=ls, lw=lw, marker=marker, markevery=50, ms=5, label=f'{labels[e]} minus GMR')

for e in ['GMR', 70, 76]:
    a.axhline(oracle[e], color=styles[e][0], ls=':', lw=1.2, alpha=0.7, label=f'{labels[e]} oracle ({oracle[e]:.2f})')
b.plot(tau, oracle['GMR'] - curves['GMR'], color='#8A661C', ls=':', lw=2.0, label='GMR oracle minus GMR')

a.set_title('(a) G-mIoU@1 versus existence threshold', loc='left', weight='bold')
b.set_title('(b) Gain relative to Pretrained GMR (E0)', loc='left', weight='bold')
a.set_ylabel('G-mIoU@1 (%)')
b.set_ylabel('Delta G-mIoU@1 vs GMR (pp)')
a.set_ylim(0, max(oracle.values()) + 6)
b.axhline(0, color='#64748B', ls='--', lw=1.2)

for ax in axes:
    for t in (.4, .5, .6):
        ax.axvline(t, color='#94A3B8', ls=':', lw=1)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel('Existence threshold ' + r'$\tau$')
    ax.grid(axis='y', ls=':', color='#CBD5E1')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='best', framealpha=.94)

fig.suptitle('FlashVTG Scratch (E3) vs Pretrained GMR (E0)\nEach model retains its own temporal windows and existence scores', fontsize=15, y=.99)
fig.text(.5, .025, 'Official Test: N=1,036 (544 positive, 492 negative). Decision: score > threshold.\nOracle uses Test labels for analysis only; no threshold or checkpoint is selected from this sweep.', ha='center', fontsize=10, color='#475569')
fig.subplots_adjust(left=.065, right=.99, bottom=.22, top=.79, wspace=.17)

for ext in ('pdf', 'svg', 'png'):
    fig.savefig(DEST / f'{name}.{ext}', dpi=220, bbox_inches='tight')
plt.close(fig)

for j, t in enumerate(tau):
    for e in paths:
        all_records.append(dict(
            setting=name, model=str(e), threshold=float(t),
            gmiou1=float(curves[e][j]),
            delta_vs_gmr=float(curves[e][j] - curves['GMR'][j]),
            oracle_gmiou1=float(oracle[e])
        ))

with (DEST / 'curves.csv').open('w') as f:
    w = csv.DictWriter(f, fieldnames=list(all_records[0]))
    w.writeheader()
    w.writerows(all_records)

print(f"Plot with GMR saved successfully in {DEST}")
