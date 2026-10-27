#!/usr/bin/env python3
"""Reference-style Standard Val/Test distributions from frozen predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
BACKBONES = ('moment_detr', 'qd_detr', 'cg_detr', 'eatr', 'flashvtg')
NAMES = dict(moment_detr='Moment-DETR', qd_detr='QD-DETR', cg_detr='CG-DETR', eatr='EaTR', flashvtg='FlashVTG')
METHODS = (('GMR', '#0072B2', '-'), ('E1', '#D55E00', '--'), ('E2', '#009E73', '-.'))
REFERENCE = Path('/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/experiments/gmr-Global-Eventquery/figures/full_standard_bn45_val_existence_score_distributions.pdf')

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    split = parser.parse_args().split
    display = 'Validation' if split == 'val' else 'Test'
    expected_total, expected_positive = (465, 255) if split == 'val' else (1036, 544)
    destination = ROOT/'figures'; destination.mkdir(exist_ok=True)
    stem = destination/f'standard_{split}_existence_score_distributions'
    truth_path = ROOT/f'data/{split}.jsonl'; truth = rows(truth_path)
    qids = [x['qid'] for x in truth]; target = np.array([bool(x['relevant_windows']) for x in truth])
    assert len(qids) == len(set(qids)) == expected_total and target.sum() == expected_positive
    bins = np.linspace(0, 1, 41)
    scores = {}; provenance = {str(truth_path): hashlib.sha256(truth_path.read_bytes()).hexdigest()}
    stats = []; histogram = []
    for backbone in BACKBONES:
        for method, _, _ in METHODS:
            path = ROOT/(f'reused_baselines/{backbone}/{split}_predictions.jsonl' if method == 'GMR' else f'runs/{backbone}/{method}/{split}_predictions.jsonl')
            if method != 'GMR':
                marker = json.loads((path.parent/'COMPLETE.json').read_text())
                assert marker['epoch'] == 4 and marker['seed'] == 2026
            records = rows(path); index = {p['qid']: p for p in records}
            assert len(records) == len(index) == len(qids) and set(index) == set(qids)
            values = np.array([index[qid]['pred_exist_score'] for qid in qids], dtype=float)
            assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
            scores[backbone, method] = values
            provenance[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for group, mask in [('positive', target), ('negative', ~target)]:
                v = values[mask]; density, _ = np.histogram(v, bins=bins, density=True)
                counts, _ = np.histogram(v, bins=bins)
                assert counts.sum() == mask.sum() and np.isclose(np.sum(density*np.diff(bins)), 1.)
                stats.append(dict(Backbone=NAMES[backbone],Method=method,Split=split,Group=group,N=len(v),Mean=float(v.mean()),Median=float(np.median(v)),Std=float(v.std()),AcceptedAt04=int((v > .4).sum()),RejectedAt04=int((v <= .4).sum())))
                histogram.extend(dict(Backbone=NAMES[backbone],Method=method,Group=group,BinStart=bins[i],BinEnd=bins[i+1],Count=int(counts[i]),Density=float(density[i])) for i in range(len(counts)))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.labelcolor':'#25313E','text.color':'#152536','xtick.color':'#334155','ytick.color':'#334155'})
    fig, axes = plt.subplots(5, 2, figsize=(12.2, 13.9), sharex=True)
    fig.subplots_adjust(left=.07,right=.98,bottom=.055,top=.875,hspace=.43,wspace=.20)
    fig.suptitle(f'{display} existence-score distributions: GMR, E1 and E2',fontsize=16,fontweight='bold',y=.985)
    fig.text(.5,.961,f'Standard {split.title()}: {len(target):,} queries ({target.sum()} positive, {(~target).sum()} negative); seed 2026; E1/E2 at epoch 4',ha='center',fontsize=10.5,color='#475569')
    legend = [Line2D([],[],color=color,lw=2,ls=style,label=label) for (method,color,style),label in zip(METHODS,['GMR (pretrained)','E1: end-to-end GMR + GEQ','E2: end-to-end GMR + GEQ + PV'])]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.947),ncol=3,frameon=False,fontsize=10,columnspacing=1.8,handlelength=3)
    fig.text(.5,.906,'40 equal-width bins on [0, 1]; density normalized within each class; vertical dotted line: threshold 0.4',ha='center',fontsize=9.5,color='#475569')
    for r, backbone in enumerate(BACKBONES):
        for c, (group,mask) in enumerate([('Positive queries',target),('Negative queries',~target)]):
            ax = axes[r,c]
            ax.set_title(f'{NAMES[backbone]} — {group}',fontsize=11,pad=7)
            ax.set_axisbelow(True);ax.grid(axis='y',alpha=.22,lw=.5)
            for method,color,style in METHODS:
                v = scores[backbone,method][mask];density,_=np.histogram(v,bins=bins,density=True)
                ax.stairs(density,bins,fill=True,color=color,alpha=.085,lw=0)
                ax.stairs(density,bins,color=color,lw=1.75,linestyle=style,label=method,zorder=3)
                ax.axvline(np.median(v),color=color,lw=1,ls=style,alpha=.72,zorder=2)
            ax.axvline(.4,color='#475569',lw=1.15,ls=':',zorder=4)
            ax.set_xlim(0,1);ax.set_ylim(bottom=0);ax.set_xticks(np.linspace(0,1,6));ax.set_ylabel('Density')
            ax.text(.025,.95,f'n = {mask.sum()}',transform=ax.transAxes,ha='left',va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1.5))
            handles=[Line2D([],[],color=color,lw=1.8,ls=style,label=f'{method}: median {np.median(scores[backbone,method][mask]):.2f}') for method,color,style in METHODS]
            ax.legend(handles=handles,loc='upper right',frameon=True,facecolor='white',edgecolor='none',framealpha=.9,fontsize=8.5,handlelength=2.7,borderpad=.3,labelspacing=.3)
            if r == 4:ax.set_xlabel('Existence score')
    for ext in ('pdf','svg','png'):fig.savefig(stem.with_suffix('.'+ext),dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    for suffix, records in [('_statistics.csv',stats),('_histograms.csv',histogram)]:
        with stem.with_name(stem.name+suffix).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    manifest=dict(split=split,n_total=len(target),n_positive=int(target.sum()),n_negative=int((~target).sum()),seed=2026,adapted_epoch=4,methods=[x[0] for x in METHODS],reference_layout=str(REFERENCE),threshold=.4,bins=bins.tolist(),normalization='Each backbone/method/class density integrates to 1; panel y ranges are independent.',score_field='pred_exist_score (GEQ for E1/E2); PV is not used for gating or fusion.',source_sha256=provenance,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),outputs=[str(stem.with_suffix('.'+x)) for x in ('pdf','svg','png')])
    stem.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
    caption=f'End-to-end GEQ shifts existence-score distributions, while adding training-only PV yields similar distributions in this fixed-budget experiment. Each row shows one backbone, with positive queries on the left and negative queries on the right; histograms use 40 equal-width bins and each class distribution integrates to one. GMR uses its pretrained native existence score; E1 and E2 use GEQ scores after four joint-training epochs (seed 2026). Colored vertical lines mark medians and the gray dotted line marks the fixed threshold 0.4. Panels use independent density-axis ranges. The data are the Standard {display.lower()} split ({target.sum()} positive and {(~target).sum()} negative queries), rather than the Full BN45 split used in the reference figure.'
    stem.with_name(stem.name+'_caption.md').write_text(caption+'\n')
    print(stem.with_suffix('.pdf'))

if __name__ == '__main__':main()
