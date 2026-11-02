#!/usr/bin/env python3
"""Offline paired five-backbone plots from a portable, verified curve snapshot."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

NAMES=dict(moment_detr='Moment-DETR',eatr='EaTR',qd_detr='QD-DETR',cg_detr='CG-DETR',sim_detr='Sim-DETR')
METHODS=('gmr','geq_full')
LABELS={'gmr':'GMR baseline','geq_full':'GMR + GEQ Full'}
STYLES={'gmr':dict(color='#535353',linestyle='-',marker='o'),
        'geq_full':dict(color='#009E73',linestyle='-.',marker='^')}


def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=220,facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(path.with_suffix('.svg'))


def panel(ax,series,k,limit):
    for method in METHODS:
        row=series[method]
        ax.plot(row['thresholds'],row['gmiou'][str(k)],linewidth=1.85,markersize=4.6,
            markevery=[10,30,40,50,60,70,90],markerfacecolor='white',markeredgewidth=1.15,
            label=LABELS[method],**STYLES[method])
    oracle=max(series[m]['oracle'][str(k)] for m in METHODS)
    ax.axhline(oracle,color='#1F2937',linewidth=1.25,linestyle=(0,(5,3)))
    ax.text(.98,oracle+.008,f'Oracle envelope = {oracle:.4f}',ha='right',va='bottom',
        transform=ax.get_yaxis_transform(),fontsize=9.2,color='#1F2937',
        bbox=dict(facecolor='white',edgecolor='none',alpha=.94,pad=1.))
    for tau in (.4,.5,.6):ax.axvline(tau,color='#CBD5E1',lw=.7,zorder=0)
    ax.set(xlim=(0,1),ylim=(0,limit),xticks=np.arange(0,1.01,.2),yticks=np.arange(0,limit+.001,.1))
    ax.set_ylabel(f'G-mIoU@{k} (0–1)')
    ax.grid(axis='y',color='#E2E8F0',lw=.65)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color('#94A3B8')
    ax.tick_params(length=3,color='#94A3B8')


def delta_text(fig,x,y,label,value):
    fig.text(x,y,f'{label}: {value:+.2f} pp',ha='center',va='top',fontsize=9.4,
             color='#13804A' if value>0 else '#C43C39' if value<0 else '#64748B')


def annotate_row(fig,ax,series,y):
    handles=[Line2D([0],[0],linewidth=1.85,markersize=5,markerfacecolor='white',**STYLES[m]) for m in METHODS]
    labels=[f'{LABELS[m]}  |  mAP {series[m]["mAP"]:.2f}%' for m in METHODS]
    fig.legend(handles,labels,ncol=2,loc='upper center',bbox_to_anchor=(.5,y),
               frameon=False,fontsize=10,columnspacing=3,handlelength=3.)
    baseline,full=series['gmr'],series['geq_full']
    delta_text(fig,.25,y-.025,'ΔmAP vs GMR',full['mAP']-baseline['mAP'])
    for x,k in [(.51,1),(.77,3)]:
        values=[np.mean([row['gmiou'][str(k)][i] for i in (40,50,60)]) for row in (baseline,full)]
        delta_text(fig,x,y-.025,f'ΔG{k} mean (.4/.5/.6)',100*(values[1]-values[0]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();snapshot=json.loads(Path(args.data).read_text());output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    limit=max(.65,np.ceil((max(s['oracle'][str(k)] for split in snapshot['splits'].values()
          for series in split['backbones'].values() for s in series.values() for k in (1,3))+.035)*10)/10)
    with PdfPages(output/'BOTH_TESTS.pdf',metadata={'CreationDate':None,'ModDate':None}) as combined:
        for split_name,split in snapshot['splits'].items():
            fig,axes=plt.subplots(5,2,figsize=(12.4,18),sharex=True,sharey=True)
            fig.subplots_adjust(left=.075,right=.985,top=.905,bottom=.15,hspace=.72,wspace=.24)
            fig.suptitle('G-mIoU sensitivity by backbone · mAP and changes versus GMR',fontsize=15,fontweight='bold',y=.982)
            fig.text(.5,.954,f'Standard-trained, fixed Val-selected checkpoints → {split["label"]}',ha='center',fontsize=11)
            fig.text(.5,.934,'Historical GEQ protocol: 6 non-improving epochs; no Test selection or tuning',ha='center',fontsize=10,color='#475569')
            for i,(name,label) in enumerate(NAMES.items()):
                series=split['backbones'][name]
                for j,k in enumerate((1,3)):
                    panel(axes[i,j],series,k,limit)
                    axes[i,j].set_title(f'{label} · G-mIoU@{k}',pad=8)
                    if i==4:axes[i,j].set_xlabel(r'Existence threshold $\tau$')
                annotate_row(fig,axes[i,0],series,axes[i,0].get_position().y0-(.05 if i==4 else .012))
            fig.text(.5,.036,'Oracle envelope: max of the two plotted methods with GT presence and each method’s fixed candidates.',ha='center',fontsize=9.2,color='#475569')
            fig.text(.5,.021,'Curves: all 101 fixed thresholds. Δ: GEQ Full − matching GMR. G1/G3 means use τ = .4/.5/.6; pp = percentage points.',ha='center',fontsize=9.2,color='#475569')
            if split_name=='full':fig.text(.5,.006,'Legacy 75-token input limit retained: two long Full videos are truncated; 16 undecodable queries remain uncovered.',ha='center',fontsize=9.2,color='#475569')
            else:fig.text(.5,.006,'One fixed checkpoint per method and backbone; lines show threshold sensitivity, not repeated-run uncertainty.',ha='center',fontsize=9.2,color='#475569')
            stem=output/('STANDARD_TEST' if split_name=='standard' else 'FULL_TEST')
            save(fig,stem);combined.savefig(fig);plt.close(fig)
            with PdfPages(output/(split_name.upper()+'_BY_BACKBONE.pdf'),metadata={'CreationDate':None,'ModDate':None}) as pages:
                for name,label in NAMES.items():
                    fig,axes=plt.subplots(1,2,figsize=(12.4,4.8),sharey=True)
                    fig.subplots_adjust(left=.075,right=.985,top=.76,bottom=.27,wspace=.24)
                    for ax,k in zip(axes,(1,3)):
                        panel(ax,split['backbones'][name],k,limit);ax.set_title(f'G-mIoU@{k}');ax.set_xlabel(r'Existence threshold $\tau$')
                    fig.suptitle(f'{label} · {split["label"]}',fontsize=14,fontweight='bold',y=.96)
                    fig.text(.5,.87,'Standard-trained; historical Val-selected best; no Test tuning',ha='center',fontsize=10)
                    annotate_row(fig,axes[0],split['backbones'][name],.145)
                    pages.savefig(fig);plt.close(fig)
    print('Created paired five-backbone figures and vector exports',flush=True)

if __name__=='__main__':main()
