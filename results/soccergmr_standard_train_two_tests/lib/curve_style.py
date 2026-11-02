from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
SPLIT_LABEL=""
BACKBONES = dict(moment_detr='Moment-DETR', eatr='EaTR', qd_detr='QD-DETR', cg_detr='CG-DETR')
METHODS = ['gmr','gmr_geq_frozen','gmr_geq_full','multi','multi_geq_frozen','multi_geq_full']
LABELS = ['GMR (baseline)', 'GMR + GEQ Frozen', 'GMR + GEQ Full',
          'GMR + Multi', 'GMR + Multi + GEQ Frozen', 'GMR + Multi + GEQ Full']
STYLES = [('#4D4D4D','-','o'),('#0072B2','--','s'),('#009E73','-.','^'),
          ('#CC79A7','-','D'),('#E69F00','--','v'),('#D55E00','-.','P')]
THRESHOLDS = [i/100 for i in range(101)]
KS = [1,3]

def panel(ax, curves, backbone, k, oracle_ceilings):
    for method,label,(color,ls,marker) in zip(METHODS,LABELS,STYLES):
        ax.plot(THRESHOLDS,curves[(backbone,method)][k],label=label,color=color,
                linestyle=ls,linewidth=1.65,marker=marker,markersize=4,
                markevery=[10,30,40,50,60,70,90],markerfacecolor='white',markeredgewidth=1,
                zorder=4 if method=='gmr' else 3)
    ceiling=oracle_ceilings[(backbone,k)]
    ax.axhline(ceiling,color='#111827',linestyle=(0,(5,3)),linewidth=1.2,zorder=2)
    ax.text(.98,ceiling+.008,f'Oracle ceiling = {ceiling:.4f}',
            transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=8.3,color='#111827',
            bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=1.2))
    for t in (.4,.5,.6): ax.axvline(t,color='#CBD5E1',lw=.65,zorder=0)
    ax.set_xlim(0,1);ax.set_ylim(0,.65)
    ax.set_xticks(np.arange(0,1.01,.2));ax.set_yticks(np.arange(0,.61,.1))
    ax.set_xlabel(r'Existence threshold $\tau$');ax.set_ylabel(f'G-mIoU@{k} (0–1)')
    ax.grid(axis='y',color='#E2E8F0',linewidth=.6)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color('#94A3B8')
    ax.tick_params(length=3,color='#94A3B8')

def map_legend(fig,ax,backbone,maps,anchor=.035,loc='lower center'):
    handles,_=ax.get_legend_handles_labels()
    labels=[f'{label}  |  mAP {maps[(backbone,method)]:.2f}%'
            for method,label in zip(METHODS,LABELS)]
    return fig.legend(handles,labels,loc=loc,bbox_to_anchor=(.5,anchor),ncol=3,
                      frameon=False,fontsize=8.3,handlelength=2.8,columnspacing=1.5)

def decorate(fig,axes,title,bottom=.22,top=.79):
    fig.suptitle(title,fontsize=13,fontweight='bold',y=.98)
    fig.text(.5,.91,f'Standard-trained checkpoints → SoccerGMR {SPLIT_LABEL}',ha='center',fontsize=9)
    fig.text(.5,.012,'Dashed horizontal line: max oracle-presence G-mIoU over six methods; fixed localization. mAP: %. No threshold selection.',
             ha='center',fontsize=8,color='#475569')
    fig.subplots_adjust(left=.08,right=.985,bottom=bottom,top=top,hspace=.48,wspace=.24)

def save(fig,stem):
    fig.savefig(HERE/(stem+'.png'),dpi=220)
    fig.savefig(HERE/(stem+'.pdf'),metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(HERE/(stem+'.svg'))
