#!/usr/bin/env python3
"""Portable 11-backbone paired Test figures, preserving both protocol groups."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from plot_test_thresholds import panel,STYLES,LABELS


def footer(ax,series):
 ax.axis('off')
 handles=[Line2D([0],[0],lw=1.85,markersize=4.6,markerfacecolor='white',**STYLES[m]) for m in ['gmr','geq_full']]
 ax.legend(handles,[f'{LABELS[m]} | mAP {series[m]["mAP"]:.2f}%' for m in ['gmr','geq_full']],loc='upper center',ncol=2,frameon=False,fontsize=10,columnspacing=3,handlelength=3,borderaxespad=0)
 b,f=series['gmr'],series['geq_full']
 for x,label,value in [(0.18,'ΔmAP',f['mAP']-b['mAP']),(0.51,'ΔG1 mean',100*np.mean([f['gmiou']['1'][i]-b['gmiou']['1'][i] for i in [40,50,60]])),(0.83,'ΔG3 mean',100*np.mean([f['gmiou']['3'][i]-b['gmiou']['3'][i] for i in [40,50,60]]))]:
  ax.text(x,.06,f'{label}: {value:+.2f} pp',ha='center',va='bottom',fontsize=10,color='#13804A' if value>0 else '#C43C39' if value<0 else '#64748B')

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='results/test_thresholds_11_backbones/PLOT_DATA.json');p.add_argument('--output',default='results/test_thresholds_11_backbones');a=p.parse_args();snapshot=json.loads(Path(a.data).read_text());dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
 names=snapshot['backbone_labels'];assert len(names)==11
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
 limit=max(.65,np.ceil((max(s['oracle'][str(k)] for split in snapshot['splits'].values() for pair in split['backbones'].values() for s in pair.values() for k in [1,3])+.035)*10)/10)
 metadata={'CreationDate':None,'ModDate':None}
 def save(fig,stem):
  for ext in ['png','pdf','svg']:
   target=dest/(stem+'.'+ext)
   fig.savefig(target,dpi=180,facecolor='white',**({'metadata':metadata} if ext=='pdf' else {}))
   if ext=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
 with PdfPages(dest/'BOTH_TESTS.pdf',metadata=metadata) as combined:
  for split_name,split in snapshot['splits'].items():
   fig=plt.figure(figsize=(12.4,36.8));grid=fig.add_gridspec(11,1,left=.075,right=.985,top=.947,bottom=.035,hspace=.38)
   fig.suptitle('11 backbones · GMR baseline vs GMR + GEQ Full',fontsize=17,fontweight='bold',y=.992)
   fig.text(.5,.980,f'Standard-trained, fixed Val-selected checkpoints → {split["label"]}',ha='center',fontsize=12)
   fig.text(.5,.969,'Historical 5: GEQ patience 6, video limit 75 | Expansion 6: GEQ patience 10, video limit 128, seed 2023',ha='center',fontsize=10,color='#475569')
   for i,(name,label) in enumerate(names.items()):
    inner=grid[i].subgridspec(2,2,height_ratios=[1,.32],hspace=.50,wspace=.24);series=split['backbones'][name]
    group='historical · 75 tokens · GEQ patience 6' if i<5 else 'expansion · 128 tokens · GEQ patience 10'
    for j,k in enumerate([1,3]):
     ax=fig.add_subplot(inner[0,j]);panel(ax,series,k,limit);ax.set_title(f'{label} · G-mIoU@{k}\n{group}',pad=6,fontsize=10.5)
     if i==10:ax.set_xlabel(r'Existence threshold $\tau$')
    footer(fig.add_subplot(inner[1,:]),series)
   fig.text(.5,.019,'Curves: 101 fixed thresholds, 0–1 units. Δ = GEQ Full − matching GMR; G1/G3 means use τ = .4/.5/.6; pp = percentage points.',ha='center',fontsize=10,color='#475569')
   fig.text(.5,.012,'Oracle envelope: GT presence with each method’s fixed candidates; diagnostic only. Test never selects checkpoints or thresholds.',ha='center',fontsize=10,color='#475569')
   fig.text(.5,.005,'Full-Decodable: 2970/2986 queries. Historical 75-token truncation is retained; expansion uses full valid videos (up to 128 tokens).' if split_name=='full' else 'One seed per experiment; curves show threshold sensitivity, not run uncertainty. Protocol groups are not a controlled cross-backbone ranking.',ha='center',fontsize=10,color='#475569')
   save(fig,split_name.upper()+'_TEST');combined.savefig(fig);plt.close(fig)
   with PdfPages(dest/(split_name.upper()+'_BY_BACKBONE.pdf'),metadata=metadata) as pages:
    for i,(name,label) in enumerate(names.items()):
     fig=plt.figure(figsize=(12.4,4.8));grid=fig.add_gridspec(2,2,left=.075,right=.985,top=.79,bottom=.08,height_ratios=[1,.28],hspace=.22,wspace=.24)
     for j,k in enumerate([1,3]):
      ax=fig.add_subplot(grid[0,j]);panel(ax,split['backbones'][name],k,limit);ax.set_title(f'G-mIoU@{k}');ax.set_xlabel(r'Existence threshold $\tau$')
     footer(fig.add_subplot(grid[1,:]),split['backbones'][name]);fig.suptitle(f'{label} · {split["label"]}',fontsize=14,fontweight='bold',y=.97)
     fig.text(.5,.88,'Historical: GEQ patience6; 75 video tokens' if i<5 else 'Expansion: GEQ patience10; 128 video tokens',ha='center',fontsize=10)
     pages.savefig(fig);plt.close(fig)
 print('11-backbone Standard and Full plots, paired PDF, and individual pages generated')
if __name__=='__main__':main()
