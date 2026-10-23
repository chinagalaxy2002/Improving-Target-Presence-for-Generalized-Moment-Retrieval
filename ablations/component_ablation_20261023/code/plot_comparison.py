"""Matched visual language of reference branch; only E0 and S4-Fusion."""
import csv,json,gzip
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from plot_utils import ROOT,plt,ORDER,COLORS,LINES,MARKERS,LABELS,axis,legend,save
D=json.loads((ROOT/'reports/figure_data.json').read_text());R=D['reports']
with (ROOT/'reports/threshold_grid_test.csv').open() as f:records=list(csv.DictReader(f))
G={s:[{k:float(v) for k,v in r.items() if k!='model'} for r in records if r['model']==s] for s in ORDER}
assert all(len(G[s])==101 for s in ORDER)
SHORT={'E0':'baseline','S4Fusion':'S4-Fusion'}
def curve(ax,s,x,y):ax.plot(x,y,color=COLORS[s],ls=LINES[s],marker=MARKERS[s],markevery=10,ms=3.7,lw=1.8,label=LABELS[s])
def bars(ax,keys,labels,title,ylabel,ylim):
 x=np.arange(len(keys));w=.32
 for i,s in enumerate(ORDER):
  b=ax.bar(x+(i-.5)*w,[R[s]['fixed_05'][k] for k in keys],w,color=COLORS[s],edgecolor='white',linewidth=.7)
  ax.bar_label(b,fmt='%.2f',fontsize=8,padding=3)
 ax.set_xticks(x,labels);axis(ax,title,'',ylabel,ylim)
def sweep():
 fig,axs=plt.subplots(2,3,figsize=(14.3,8.4),layout='constrained')
 for s in ORDER:
  t=[r['threshold'] for r in G[s]]
  for ax,k in [(axs[0,0],'G-mIoU@1'),(axs[0,1],'G-mIoU@3'),(axs[1,1],'Rej_F1')]:curve(ax,s,t,[r[k] for r in G[s]])
  curve(axs[0,2],s,t,[r['G-mIoU@1']-b['G-mIoU@1'] for r,b in zip(G[s],G['E0'])])
  curve(axs[1,0],s,[r['positive_recall'] for r in G[s]],[r['N_acc'] for r in G[s]])
  r=R[s]['fixed_05'];axs[1,0].scatter(r['positive_recall'],r['N_acc'],color=COLORS[s],marker=MARKERS[s],s=65,edgecolors='white',zorder=5)
 axis(axs[0,0],'(a) Threshold sensitivity','Presence threshold','G-mIoU@1 (%)',(0,65))
 axis(axs[0,1],'(b) Top-3 retrieval','Presence threshold','G-mIoU@3 (%)',(0,65))
 axis(axs[0,2],'(c) Relative to baseline','Presence threshold','Delta G-mIoU@1 (pp)');axs[0,2].axhline(0,color='#64748b',lw=.8)
 axis(axs[1,0],'(d) Rejection / positive retention','Positive recall (%)','N-acc (%)',(0,105));axs[1,0].set_xlim(0,105)
 axis(axs[1,1],'(e) Negative-class F1','Presence threshold','Rejection F1 (%)',(0,100))
 x=np.arange(3);w=.32
 for i,s in enumerate(ORDER):
  v=[next(r['G-mIoU@1'] for r in G[s] if r['threshold']==t) for t in [.4,.5,.6]]
  b=axs[1,2].bar(x+(i-.5)*w,v,w,color=COLORS[s],edgecolor='white');axs[1,2].bar_label(b,fmt='%.2f',fontsize=8,padding=3)
 axs[1,2].set_xticks(x,['0.4','0.5','0.6']);axis(axs[1,2],'(f) Fixed operating points','Presence threshold','G-mIoU@1 (%)',(0,70))
 legend(fig,axs[0,0]);save(fig,'fig7_threshold_sweep')
def profiles():
 fig,axs=plt.subplots(2,2,figsize=(14.5,8.5),layout='constrained',gridspec_kw={'height_ratios':[.7,1.]})
 cfg=np.array([[0,0,0,0,0,0],[1,1,1,0,0,0]])
 ax=axs[0,0];ax.pcolormesh(np.arange(7)-.5,np.arange(3)-.5,cfg,cmap='Blues',vmin=0,vmax=1);ax.set_ylim(1.5,-.5);ax.set_yticks([0,1],['baseline','S4-Fusion']);ax.set_xticks(range(6),['Global query','LoRA r8','Refiner','Candidate fusion','Evidence gate','Safeguard'],rotation=25,ha='right')
 for i in range(2):
  for j in range(6):ax.text(j,i,'+' if cfg[i,j] else '−',ha='center',va='center',color='white' if cfg[i,j] else '#64748b')
 ax.set_title('(a) Active architecture components',loc='left',pad=10)
 keys=['G-mIoU@1','G-mIoU@3','mAP','raw_mAP','N_acc','positive_recall','Multi_G3','AUROC'];a=np.array([[R[s]['fixed_05'][k] for k in keys] for s in ORDER]);ax=axs[0,1]
 ax.pcolormesh(np.arange(9)-.5,np.arange(3)-.5,a,cmap='Blues',vmin=0,vmax=100);ax.set_ylim(1.5,-.5);ax.set_yticks([0,1],['baseline','S4-Fusion']);ax.set_xticks(range(8),['G@1','G@3','mAP','raw mAP','N-acc','PosRec','Multi G@3','AUROC'],rotation=30,ha='right')
 for i in range(2):
  for j in range(8):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if a[i,j]>65 else '#0f172a')
 ax.set_title('(b) Official Test metrics at threshold 0.5 (%)',loc='left',pad=10)
 bars(axs[1,0],['G-mIoU@1','G-mIoU@3','G-mIoU@5'],['G@1','G@3','G@5'],'(c) Retrieval at threshold 0.5','G-mIoU (%)',(0,70))
 bars(axs[1,1],['positive_recall','N_acc','Multi_positive_recall'],['Positive recall','N-acc','Multi recall'],'(d) Positive retention and negative rejection','Rate (%)',(0,114))
 fig.legend(handles=[Line2D([0],[0],color=COLORS[s],marker=MARKERS[s],ls=LINES[s],label=LABELS[s]) for s in ORDER],loc='outside upper center',ncol=2,frameon=False);save(fig,'fig8_profiles')
def effects():
 fig,axs=plt.subplots(1,3,figsize=(16.5,5.4),layout='constrained',gridspec_kw={'width_ratios':[1,1.05,1.25]})
 keys=['G-mIoU@1','G-mIoU@3','G-mIoU@5','mAP','raw_mAP','N_acc','positive_recall','AUROC'];vals=[R['S4Fusion']['fixed_05'][k]-R['E0']['fixed_05'][k] for k in keys]
 ax=axs[0];b=ax.barh(range(8),vals,color=['#0072B2' if v>=0 else '#D55E00' for v in vals]);ax.bar_label(b,fmt='%+.2f',padding=3,fontsize=8);ax.set_yticks(range(8),['G@1','G@3','G@5','mAP','raw mAP','N-acc','PosRec','AUROC']);ax.invert_yaxis();ax.axvline(0,color='#64748b',lw=.8);ax.margins(x=.23);axis(ax,'(a) S4-Fusion minus baseline at 0.5','Change (pp)','')
 ax=axs[1]
 for k,color,style,mark in [('G-mIoU@1','#0072B2','-','o'),('G-mIoU@3','#009E73','--','s'),('G-mIoU@5','#CC79A7',':','^')]:ax.plot([r['threshold'] for r in G['E0']],[a[k]-b[k] for a,b in zip(G['S4Fusion'],G['E0'])],color=color,ls=style,marker=mark,markevery=10,ms=3,lw=1.7,label=k.replace('G-mIoU','G'))
 ax.axhline(0,color='#64748b',lw=.8);axis(ax,'(b) Paired retrieval differences','Presence threshold','Delta G-mIoU (pp)');ax.legend(frameon=False,fontsize=8)
 # Actual paired decisions, not counterfactual model components or invented interactions.
 from environment import read_rows,write_json
 gt=read_rows(ROOT/'data/test.jsonl');positive=np.array([bool(g['relevant_windows']) for g in gt]);accept={}
 for s in ORDER:
  with gzip.open(ROOT/'results'/s/'test_predictions.jsonl.gz','rt') as f:p={r['qid']:r for r in map(json.loads,f)}
  accept[s]=torch.sigmoid(torch.tensor([p[g['qid']]['pred_exist_logit_raw'] for g in gt],dtype=torch.float32)).numpy()>=.5
 a,b=accept['E0'],accept['S4Fusion'];n=[int(np.sum(positive&a&~b)),int(np.sum(positive&~a&b)),int(np.sum(~positive&a&~b)),int(np.sum(~positive&~a&b))]
 assert n[1]-n[0]==R['S4Fusion']['fixed_05']['TP']-R['E0']['fixed_05']['TP'];assert n[2]-n[3]==R['S4Fusion']['fixed_05']['TN']-R['E0']['fixed_05']['TN']
 labels=['Positive: kept → rejected','Positive: rejected → kept','Negative: accepted → rejected','Negative: rejected → accepted'];write_json(ROOT/'reports/paired_decisions_tau05.json',dict(zip(labels,n)))
 ax=axs[2];bars_=ax.barh(range(4),n,color=['#D55E00','#0072B2','#0072B2','#D55E00']);ax.bar_label(bars_,padding=3,fontsize=8);ax.set_yticks(range(4),labels,fontsize=8);ax.invert_yaxis();ax.margins(x=.16);axis(ax,'(c) Changed decisions at threshold 0.5','Queries','')
 fig.suptitle('Two frozen Val-selected models, seed 2026 · descriptive paired comparison',fontsize=11);save(fig,'fig9_paired_effects')
if __name__=='__main__':sweep();profiles();effects();print('Two-model figures 7–9 rendered: PNG / PDF / SVG',flush=True)
