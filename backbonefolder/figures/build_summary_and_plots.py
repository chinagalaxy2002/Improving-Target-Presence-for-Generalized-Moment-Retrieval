#!/usr/bin/env python3
"""Rebuild the one-table summary and frozen-prediction threshold plots."""
import csv, gzip, hashlib, json, math, pathlib, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

FIGURES=pathlib.Path(__file__).resolve().parent
BASE=FIGURES.parent
PROJECT=BASE.parent
SHARED=PROJECT/'Phase2(addbackbone_with_S4_minusFusion)'/'shared'
sys.path.insert(0,str(SHARED))
from official_eval.eval_main import evaluate_gmr
from official_eval.metrics import get_existence_score, _compute_set_iou_score, prepare_submission_for_gmiou, compute_G_mIoU, compute_gmr_cls

LABELS=pathlib.Path('/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/data/label/Standard/test.jsonl')
NAMES={'moment_detr':'Moment-DETR','qd_detr':'QD-DETR','cg_detr':'CG-DETR','eatr':'EaTR','flashvtg':'FlashVTG'}
ORDER=list(NAMES)
GRID=np.round(np.linspace(0,1,101),2)
METRICS=['AUROC','Rej-F1@0.4','mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3']
COLORS={'original':'#0072B2','gmr':'#D55E00'}
MARKERS={'original':'o','gmr':'s'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.titlesize':11,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.linewidth':.8,'savefig.facecolor':'white','figure.facecolor':'white'})

def read_rows(path):
 opener=gzip.open if path.suffix=='.gz' else open
 with opener(path,'rt') as stream:return [json.loads(line) for line in stream if line.strip()]
def write_json(path,data):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
def source_for(model,variant):
 directory=BASE/model/variant
 names=['test_predictions.jsonl','final_test_predictions.jsonl']
 return next(directory/name for name in names if (directory/name).exists())
def curves_for(scores,local,positive,thresholds):
 rows=[]
 for threshold in thresholds:
  accept=scores>float(threshold)
  tp=int((accept&positive).sum());tn=int((~accept&~positive).sum());fp=int((accept&~positive).sum());fn=int((~accept&positive).sum())
  contribution=np.where(accept[:,None],local,(~positive)[:,None])
  rows.append({'threshold':float(threshold),'G-mIoU@1':round(100*sum(contribution[:,0].tolist())/len(positive),2),'G-mIoU@3':round(100*sum(contribution[:,1].tolist())/len(positive),2),'N_acc':100*tn/int((~positive).sum()),'positive_recall':100*tp/int(positive.sum()),'Rej-F1':round(100*2*tn/(2*tn+fp+fn),2) if 2*tn+fp+fn else 0.,'TP':tp,'TN':tn,'FP':fp,'FN':fn})
 return rows

def measure(model,variant,gt):
 path=source_for(model,variant);pred=read_rows(path)
 gt_by_id={str(g['qid']):g for g in gt};by_id={str(p['qid']):p for p in pred}
 assert len(pred)==len(by_id)==len(gt)==1036 and set(by_id)==set(gt_by_id)
 # Align identifiers and ordering with the same official Standard test labels.
 aligned=[]
 for g in gt:
  p=dict(by_id[str(g['qid'])]);p['qid']=g['qid'];aligned.append(p)
 pred=aligned
 result=evaluate_gmr(pred,gt,cls_thresholds=(.4,),gmiou_cls_threshold=.4,map_num_workers=0,verbose=False)
 scores=np.array([get_existence_score(p)[0] for p in pred]);assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
 positive=np.array([bool(g['relevant_windows']) for g in gt])
 # Use the official cleaner and exact official per-query set-IoU function.
 clean,_=prepare_submission_for_gmiou(pred,cls_threshold=-1,max_pred_windows=10)
 local=np.array([[_compute_set_iou_score([w[:2] for w in p['pred_relevant_windows'][:k]],g['relevant_windows']) for k in (1,3)] for p,g in zip(clean,gt)])
 grid=curves_for(scores,local,positive,GRID)
 # Check the cached factorization against official gating, G-mIoU and CLS at endpoints and requested operating points.
 for threshold in (0.,.4,.6,.8,1.):
  row=next(x for x in grid if x['threshold']==threshold)
  gated,_=prepare_submission_for_gmiou(pred,cls_threshold=threshold,max_pred_windows=10)
  official_g=compute_G_mIoU(gated,gt,k_list=(1,3));official_c=compute_gmr_cls(pred,gt,thresholds=(threshold,))['per_threshold'][str(threshold)]
  assert all(row[k]==official_g[k] for k in ['G-mIoU@1','G-mIoU@3'])
  assert row['Rej-F1']==official_c['Rej-F1'] and all(row[k]==official_c[k] for k in ['TP','TN','FP','FN'])
 point=grid[40];assert all(point[k]==result['brief'][k] for k in ['G-mIoU@1','G-mIoU@3']) and point['Rej-F1']==result['brief']['Rej-F1@0.4']
 empirical=curves_for(scores,local,positive,np.unique(np.r_[0.,scores,1.]))
 data={'backbone':model,'variant':variant,'name':NAMES[model]+('-GMR' if variant=='gmr' else ''),'source_predictions':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'score_sources':result['GMR-CLS']['score_sources'],'tau_0.4':{k:result['brief'][k] for k in METRICS},'grid':grid,'empirical_tradeoff':empirical,'checks':{'query_coverage':1036,'positive':int(positive.sum()),'negative':int((~positive).sum()),'official_operating_points_verified':[0,.4,.6,.8,1]},'official_metrics_path':str(BASE/model/variant/'summary_table_metrics_tau_0.4.json')}
 write_json(BASE/model/variant/'summary_table_metrics_tau_0.4.json',result)
 write_json(FIGURES/model/'data'/f'{model}_{variant}.json',data)
 with (FIGURES/model/'data'/f'{model}_{variant}_threshold_grid.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(grid[0]));w.writeheader();w.writerows(grid)
 print('MEASURED',data['name'],json.dumps(data['tau_0.4']),flush=True)
 return data

def axes_style(ax,xlabel,ylabel,tradeoff=False):
 ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_ylim(0,100 if tradeoff or ylabel.startswith('Rej') else 80)
 ax.set_xlim(0,100 if tradeoff else 1)
 ax.xaxis.set_major_locator(MultipleLocator(25 if tradeoff else .25))
 ax.yaxis.set_major_locator(MultipleLocator(25 if tradeoff or ylabel.startswith('Rej') else 20))
 ax.grid(True,alpha=.25,color='#94A3B8',linewidth=.6);ax.set_axisbelow(True)

def panel(ax,kind,series):
 if kind=='gmiou':
  axes_style(ax,'Existence threshold '+r'$\tau$','G-mIoU (%)')
  for variant,d in series.items():
   rows=d['grid'];x=[r['threshold'] for r in rows]
   for k,style in [(1,'-'),(3,'--')]:
    ax.plot(x,[r[f'G-mIoU@{k}'] for r in rows],color=COLORS[variant],linestyle=style,marker=MARKERS[variant],markevery=20,markersize=3,linewidth=1.8,label=('Original' if variant=='original' else 'GMR')+f', k={k}')
    ax.scatter([.4],[rows[40][f'G-mIoU@{k}']],color=COLORS[variant],marker=MARKERS[variant],s=32,zorder=5)
  ax.axvline(.4,color='#64748B',linestyle=':',linewidth=1.1)
  ax.legend(loc='upper left',frameon=False,ncol=2,handlelength=1.6,columnspacing=.7,fontsize=10)
 elif kind=='tradeoff':
  axes_style(ax,'Positive recall (%)','N-acc (%)',tradeoff=True)
  for variant,d in series.items():
   rows=d['empirical_tradeoff'];operating=d['grid'][40]
   ax.plot([r['positive_recall'] for r in rows],[r['N_acc'] for r in rows],color=COLORS[variant],linestyle='--' if variant=='original' else '-',linewidth=1.8,label='Original' if variant=='original' else 'GMR')
   ax.scatter([operating['positive_recall']],[operating['N_acc']],color=COLORS[variant],marker=MARKERS[variant],s=38,zorder=5)
  ax.legend(loc='lower left',frameon=False)
 elif kind=='rej_f1':
  axes_style(ax,'Existence threshold '+r'$\tau$','Rej-F1 (%)')
  for variant,d in series.items():
   rows=d['grid'];ax.plot([r['threshold'] for r in rows],[r['Rej-F1'] for r in rows],color=COLORS[variant],linestyle='--' if variant=='original' else '-',marker=MARKERS[variant],markevery=20,markersize=3,linewidth=1.8,label='Original' if variant=='original' else 'GMR')
   ax.scatter([.4],[rows[40]['Rej-F1']],color=COLORS[variant],marker=MARKERS[variant],s=32,zorder=5)
  ax.axvline(.4,color='#64748B',linestyle=':',linewidth=1.1);ax.legend(loc='upper left',frameon=False)

def export(fig,stem):
 model=next(name for name in ORDER if stem.startswith(name+'_'))
 for suffix in ['png','pdf','svg']:fig.savefig(FIGURES/model/f'{stem}.{suffix}',dpi=220,bbox_inches='tight',pad_inches=.08)
 plt.close(fig)

def plot_model(model,series):
 modelname=NAMES[model];fig,axs=plt.subplots(1,3,figsize=(10.2,3.8));fig.subplots_adjust(left=.065,right=.985,bottom=.22,top=.79,wspace=.40)
 for ax,kind,title in zip(axs,['gmiou','tradeoff','rej_f1'],['(a) Generalized retrieval','(b) Rejection / retention','(c) Null-query rejection']):
  panel(ax,kind,series);ax.set_title(title,pad=8)
 fig.suptitle(modelname+' | Soccer-GMR Standard test',fontsize=13,y=.99)
 note='Large markers indicate '+r'$\tau=0.4$'+'. N-acc = negative-query rejection rate.'
 if model=='flashvtg':note+=' FlashVTG-GMR (frozen E0) only.'
 fig.text(.5,.015,note,ha='center',va='bottom',fontsize=10.5)
 export(fig,model+'_analysis')
 for kind in ['gmiou','tradeoff','rej_f1']:
  fig,ax=plt.subplots(figsize=(4.2,3.6));panel(ax,kind,series);ax.set_title(modelname+(' / GMR only' if model=='flashvtg' else ''))
  fig.tight_layout();export(fig,model+'_'+kind)
 print('PLOTTED',model,flush=True)

def main():
 # Keep the canonical table complete after adding the fixed-epoch S4 archive.
 if (FIGURES/'build_complete_comparison.py').exists() and (PROJECT/'Phase2(addbackbone_with_S4_minusFusion)').is_dir():
  from build_complete_comparison import main as build_complete
  return build_complete()
 FIGURES.mkdir(exist_ok=True)
 for model in ORDER:(FIGURES/model/'data').mkdir(parents=True,exist_ok=True)
 gt=read_rows(LABELS);assert hashlib.sha256(LABELS.read_bytes()).hexdigest()=='d8c47dee986b3ea87ba3e008ac2f5f34b0ab000337631e324a6ecffd1dea910c'
 series={}
 for model in ORDER:
  variants=['gmr'] if model=='flashvtg' else ['original','gmr']
  series[model]={variant:measure(model,variant,gt) for variant in variants}
 # This Markdown file intentionally contains exactly one table and no other content.
 headers=['Model','AUROC (%)','Rej-F1 (τ=0.4, %)','mAP (%)','mR@1 (%)','mR@5 (%)','mR+@5 (%)','G-mIoU@1 (τ=0.4, %)','G-mIoU@3 (τ=0.4, %)']
 lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']+['---:']*8)+' |']
 table=[]
 # Match Table2 grouping: all original backbones first, then their GMR variants.
 for variant in ['original','gmr']:
  for model in ORDER:
   if variant not in series[model]:continue
   d=series[model][variant];name=d['name']+(' (E0)' if model=='flashvtg' else '')
   lines.append('| '+name+' | '+' | '.join(f'{d["tau_0.4"][k]:.2f}' for k in METRICS)+' |')
   table.append({'model':name,**d['tau_0.4']})
 tablepath=BASE/'TABLE2_OUR_RESULTS.md';tablepath.write_text('\n'.join(lines)+'\n')
 assert len(lines)==11 and all(line.startswith('|') and line.endswith('|') for line in lines)
 for model,data in series.items():plot_model(model,data)
 manifest={'table':str(tablepath),'table_rows':9,'threshold_for_summary':.4,'test_sha256':hashlib.sha256(LABELS.read_bytes()).hexdigest(),'test_queries':len(gt),'grid':{'min':0,'max':1,'step':.01,'points':101},'gate_rule':'positive iff score > threshold; explicit GMR score, else native max proposal score','N_acc':'TN / (TN+FP), percentage','positive_recall':'TP / (TP+FN), percentage','flashvtg':'Only frozen E0 FlashVTG-GMR included; original FlashVTG omitted per user instruction.','no_checkpoint_or_threshold_selection':True,'models':[{k:v for k,v in d.items() if k not in ['grid','empirical_tradeoff']} for data in series.values() for d in data.values()],'figure_types':['five combined three-panel plots','five threshold / G-mIoU@1,@3 plots','five Positive recall / N-acc plots','five threshold / Rej-F1 plots'],'formats':['png','pdf','svg']}
 write_json(FIGURES/'SOURCE_MANIFEST.json',manifest)
 print('DONE',str(tablepath),flush=True)

if __name__=='__main__':main()
