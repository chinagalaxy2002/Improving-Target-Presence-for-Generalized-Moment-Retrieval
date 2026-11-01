"""Reference-style, class-conditional score histograms from saved formal Test exports."""
import argparse,csv,json,os,fcntl
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import OUT,BACKBONES,rows,sha,dump

NAMES={'moment_detr':'Moment-DETR','eatr':'EaTR','qd_detr':'QD-DETR','flashvtg':'FlashVTG'}
COLORS=['#64748b','#0072b2','#d55e00'];STYLES=['--','-','-.']
REFERENCE=dict(url='https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/main/ablations/geq_gap_all_backbones_20261026/figures/test_existence_score_distributions.pdf',git_blob_sha='c47827e422ef039b888a9c5d6a48bb3afffe021a',local_sha256='0e6419c3b0f91049464e73659e299a92b233a8480e5ac8061ee31728cf0ea506')


def distribution(values,edges):
 values=np.asarray(values,dtype=float);assert len(values)>0 and np.isfinite(values).all() and (values>=0).all() and (values<=1).all()
 counts,_=np.histogram(values,bins=edges);density=counts/len(values)/np.diff(edges)
 assert int(counts.sum())==len(values) and abs(float(np.dot(density,np.diff(edges)))-1)<1e-12
 return dict(n=len(values),counts=counts.tolist(),density=density.tolist(),mean=float(values.mean()),median=float(np.median(values)),minimum=float(values.min()),maximum=float(values.max()),fraction_above={str(t):float((values>t).mean()) for t in [.4,.5,.6]})


def build_snapshot(split,family):
 gtpath=OUT/'data'/f'{split}.jsonl';gt=rows(gtpath);positive=np.array([bool(g['relevant_windows']) for g in gt]);ids=[g['evaluation_id'] for g in gt];assert len(set(ids))==len(ids)
 methods=[family,family+'_geq_frozen',family+'_geq_full'];edges=np.linspace(0,1,41);panels={};sources={}
 for b in BACKBONES:
  panel=dict(methods={},pending=[])
  for method in methods:
   d=OUT/'runs'/b/method/split;marker=d/'COMPLETE.json'
   if not marker.exists():panel['pending'].append(method);continue
   done=json.loads(marker.read_text());assert done['rows']==len(gt) and done['prediction_sha256']==sha(d/'predictions.jsonl')
   pred=rows(d/'predictions.jsonl');assert [p['qid'] for p in pred]==ids and all(p['vid']==g['vid'] and p['original_qid']==g['original_qid'] for p,g in zip(pred,gt))
   scores=np.array([p['pred_exist_score'] for p in pred]);panel['methods'][method]={'positive':distribution(scores[positive],edges),'negative':distribution(scores[~positive],edges)}
   sources[b+'/'+method]=dict(prediction_sha256=done['prediction_sha256'],checkpoint_sha256=done['checkpoint_sha256'],complete_sha256=sha(marker))
  panels[b]=panel
 return dict(type='existence_score_distributions',split=split,family=family,methods=methods,bin_edges=edges.tolist(),panels=panels,
  coverage=dict(rows=len(gt),positive=int(positive.sum()),negative=int((~positive).sum())),ground_truth_sha256=sha(gtpath),sources=sources,reference=REFERENCE,
  definition='Class-conditional density over the complete 0–1 score range; 40 equal-width bins. Each method/class integrates to1. Means use all rows in that class. No sample selection, smoothing or score clipping.',
  oracle_interpretation='Black endpoint reference assigns binary-label Oracle scores1 to positives and0 to negatives. It is not a finite density upper bound.',test_used_for_selection=False)


def render(snapshot,destination):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'charades-score-distributions','axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(4,2,figsize=(12.2,11.7),sharex=True);edges=np.array(snapshot['bin_edges']);family=snapshot['family'];stats=[];bins=[]
 labels=[('GMR' if family=='gmr' else 'Multi'),'+ GEQ Frozen','+ GEQ Full']
 for row,b in enumerate(BACKBONES):
  panel=snapshot['panels'][b]
  for col,kind in enumerate(['positive','negative']):
   axis=axes[row,col];class_name='Positive' if kind=='positive' else 'Negative'
   for index,method in enumerate(snapshot['methods']):
    if method not in panel['methods']:continue
    v=panel['methods'][method][kind];color=COLORS[index];style=STYLES[index]
    mean_label=f"{v['mean']:.2e}" if 0<v['mean']<.001 else f"{v['mean']:.3f}"
    axis.stairs(v['density'],edges,color=color,ls=style,lw=1.7,label=f"{labels[index]}: mean {mean_label}")
    axis.axvline(v['median'],color=color,ls=style,alpha=.5,lw=.8)
    stats.append(dict(backbone=b,method=method,class_name=kind,**{k:v[k] for k in ['n','mean','median','minimum','maximum']},**{'fraction_above_'+t:v['fraction_above'][t] for t in ['0.4','0.5','0.6']}))
    for i in range(40):bins.append(dict(backbone=b,method=method,class_name=kind,bin_left=edges[i],bin_right=edges[i+1],count=v['counts'][i],density=v['density'][i],class_n=v['n']))
   for t in [.4,.5,.6]:axis.axvline(t,color='#64748b',ls=':',lw=.75,alpha=.65)
   ideal=1 if kind=='positive' else 0;axis.axvline(ideal,color='#111827',ls=(0,(5,2)),lw=1.4,clip_on=False)
   axis.text(.98 if ideal else .02,.96,f'Oracle score = {ideal}',ha='right' if ideal else 'left',va='top',transform=axis.transAxes,fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
   title=NAMES[b]+(' (Multi-TQAux)' if b=='flashvtg' and family=='multi' else '')
   axis.set(xlim=(0,1),ylim=(0,None),ylabel='Density',title=f"{title} — {class_name} (n={snapshot['coverage'][kind]})",xticks=np.linspace(0,1,6));axis.grid(axis='y',alpha=.12);axis.tick_params(labelbottom=True)
   if panel['methods']:axis.legend(fontsize=8,frameon=False,loc='upper left' if ideal else 'upper right')
   else:axis.text(.5,.5,'Formal Test exports pending\nNo distribution fabricated',ha='center',va='center',transform=axis.transAxes,color='#64748b',fontsize=10)
   if panel['pending']:axis.text(.98,.02,'Pending: '+', '.join(panel['pending']),ha='right',va='bottom',transform=axis.transAxes,fontsize=6.5,color='#64748b')
   if row==3:axis.set_xlabel('Existence score')
 fig.suptitle(f"Charades {snapshot['split'].replace('_','-')} existence-score distributions | {labels[0]} / GEQ Frozen / GEQ Full",fontsize=13)
 fig.text(.5,.01,'Each class/method density integrates to 1 | colored verticals: medians | gray dotted: thresholds 0.4/0.5/0.6\nBlack endpoint: binary-label Oracle (positive 1, negative 0) | all saved Test rows retained | seed 2023, Val-selected checkpoints',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.045,1,.97));destination.parent.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf','svg']:
  target=destination.with_suffix('.'+ext);tmp=target.with_name(target.stem+f'.{os.getpid()}.tmp.'+ext);metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else None;fig.savefig(tmp,dpi=180,bbox_inches='tight',metadata=metadata);tmp.replace(target)
 plt.close(fig)
 for suffix,values in [('.csv',bins),('.stats.csv',stats)]:
  if not values:continue
  target=destination.with_suffix(suffix);tmp=target.with_name(target.name+f'.{os.getpid()}.tmp')
  with tmp.open('w') as stream:
   writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
  tmp.replace(target)


def generate_score_distributions():
 directory=OUT/'figures/score_distributions';directory.mkdir(parents=True,exist_ok=True);readme=['# Charades existence-score distributions','', 'Reference layout: one backbone per row, positives left and negatives right. Full score range0–1, 40 bins, independent class-conditional densities. Curves and mean scores use only immutable formal Test prediction exports; pending branches are labelled. No training or Test rerun.',''];audit=[]
 for split in ['test_id','test_ood']:
  for family in ['gmr','multi']:
   snapshot=build_snapshot(split,family);stem=f'{split}_{family}_existence_score_distributions';dest=directory/stem;dump(dest.with_suffix('.snapshot.json'),snapshot);render(snapshot,dest)
   readme += [f'## {split}: {family}','',f'![Score distributions]({stem}.png)','',f'[PDF]({stem}.pdf) · [SVG]({stem}.svg) · [Histogram CSV]({stem}.csv) · [Class statistics]({stem}.stats.csv)','']
   audit.append(dict(split=split,family=family,available=list(snapshot['sources']),pending={b:p['pending'] for b,p in snapshot['panels'].items() if p['pending']},snapshot_sha256=sha(dest.with_suffix('.snapshot.json'))))
 (directory/'README.md').write_text('\n'.join(readme)+'\n');dump(directory/'AUDIT.json',dict(passed=True,reference=REFERENCE,figures=audit,checks=['Complete unique row identities and unchanged prediction SHA','Scores finite and in0–1 without clipping','Each class/method histogram integrates to1','40 common bins and full0–1 x-axis','Means/medians computed from all class rows','Only COMPLETE formal tests plotted; missing methods explicitly pending']))


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--snapshot');parser.add_argument('--output');args=parser.parse_args()
 if args.snapshot:render(json.loads(Path(args.snapshot).read_text()),Path(args.output))
 else:
  (OUT/'logs').mkdir(parents=True,exist_ok=True)
  with (OUT/'logs/plot_report.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX);generate_score_distributions()


if __name__=='__main__':main()
