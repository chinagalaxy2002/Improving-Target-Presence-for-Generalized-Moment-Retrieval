"""Offline reproducible plotting from saved metrics, never from selected Test peaks."""
import argparse,csv,hashlib,json,shutil,sys,os,fcntl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from paths import *
from oracle_presence import cached_oracle_presence,selected_val_reference
NAMES={'moment_detr':'Moment-DETR-GMR','eatr':'EaTR-GMR','qd_detr':'QD-DETR-GMR','flashvtg':'FlashVTG-GMR (Multi-TQAux adaptation)'}
LABELS=dict(zip(METHODS,['GMR','GMR + Multi','GMR + GEQ Frozen','GMR + GEQ Full','GMR + Multi + GEQ Frozen','GMR + Multi + GEQ Full']))
COLORS=['#64748b','#e69f00','#009e73','#56b4e9','#0072b2','#cc79a7']
STYLES=dict(zip(METHODS,COLORS))
MARKERS=dict(zip(METHODS,['o','s','^','v','D','P']))
LABELS['multi_positive_only']='GMR + Multi (positive-only ablation)'
STYLES['multi_positive_only']='#dc2626'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'charades-seed2023'})

def save(fig,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 for suffix in ('png','pdf','svg'):
  metadata={'Date':None} if suffix=='svg' else {'CreationDate':None,'ModDate':None} if suffix=='pdf' else None
  target=path.with_suffix('.'+suffix);temporary=target.with_name(target.stem+f'.{os.getpid()}.tmp.'+suffix)
  fig.savefig(temporary,dpi=200,bbox_inches='tight',metadata=metadata);temporary.replace(target)
 plt.close(fig)

def write_csv(path,records):
 if not records:return
 path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
 with temporary.open('w') as f:
  fields=list(dict.fromkeys(k for r in records for k in r))
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
 temporary.replace(path)

def render(snapshot,dest):
 methods=snapshot['methods'];series=snapshot['series'];baseline=series['gmr']['grid'];fig,ax=plt.subplots(2,3,figsize=(15,8.5));g1,g3,delta,trade,f1,fixed=ax.flat
 method_handles=[]
 for m in methods:
  metrics=series[m];grid=metrics['grid'];x=[r['threshold'] for r in grid];c=STYLES[m]
  label=f"{LABELS[m].replace('Multi','Multi-TQAux') if snapshot['backbone']=='flashvtg' else LABELS[m]} (mAP {metrics['mAP']:.2f}%)"
  style=dict(color=c,marker=MARKERS[m],markevery=20,markersize=4)
  handle,=g1.plot(x,[r['gmiou_1'] for r in grid],label=label,**style);method_handles.append(handle)
  g3.plot(x,[r['gmiou_3'] for r in grid],**style)
  delta.plot(x,[r['gmiou_1']-b['gmiou_1'] for r,b in zip(grid,baseline)],lw=2,**style)
  trade.plot([r['positive_retention'] for r in grid],[r['negative_rejection'] for r in grid],**style)
  f1.plot(x,[r['rejection_f1'] for r in grid],**style)
  fixed.plot([.4,.5,.6],[grid[i]['gmiou_1'] for i in (40,50,60)],marker=MARKERS[m],color=c)
  for i in (40,50,60):trade.scatter(grid[i]['positive_retention'],grid[i]['negative_rejection'],s=15,color=c)
 has_oracle=all('oracle_presence' in series[m] for m in methods)
 if has_oracle:
  short={'gmr':'GMR','multi':'Multi','gmr_geq_frozen':'GMR + GEQ-F','multi_geq_frozen':'Multi + GEQ-F','gmr_geq_full':'GMR + GEQ-Full','multi_geq_full':'Multi + GEQ-Full'}
  for axis,k in [(g1,1),(g3,3),(fixed,1)]:
   groups={}
   for m in methods:
    oracle=series[m]['oracle_presence'][f'gmiou_{k}']
    assert max(r[f'gmiou_{k}'] for r in series[m]['grid'])<=oracle+1e-12,'Curve exceeds fixed-candidate oracle'
    groups.setdefault(oracle,[]).append(m)
   handles=[]
   for ceiling,members in groups.items():
    label=' / '.join(short[m] for m in members)+f': {ceiling:.4f}'
    line=axis.axhline(ceiling,color=STYLES[members[0]],ls=(0,(7,3)),lw=1.8,alpha=1.,zorder=5,label=label);handles.append(line)
   low=min(groups);high=max(groups);bounds=f'{low:.4f}' if low==high else f'{low:.4f}–{high:.4f}'
   axis.text(.98,min(.9,high+.055),f'Oracle upper bound @{k}\n{bounds}',transform=axis.get_yaxis_transform(),ha='right',va='bottom',fontsize=9,fontweight='bold',bbox=dict(facecolor='white',edgecolor='#9ca3af',alpha=.95,boxstyle='round,pad=.3'))
   if axis is not fixed:axis.legend(handles=handles,title='Oracle upper bounds: perfect presence',loc='lower left',fontsize=8,title_fontsize=8,framealpha=.95)
  trade.scatter([1],[1],marker='*',s=90,color='#111827',clip_on=False,zorder=8)
  trade.annotate('Oracle: perfect presence',(1,1),xytext=(.40,.91),arrowprops=dict(arrowstyle='->',color='#4b5563'),fontsize=8)
  f1.axhline(1,color='#111827',ls='--',lw=1.8)
  f1.text(.98,.96,'Oracle = 1',ha='right',va='top',transform=f1.transAxes,fontsize=8)
  for m in methods:
   delta.plot([r['threshold'] for r in baseline],[series[m]['oracle_presence']['gmiou_1']-r['gmiou_1'] for r in baseline],color=STYLES[m],ls='--',lw=1.6,alpha=.9)
  delta.text(.98,.97,'Dashed: Oracle − observed GMR',transform=delta.transAxes,ha='right',va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
  detail=delta.inset_axes([.12,.42,.65,.36]);draw_relative(detail,snapshot,zoom=True)
  detail.set_title('Fixed-point detail (0.35–0.65)',fontsize=7);detail.set_xlabel('');detail.set_ylabel('');detail.tick_params(labelsize=6)
 for a,title,xlabel,ylabel in [(g1,'(a) G-mIoU@1','Presence threshold','G-mIoU@1 (0–1)'),(g3,'(b) G-mIoU@3','Presence threshold','G-mIoU@3 (0–1)'),(delta,'(c) Relative to GMR','Presence threshold','Δ G-mIoU@1'),(trade,'(d) Rejection / retention','Positive query retention','Negative query rejection'),(f1,'(e) Rejection F1','Presence threshold','Rej-F1 (0–1)'),(fixed,'(f) Fixed operating points','Presence threshold','G-mIoU@1 (0–1)')]:
  a.set(title=title,xlabel=xlabel,ylabel=ylabel,xlim=(0,1));a.grid(alpha=.2)
  if a is not delta:a.set_ylim(0,1)
 for a in (g1,g3,fixed):a.axhline(series['gmr']['all_reject_gmiou'],color='#9ca3af',ls=':',lw=1)
 delta.axhline(0,color='#9ca3af',lw=.7)
 fixed.set_xticks([.4,.5,.6]);fixed.set_xlim(.35,.65)
 fig.suptitle(f"{NAMES[snapshot['backbone']]} | Charades {snapshot['split']} | {snapshot['comparison']}",y=1.04)
 handles=method_handles+[Line2D([],[],color='#9ca3af',ls=':',label=f"All reject = {series['gmr']['all_reject_gmiou']:.2f}")]
 if has_oracle:handles.append(Line2D([],[],color='#111827',ls='--',lw=1.8,label='Oracle upper bounds: dashed, fixed localization'))
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.005),ncol=2,frameon=False)
 caption='seed 2023 | Val-selected checkpoints | 101 thresholds | presence score > threshold'
 if has_oracle:caption+='\nOracle fixes localization; in (c), Oracle(method) minus observed GMR. Diagnostic only; no threshold or checkpoint selection.'
 fig.text(.5,-.045,caption,ha='center',fontsize=9)
 fig.tight_layout(rect=(0,0,1,.94));save(fig,dest)
 records=[]
 for m in methods:
  for i,r in enumerate(series[m]['grid']):
   record=dict(method=m,**r,relative_to_gmr=r['gmiou_1']-baseline[i]['gmiou_1'])
   if 'oracle_presence' in series[m]:record.update(oracle_gmiou_1=series[m]['oracle_presence']['gmiou_1'],oracle_gmiou_3=series[m]['oracle_presence']['gmiou_3'],oracle_relative_to_gmr=series[m]['oracle_presence']['gmiou_1']-baseline[i]['gmiou_1'])
   records.append(record)
 write_csv(dest.with_suffix('.csv'),records)

def relative_records(snapshot):
 baseline=snapshot['series']['gmr']['grid'];records=[]
 for m in snapshot['methods']:
  values=snapshot['series'][m];ceiling=values['oracle_presence']['gmiou_1']
  for r,b in zip(values['grid'],baseline):
   actual=r['gmiou_1']-b['gmiou_1'];oracle=ceiling-b['gmiou_1'];assert actual<=oracle+1e-12
   records.append(dict(method=m,threshold=r['threshold'],actual_relative_to_gmr=actual,oracle_relative_to_gmr=oracle))
 return records

def draw_relative(axis,snapshot,zoom=False):
 records=relative_records(snapshot);visible=[]
 for m in snapshot['methods']:
  values=[r for r in records if r['method']==m];x=[r['threshold'] for r in values];actual=[r['actual_relative_to_gmr'] for r in values];oracle=[r['oracle_relative_to_gmr'] for r in values]
  axis.plot(x,actual,color=STYLES[m],lw=2);axis.plot(x,oracle,color=STYLES[m],ls='--',lw=1.6)
  if zoom:
   for r in values:
    if .35<=r['threshold']<=.65:visible.extend([r['actual_relative_to_gmr'],r['oracle_relative_to_gmr']])
   for index in [40,50,60]:
    axis.scatter([values[index]['threshold']],[values[index]['actual_relative_to_gmr']],s=14,color=STYLES[m])
    axis.scatter([values[index]['threshold']],[values[index]['oracle_relative_to_gmr']],s=21,marker='^',facecolors='none',edgecolors=STYLES[m])
 axis.axhline(0,color='#374151',lw=.8);axis.grid(alpha=.2);axis.set(xlabel='Presence threshold',ylabel='Δ G-mIoU@1 relative to observed GMR',xlim=(.35,.65) if zoom else (0,1))
 if zoom:
  low=min([0]+visible);high=max([0]+visible);pad=max(.003,.12*(high-low));axis.set_ylim(low-pad,high+pad);axis.set_xticks([.4,.5,.6])

def render_relative(snapshot,destination):
 fig,axes=plt.subplots(1,2,figsize=(13,5.5));draw_relative(axes[0],snapshot);draw_relative(axes[1],snapshot,zoom=True)
 axes[0].set_title('Relative to GMR: full threshold range');axes[0].axvspan(.35,.65,color='#9ca3af',alpha=.1)
 axes[1].set_title('Detail at fixed operating points (0.35–0.65)')
 handles=[Line2D([],[],color=STYLES[m],label=f"{LABELS[m]} (mAP {snapshot['series'][m]['mAP']:.2f}%)") for m in snapshot['methods']]
 handles.extend([Line2D([],[],color='#111827',lw=2,label='Solid: actual increment'),Line2D([],[],color='#111827',ls='--',lw=1.6,label='Dashed: Oracle increment upper bound')])
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False,fontsize=9)
 fig.suptitle(f"{NAMES[snapshot['backbone']]} | Charades {snapshot['split']} | Relative to GMR",fontsize=13)
 fig.text(.5,.015,'Oracle increment = fixed-candidate Oracle(method) − observed GMR at the same threshold. Negative actual increments are retained.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.06,1,.76));save(fig,destination);write_csv(destination.with_suffix('.csv'),relative_records(snapshot))

def render_selection(snapshot,destination):
 fig,axes=plt.subplots(1,3,figsize=(20,5.5));references=snapshot.get('selected_oracles',{})
 for m,history in snapshot['history'].items():
  history=[r for r in history if r.get('validated',True) and r.get('selection') is not None]
  if not history:continue
  axis=axes[1 if '_geq_' in m else 0]
  label=LABELS[m].replace('Multi','Multi-TQAux') if snapshot['backbone']=='flashvtg' else LABELS[m]
  axis.plot([r['epoch'] for r in history],[r['selection'] for r in history],'.-',label=label,color=STYLES[m])
  reference=references.get(m)
  if reference:
   if '_geq_' in m:
    axis.vlines(reference['epoch'],reference['gmiou_1_mean'],reference['oracle_presence']['gmiou_1'],color=STYLES[m],ls=':',lw=1)
    axis.scatter([reference['epoch']],[reference['oracle_presence']['gmiou_1']],marker='^',s=48,facecolors='none',edgecolors=STYLES[m],zorder=6)
   else:axis.scatter([reference['epoch']],[reference['mAP_percent']],marker='D',s=50,facecolors='none',edgecolors=STYLES[m],zorder=6)
 for axis,label in zip(axes[:2],['Val positive-query mAP (%)','Val mean G-mIoU@1 at .4/.5/.6 (0–1)']):
  axis.set(xlabel='Epoch',ylabel=label);axis.grid(alpha=.2)
  if axis.lines:axis.legend(fontsize=8)
 axes[1].set_ylim(0,1)
 axes[0].text(.02,.02,'Presence oracle leaves ungated mAP unchanged',transform=axes[0].transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.85,edgecolor='none'))
 axes[0].set_title('Stage A: mAP (oracle unchanged)',fontsize=10)
 axes[1].set_title('Stage B: triangle = oracle at picked epoch',fontsize=10)
 names=[]
 for index,(m,r) in enumerate(references.items()):
  names.append(LABELS[m].replace('Multi','Multi-TQAux') if snapshot['backbone']=='flashvtg' else LABELS[m])
  for k,offset,observed_marker,oracle_marker in [(1,-.15,'o','^'),(3,.15,'s','D')]:
   observed=r[f'gmiou_{k}_mean'];oracle=r['oracle_presence'][f'gmiou_{k}'];y=index+offset
   axes[2].plot([observed,oracle],[y,y],color=STYLES[m],lw=1)
   axes[2].scatter([observed],[y],color=STYLES[m],marker=observed_marker,s=23)
   axes[2].scatter([oracle],[y],edgecolors=STYLES[m],facecolors='none',marker=oracle_marker,s=37)
 axes[2].set(xlim=(0,1),xlabel='G-mIoU (0–1)',title='Current Val-best: actual and presence oracle')
 axes[2].set_yticks(range(len(names)),names,fontsize=8);axes[2].invert_yaxis();axes[2].grid(alpha=.2,axis='x')
 legend=[Line2D([],[],color='#374151',marker=marker,ls='none',markerfacecolor='none' if oracle else '#374151',label=label) for marker,oracle,label in [('o',False,'Actual G1 mean'),('^',True,'Oracle G1'),('s',False,'Actual G3 mean'),('D',True,'Oracle G3')]]
 axes[2].legend(handles=legend,fontsize=7.5,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.15))
 if not names:axes[2].text(.5,.5,'Oracle not retained in this historical snapshot',ha='center',transform=axes[2].transAxes,fontsize=9)
 fig.suptitle(f"{NAMES[snapshot['backbone']]} | Val selection, seed 2023 (dots = actual validations)")
 fig.text(.5,.01,'Oracle references apply only to current exported Val-best candidates; no per-epoch Oracle is invented for historical predictions.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.13,1,.95));save(fig,destination)

def render_oracle_detail(snapshot,destination):
 methods=snapshot['methods'];fig,axes=plt.subplots(len(methods),2,figsize=(12,3.1*len(methods)),squeeze=False)
 records=[]
 for row,m in enumerate(methods):
  series=snapshot['series'][m];oracle=series['oracle_presence'];grid=series['grid']
  for col,k in enumerate((1,3)):
   axis=axes[row,col];bound=oracle[f'gmiou_{k}'];x=[r['threshold'] for r in grid];y=[r[f'gmiou_{k}'] for r in grid]
   axis.plot(x,y,color=STYLES[m],lw=2,label='Actual G-mIoU')
   axis.axhline(bound,color='black',ls=(0,(7,3)),lw=2.6,zorder=6,label=f'Oracle upper bound = {bound:.4f}')
   axis.text(.97,min(.92,bound+.065),f'ORACLE = {bound:.4f}',ha='right',va='bottom',transform=axis.get_yaxis_transform(),fontweight='bold',fontsize=11,bbox=dict(facecolor='white',edgecolor='black',boxstyle='round,pad=.3'))
   axis.set(xlim=(0,1),ylim=(0,1),xlabel='Presence threshold',ylabel=f'G-mIoU@{k} (0–1)',title=f'{LABELS[m]} | G-mIoU@{k}');axis.grid(alpha=.2);axis.legend(loc='lower left',fontsize=9)
   records.append(dict(method=m,k=k,oracle=bound,prediction_sha256=oracle['source']['prediction_sha256']))
 fig.suptitle(f"{NAMES[snapshot['backbone']]} | Charades {snapshot['split']} | Oracle upper bounds",fontsize=14)
 fig.text(.5,.005,'Black dashed line: perfect presence, with each model\'s localization candidates fixed. Colored curve: original scored predictions.',ha='center',fontsize=10)
 fig.tight_layout(rect=(0,.025,1,.97));save(fig,destination);write_csv(destination.with_suffix('.csv'),records)

def selection_plot(b):
 history={};sources={}
 for m in (*METHODS,'multi_positive_only') if b=='eatr' else METHODS:
  p=OUT/'runs'/b/m/'history.jsonl'
  if not p.exists():continue
  payload=p.read_bytes();history[m]=[json.loads(l) for l in payload.decode().splitlines() if l.strip()];sources[m]=hashlib.sha256(payload).hexdigest()
  write_csv(OUT/'figures'/b/'val'/f'{m}_history.csv',[{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in history[m]])
 if not history:return False
 references={}
 for m,h in history.items():
  reference=selected_val_reference(OUT/'runs'/b/m,OUT/'figures'/b/'val'/f'{m}_oracle_presence.json',h)
  if reference:references[m]=reference
 write_csv(OUT/'figures'/b/'val/oracle_selected.csv',[dict(method=m,epoch=r['epoch'],actual_gmiou_1=r['gmiou_1_mean'],actual_gmiou_3=r['gmiou_3_mean'],oracle_gmiou_1=r['oracle_presence']['gmiou_1'],oracle_gmiou_3=r['oracle_presence']['gmiou_3'],reference='current Val-best only',**r['source']) for m,r in references.items()])
 snapshot=dict(type='val_selection',backbone=b,history=history,source_hashes=sources,selected_oracles=references)
 dump(OUT/'figures'/b/'val/selection.snapshot.json',snapshot);render_selection(snapshot,OUT/'figures'/b/'val/selection')
 for split in ('test_id','test_ood'):
  dest=OUT/'figures'/b/split;dest.mkdir(parents=True,exist_ok=True)
  for suffix in ('png','pdf','svg'):
   target=dest/('val_selection.'+suffix);temporary=target.with_name(target.name+f'.{os.getpid()}.tmp');shutil.copyfile(OUT/'figures'/b/'val'/('selection.'+suffix),temporary);temporary.replace(target)
 return True

def summarize(b,s,series):
 records=[]
 for m,v in series.items():
  cfg=json.loads((OUT/'runs'/b/m/'config.json').read_text());hist=rows(OUT/'runs'/b/m/'history.jsonl')
  record=dict(backbone=b,split=s,method=m,mAP_percent=v['mAP'],mR1_percent=v['mR1'],mR5_percent=v['mR5'],AUROC_percent=v['AUROC'],R1_tIoU_05_percent=v['R1']['0.5'],R1_tIoU_07_percent=v['R1']['0.7'],mR_plus='N/A',multi_gt_coverage='N/A',parameters=cfg['parameter_count'],trainable_parameters=cfg['trainable_parameters'],training_seconds=sum(h['seconds'] for h in hist),training_time_basis='Recorded completed-epoch training plus Val wall time; interrupted partial attempts excluded',peak_memory_bytes=max(h['peak_memory_bytes'] for h in hist),inference_ms_per_row=v['inference_ms_per_row'])
  if 'oracle_presence' in v:record.update(oracle_gmiou_1=v['oracle_presence']['gmiou_1'],oracle_gmiou_3=v['oracle_presence']['gmiou_3'])
  record.update(oracle_AUROC_percent=100.,oracle_mAP_percent=v['mAP'],oracle_mR1_percent=v['mR1'],oracle_mR5_percent=v['mR5'])
  for key in ('gmiou_1','gmiou_3','rejection_f1','negative_rejection','positive_retention'):
   for i in (40,50,60):record[f'{key}_at_{i/100}']=v['grid'][i][key]
   if key.startswith('gmiou'):record[key+'_mean']=float(np.mean([v['grid'][i][key] for i in (40,50,60)]))
  profile_path=OUT/'runs'/b/m/'PROFILE.json'
  if profile_path.exists():
   profile=json.loads(profile_path.read_text());record.update(cuda_forward_ms_per_query=profile['total']['mean_ms_per_query'],matched_native_cuda_ms_per_query=profile['matched_native_trunk']['mean_ms_per_query'],incremental_geq_cuda_ms_per_query=profile['incremental_geq_ms_per_query'],latency_profile_batch=profile['batch_size'])
  else:record.update(cuda_forward_ms_per_query=None,matched_native_cuda_ms_per_query=None,incremental_geq_cuda_ms_per_query=None,latency_profile_batch=None)
  records.append(record)
 write_csv(OUT/'figures'/b/s/'summary.csv',records)
 if len(series)==6:
  keys=['mAP_percent','mR1_percent','mR5_percent','AUROC_percent','gmiou_1_mean','gmiou_3_mean']
  snapshot=dict(type='oracle_heatmap',actual=[[r[k] for r in records] for k in keys],oracle=[[metric_oracle(r,k) for r in records] for k in keys],xlabels=[LABELS[r['method']] for r in records],ylabels=keys,precision_per_row=[4 if 'gmiou' in k else 2 for k in keys],figsize=[12,6],title=f'{NAMES[b]} | {s} | actual / Oracle presence (O)\nColors compare actual methods within each row; ungated localization unchanged')
  dest=OUT/'figures'/b/s/'heatmap';dump(dest.with_suffix('.snapshot.json'),snapshot);render_oracle_heatmap(snapshot,dest)
 return records

def metric_oracle(record,metric):
 if metric=='gmiou_1_mean':return record['oracle_gmiou_1']
 if metric=='gmiou_3_mean':return record['oracle_gmiou_3']
 if metric=='AUROC_percent':return 100.
 return record[metric]

def render_oracle_heatmap(snapshot,destination):
 values=np.array(snapshot['actual'],dtype=float);oracles=np.array(snapshot['oracle'],dtype=float);color=np.full_like(values,np.nan)
 for row in range(len(values)):
  if np.isfinite(values[row]).any():color[row]=(values[row]-np.nanmin(values[row]))/max(float(np.nanmax(values[row])-np.nanmin(values[row])),1e-12)
 fig,axis=plt.subplots(figsize=snapshot['figsize']);axis.imshow(np.ma.masked_invalid(color),vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');axis.set_xticks(range(values.shape[1]),snapshot['xlabels'],rotation=20,ha='right');axis.set_yticks(range(values.shape[0]),snapshot['ylabels'])
 records=[]
 for row in range(values.shape[0]):
  for col in range(values.shape[1]):
   if np.isfinite(values[row,col]):
    digits=snapshot['precision_per_row'][row];axis.text(col,row,f'{values[row,col]:.{digits}f}\nO {oracles[row,col]:.{digits}f}',ha='center',va='center',fontsize=9,color='white' if color[row,col]>.65 else 'black')
    records.append(dict(row=snapshot['ylabels'][row],method=snapshot['xlabels'][col],actual=values[row,col],oracle=oracles[row,col]))
 axis.set_title(snapshot['title'],fontsize=10);save(fig,destination);write_csv(destination.with_suffix('.csv'),records)

def global_heatmaps(records):
 for split in ('test_id','test_ood'):
  for metric in ('mAP_percent','mR1_percent','mR5_percent','AUROC_percent','gmiou_1_mean','gmiou_3_mean'):
   values=np.full((4,6),np.nan);oracles=np.full((4,6),np.nan)
   for record in records:
    if record['split']==split:
     row=BACKBONES.index(record['backbone']);col=METHODS.index(record['method']);values[row,col]=record[metric];oracles[row,col]=metric_oracle(record,metric)
   if not np.isfinite(values).any():continue
   snapshot=dict(type='oracle_heatmap',split=split,metric=metric,actual=[[float(v) if np.isfinite(v) else None for v in row] for row in values],oracle=[[float(v) if np.isfinite(v) else None for v in row] for row in oracles],xlabels=[LABELS[m] for m in METHODS],ylabels=[NAMES[b] for b in BACKBONES],precision_per_row=[4 if 'gmiou' in metric else 2]*4,figsize=[14,5.5],title=f'{split} | {metric} | actual / Oracle presence (O)\nColors compare actual values within a backbone row; ungated localization unchanged')
   dest=OUT/'reports'/f'heatmap_{split}_{metric}';dump(dest.with_suffix('.snapshot.json'),snapshot);render_oracle_heatmap(snapshot,dest)


def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot');p.add_argument('--output');args=p.parse_args()
 if args.snapshot:
  snapshot=json.loads(Path(args.snapshot).read_text())
  if snapshot.get('type')=='existence_score_distributions':
   from plot_existence_score_distributions import render as renderer
  else:renderer={'val_selection':render_selection,'oracle_heatmap':render_oracle_heatmap,'oracle_detail':render_oracle_detail,'relative_to_gmr':render_relative}.get(snapshot.get('type'),render)
  renderer(snapshot,Path(args.output));return
 (OUT/'logs').mkdir(parents=True,exist_ok=True)
 with (OUT/'logs/plot_report.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);generate_reports()

def generate_reports():
 all_records=[];index=['# Charades Multi + GEQ, seed 2023','', 'Test-ID is primary; Test-OOD retains all 7,440 rows including 2,170 duplicate appended records. Test never selects checkpoints. Flash Multi is the Multi-TQAux structural adaptation. Single-GT data: mR+ and multi-GT coverage are N/A.','']
 for b in BACKBONES:
  val=selection_plot(b);body=[f'# {NAMES[b]}','', 'Status reflects files actually completed.','', 'Dashed colored lines show each model\'s perfect-presence ceiling with its own localization candidates held fixed. Oracle G-mIoU@k = (negative rows + sum of positive-query set-IoU@k) / all rows. Here each Test split is balanced, so this equals 0.5 + 0.5 times positive-query mean set-IoU@k. It is generally below 1.0; G-mIoU@1 and @3 have separate ceilings under the evaluator\'s set-IoU definition. These are post-hoc label-based references, not scored threshold curves or model-selection criteria.','']
  if val:body+=['![Val selection](val/selection.png)','']
  for s in ('test_id','test_ood'):
   series={};provenance={}
   for m in METHODS:
    d=OUT/'runs'/b/m/s
    if not (d/'COMPLETE.json').exists():continue
    series[m]=json.loads((d/'metrics.json').read_text());provenance[m]=json.loads((d/'COMPLETE.json').read_text())
    series[m]['oracle_presence']=cached_oracle_presence(d/'predictions.jsonl',OUT/'data'/f'{s}.jsonl',OUT/'figures'/b/s/f'{m}_oracle_presence.json',provenance[m]['prediction_sha256'])
   if not series:continue
   all_records+=summarize(b,s,series)
   body += [f'## {s}','',f'{len(series)}/6 methods evaluated. G-mIoU uses 0–1; mAP, mR, R@1 and AUROC use percent.','']
   for name,methods in [('frozen',['gmr','gmr_geq_frozen','multi','multi_geq_frozen']),('full',['gmr','gmr_geq_full','multi','multi_geq_full']),('multi_comparison',['multi','multi_geq_frozen','multi_geq_full'])]:
    if not all(m in series for m in ['gmr']+methods):continue
    snapshot=dict(backbone=b,split=s,comparison=name,methods=methods,series=series,provenance=provenance)
    dest=OUT/'figures'/b/s/name;dump(dest.with_suffix('.snapshot.json'),snapshot);render(snapshot,dest);body += [f'![{name}]({s}/{name}.png)','']
   if len(series)==6:body += [f'![Summary heatmap]({s}/heatmap.png)','']
   body += [f'[Metrics CSV]({s}/summary.csv)','',f'![Val selection]({s}/val_selection.png)','']
  (OUT/'figures'/b).mkdir(parents=True,exist_ok=True);(OUT/'figures'/b/'README.md').write_text('\n'.join(body)+'\n');index += [f'## {NAMES[b]}','',f'[All figures and metrics](../figures/{b}/README.md)','']
  if (OUT/'figures'/b/'test_id/frozen.png').exists():index += [f'![Frozen comparison](../figures/{b}/test_id/frozen.png)','']
 write_csv(OUT/'reports/all_metrics.csv',all_records)
 global_heatmaps(all_records)
 from plot_existence_score_distributions import generate_score_distributions
 generate_score_distributions()
 index += ['## Existence-score distributions','', '[All score-distribution figures and statistics](../figures/score_distributions/README.md)','']
 trained=sum((OUT/'runs'/b/m/'TRAINED.json').exists() for b in BACKBONES for m in METHODS)
 index.insert(2,f'Training completed: {trained}/24. Test evaluations completed: {len(all_records)}/48. Not complete until all acceptance checks and remote publication succeed.')
 (OUT/'reports/README.md').write_text('\n'.join(index)+'\n')
if __name__=='__main__':main()
