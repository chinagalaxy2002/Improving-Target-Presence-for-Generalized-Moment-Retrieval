"""Offline reproducible plotting from saved metrics, never from selected Test peaks."""
import argparse,csv,hashlib,json,shutil,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import *
NAMES={'moment_detr':'Moment-DETR-GMR','eatr':'EaTR-GMR','qd_detr':'QD-DETR-GMR','flashvtg':'FlashVTG-GMR (Multi-TQAux adaptation)'}
LABELS=dict(zip(METHODS,['GMR','GMR + Multi','GMR + GEQ Frozen','GMR + GEQ Full','GMR + Multi + GEQ Frozen','GMR + Multi + GEQ Full']))
COLORS=['#64748b','#e69f00','#009e73','#56b4e9','#0072b2','#cc79a7']
STYLES=dict(zip(METHODS,COLORS))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'svg.hashsalt':'charades-seed2023'})

def save(fig,path):
 path.parent.mkdir(parents=True,exist_ok=True)
 for suffix in ('png','pdf','svg'):
  metadata={'Date':None} if suffix=='svg' else {'CreationDate':None,'ModDate':None} if suffix=='pdf' else None
  fig.savefig(path.with_suffix('.'+suffix),dpi=200,bbox_inches='tight',metadata=metadata)
 plt.close(fig)

def write_csv(path,records):
 if not records:return
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def render(snapshot,dest):
 methods=snapshot['methods'];series=snapshot['series'];baseline=series['gmr']['grid'];fig,ax=plt.subplots(2,3,figsize=(15,8.5));g1,g3,delta,trade,f1,fixed=ax.flat
 for m in methods:
  metrics=series[m];grid=metrics['grid'];x=[r['threshold'] for r in grid];c=STYLES[m]
  label=f"{LABELS[m].replace('Multi','Multi-TQAux') if snapshot['backbone']=='flashvtg' else LABELS[m]} (mAP {metrics['mAP']:.2f}%)"
  g1.plot(x,[r['gmiou_1'] for r in grid],color=c,label=label)
  g3.plot(x,[r['gmiou_3'] for r in grid],color=c)
  delta.plot(x,[r['gmiou_1']-b['gmiou_1'] for r,b in zip(grid,baseline)],color=c)
  trade.plot([r['positive_retention'] for r in grid],[r['negative_rejection'] for r in grid],color=c)
  f1.plot(x,[r['rejection_f1'] for r in grid],color=c)
  fixed.plot([.4,.5,.6],[grid[i]['gmiou_1'] for i in (40,50,60)],'o-',color=c)
  for i in (40,50,60):trade.scatter(grid[i]['positive_retention'],grid[i]['negative_rejection'],s=15,color=c)
 for a,title,xlabel,ylabel in [(g1,'(a) G-mIoU@1','Presence threshold','G-mIoU@1 (0–1)'),(g3,'(b) G-mIoU@3','Presence threshold','G-mIoU@3 (0–1)'),(delta,'(c) Relative to GMR','Presence threshold','Δ G-mIoU@1'),(trade,'(d) Rejection / retention','Positive query retention','Negative query rejection'),(f1,'(e) Rejection F1','Presence threshold','Rej-F1 (0–1)'),(fixed,'(f) Fixed operating points','Presence threshold','G-mIoU@1 (0–1)')]:
  a.set(title=title,xlabel=xlabel,ylabel=ylabel,xlim=(0,1));a.grid(alpha=.2)
  if a is not delta:a.set_ylim(0,1)
 for a in (g1,g3,fixed):a.axhline(.5,color='#9ca3af',ls=':',lw=1,label='All reject = 0.5')
 delta.axhline(0,color='#9ca3af',lw=.7)
 fixed.set_xticks([.4,.5,.6]);fixed.set_xlim(.35,.65)
 fig.suptitle(f"{NAMES[snapshot['backbone']]} | Charades {snapshot['split']} | {snapshot['comparison']}",y=1.04)
 handles,labels=g1.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.005),ncol=2,frameon=False)
 fig.text(.5,-.025,'seed 2023 | Val-selected checkpoints | 101 thresholds | presence score > threshold | all-reject reference requires positive-retention interpretation',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,0,1,.94));save(fig,dest)
 records=[]
 for m in methods:
  for r in series[m]['grid']:records.append(dict(method=m,**r))
 write_csv(dest.with_suffix('.csv'),records)

def render_selection(snapshot,destination):
 fig,axes=plt.subplots(1,2,figsize=(13,4.5))
 for m,history in snapshot['history'].items():
  axis=axes[1 if '_geq_' in m else 0]
  label=LABELS[m].replace('Multi','Multi-TQAux') if snapshot['backbone']=='flashvtg' else LABELS[m]
  axis.plot([r['epoch'] for r in history],[r['selection'] for r in history],label=label,color=STYLES[m])
 for axis,label in zip(axes,['Val positive-query mAP (%)','Val mean G-mIoU@1 at .4/.5/.6 (0–1)']):
  axis.set(xlabel='Epoch',ylabel=label);axis.grid(alpha=.2)
  if axis.lines:axis.legend(fontsize=8)
 axes[1].set_ylim(0,1)
 fig.suptitle(f"{NAMES[snapshot['backbone']]} | Val selection, seed 2023");fig.tight_layout();save(fig,destination)

def selection_plot(b):
 history={};sources={}
 for m in METHODS:
  p=OUT/'runs'/b/m/'history.jsonl'
  if not p.exists():continue
  payload=p.read_bytes();history[m]=[json.loads(l) for l in payload.decode().splitlines() if l.strip()];sources[m]=hashlib.sha256(payload).hexdigest()
  write_csv(OUT/'figures'/b/'val'/f'{m}_history.csv',[{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in history[m]])
 if not history:return False
 snapshot=dict(type='val_selection',backbone=b,history=history,source_hashes=sources)
 dump(OUT/'figures'/b/'val/selection.snapshot.json',snapshot);render_selection(snapshot,OUT/'figures'/b/'val/selection')
 for split in ('test_id','test_ood'):
  dest=OUT/'figures'/b/split;dest.mkdir(parents=True,exist_ok=True)
  for suffix in ('png','pdf','svg'):shutil.copy2(OUT/'figures'/b/'val'/('selection.'+suffix),dest/('val_selection.'+suffix))
 return True

def summarize(b,s,series):
 records=[]
 for m,v in series.items():
  cfg=json.loads((OUT/'runs'/b/m/'config.json').read_text());hist=rows(OUT/'runs'/b/m/'history.jsonl')
  record=dict(backbone=b,split=s,method=m,mAP_percent=v['mAP'],mR1_percent=v['mR1'],mR5_percent=v['mR5'],AUROC_percent=v['AUROC'],R1_tIoU_05_percent=v['R1']['0.5'],R1_tIoU_07_percent=v['R1']['0.7'],mR_plus='N/A',multi_gt_coverage='N/A',parameters=cfg['parameter_count'],trainable_parameters=cfg['trainable_parameters'],training_seconds=sum(h['seconds'] for h in hist),training_time_basis='Recorded completed-epoch training plus Val wall time; interrupted partial attempts excluded',peak_memory_bytes=max(h['peak_memory_bytes'] for h in hist),inference_ms_per_row=v['inference_ms_per_row'])
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
  values=np.array([[r[k] for r in records] for k in keys]);minimum=values.min(1,keepdims=True);span=values.max(1,keepdims=True)-minimum
  fig,ax=plt.subplots(figsize=(11,4));ax.imshow((values-minimum)/np.maximum(span,1e-12),aspect='auto',vmin=0,vmax=1,cmap='YlGnBu');ax.set_xticks(range(6),[LABELS[r['method']] for r in records],rotation=22,ha='right');ax.set_yticks(range(len(keys)),keys)
  for y in range(len(keys)):
   for x in range(6):ax.text(x,y,f'{values[y,x]:.4f}' if 'gmiou' in keys[y] else f'{values[y,x]:.2f}',ha='center',va='center',color='black' if (values[y,x]-minimum[y,0])/max(span[y,0],1e-12)<.65 else 'white')
  ax.set_title(f'{NAMES[b]} | {s} | colors compare methods within each row only');save(fig,OUT/'figures'/b/s/'heatmap')
 return records

def global_heatmaps(records):
 for split in ('test_id','test_ood'):
  for metric in ('mAP_percent','mR1_percent','mR5_percent','AUROC_percent','gmiou_1_mean','gmiou_3_mean'):
   values=np.full((4,6),np.nan)
   for record in records:
    if record['split']==split:values[BACKBONES.index(record['backbone']),METHODS.index(record['method'])]=record[metric]
   if not np.isfinite(values).any():continue
   color=np.full_like(values,np.nan)
   for row in range(4):
    if np.isfinite(values[row]).any():color[row]=(values[row]-np.nanmin(values[row]))/max(float(np.nanmax(values[row])-np.nanmin(values[row])),1e-12)
   fig,axis=plt.subplots(figsize=(14,4.5));axis.imshow(np.ma.masked_invalid(color),vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');axis.set_xticks(range(6),[LABELS[m] for m in METHODS],rotation=20,ha='right');axis.set_yticks(range(4),[NAMES[b] for b in BACKBONES])
   for row in range(4):
    for col in range(6):
     if np.isfinite(values[row,col]):axis.text(col,row,f'{values[row,col]:.4f}' if 'gmiou' in metric else f'{values[row,col]:.2f}',ha='center',va='center',color='white' if color[row,col]>.65 else 'black')
   axis.set_title(f'{split} | {metric} | colors compare methods within a backbone row only');save(fig,OUT/'reports'/f'heatmap_{split}_{metric}')


def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot');p.add_argument('--output');args=p.parse_args()
 if args.snapshot:
  snapshot=json.loads(Path(args.snapshot).read_text());(render_selection if snapshot.get('type')=='val_selection' else render)(snapshot,Path(args.output));return
 all_records=[];index=['# Charades Multi + GEQ, seed 2023','', 'Test-ID is primary; Test-OOD retains all 7,440 rows including 2,170 duplicate appended records. Test never selects checkpoints. Flash Multi is the Multi-TQAux structural adaptation. Single-GT data: mR+ and multi-GT coverage are N/A.','']
 for b in BACKBONES:
  val=selection_plot(b);body=[f'# {NAMES[b]}','', 'Status reflects files actually completed.','']
  if val:body+=['![Val selection](val/selection.png)','']
  for s in ('test_id','test_ood'):
   series={};provenance={}
   for m in METHODS:
    d=OUT/'runs'/b/m/s
    if not (d/'COMPLETE.json').exists():continue
    series[m]=json.loads((d/'metrics.json').read_text());provenance[m]=json.loads((d/'COMPLETE.json').read_text())
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
 trained=sum((OUT/'runs'/b/m/'TRAINED.json').exists() for b in BACKBONES for m in METHODS)
 index.insert(2,f'Training completed: {trained}/24. Test evaluations completed: {len(all_records)}/48. Not complete until all acceptance checks and remote publication succeed.')
 (OUT/'reports/README.md').write_text('\n'.join(index)+'\n')
if __name__=='__main__':main()
