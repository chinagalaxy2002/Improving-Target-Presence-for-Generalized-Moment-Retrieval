"""Independently audit 11-backbone fixed Tests and publish 101-point curves."""
import csv,json,gzip,shutil
from pathlib import Path
import numpy as np
import torch
from soccergmr import checkpoint as ck
from soccergmr.config import load_config,stage_config_identity
from soccergmr.data import SoccerDataset
from soccergmr.backbones.registry import get_spec
from soccergmr.io import dump_json,read_jsonl,sha256,fingerprint
from soccergmr.evaluation import evaluate
from soccergmr.evaluation.official.metrics import _clean_pred_windows,_compute_set_iou_score,prepare_submission_for_gmiou,compute_G_mIoU,compute_mAP
from soccergmr.selection import score
R=Path(__file__).resolve().parents[1];RUN=R/'runs/e3_eleven/formal';OUT=R/'results/e3_eleven';OUT.mkdir(parents=True,exist_ok=True)
NAMES=['moment_detr','eatr','qd_detr','cg_detr','sim_detr','bm_detr','bam_detr','uvcom','tr_detr','taskweave','ld_detr']
f=json.loads((RUN/'FREEZE.json').read_text());assert ck.source_identity()==f['source']
for p,h in f['files'].items():assert sha256(R/p)==h,p
choices=json.loads((RUN/'VAL_FROZEN.json').read_text());reuse=json.loads((RUN/'moment_detr/REUSE.json').read_text());summary=[];curves=[];details={};audits=[];stage_records={}
for name in NAMES:
 config=load_config(R/f'configs/experiments/{name}.json');spec=get_spec(name);details[name]={}
 assert ck.asset_identity([SoccerDataset(config,s,spec) for s in ['train','val']])==f['train_val_assets']
 assert ck.asset_identity([SoccerDataset(config,s,spec) for s in ['test_standard','test_full']])==f['test_assets']
 for stage in ['gmr','geq_full']:
  d=RUN/name/stage;m=json.loads((d/'complete.json').read_text());checkpoint=Path(reuse['stages'][stage]['checkpoint_path']) if name=='moment_detr' else d/'best.pt'
  assert sha256(checkpoint)==m['best_sha256'];best=ck.load(checkpoint,stage=stage)
  if name!='moment_detr':
   expected=fingerprint(dict(config=stage_config_identity(config,stage),source=f['source'],train_val_assets=f['train_val_assets'],stage=stage));assert best['identity']==m['identity']==expected
   assert m==choices[name][stage] and m['source']==f['source']
   last=ck.load(d/'last.pt',expected,stage);hist=read_jsonl(d/'history.jsonl');assert last['history']==hist
   assert [x['epoch'] for x in hist]==list(range(1,last['epoch']+1));assert max(hist,key=lambda x:x['selection_score'])['epoch']==m['best_epoch']==best['epoch']
   assert last['epoch']-best['epoch']==(20 if stage=='gmr' else 10) or last['epoch']==200
   assert all(x['clip_loss']>0 for x in hist)
   assert score(stage,evaluate(read_jsonl(d/'best_val_predictions.jsonl'),SoccerDataset(config,'val',spec).data))==m['best_score']
   if stage=='geq_full':
    initial=ck.load(RUN/name/'gmr/best.pt');assert best['baseline_sha256']==sha256(RUN/name/'gmr/best.pt')
    assert len(last['optimizer']['param_groups'])==3
    assert any(not torch.equal(v,last['model'][k]) for k,v in initial['model'].items() if k.startswith('clip_head.'))
  else:assert best['source']==reuse['stages'][stage]['training_source'] and not reuse['retrained']
  stage_records[name+'/'+stage]=dict(source=m['source'],best_sha256=m['best_sha256'],best_epoch=m['best_epoch'],reused=name=='moment_detr')
  for split,count in [('test_standard',1036),('test_full',2970)]:
   p=RUN/name/(stage+'_test')/split;marker=json.loads((p/'complete.json').read_text());pred=read_jsonl(p/'predictions.jsonl');metrics=json.loads((p/'metrics.json').read_text());gt=SoccerDataset(config,split,spec).data
   assert marker['rows']==len(pred)==count and metrics['provenance']['checkpoint_sha256']==m['best_sha256']
   if name!='moment_detr':assert (p/'complete.json').stat().st_mtime_ns>=(RUN/'VAL_FROZEN.json').stat().st_mtime_ns
   for file,key in [('predictions.jsonl','predictions_sha256'),('metrics.json','metrics_sha256')]:assert sha256(p/file)==marker[key]
   recomputed=evaluate(pred,gt)
   for key in ['brief','gmiou','mAP_detail']:assert recomputed[key]==metrics[key]
   byid={x['qid']:x for x in pred};assert len(byid)==count;pred=[byid[g['qid']] for g in gt];scores=np.array([p['pred_exist_score'] for p in pred]);neg=np.array([_compute_set_iou_score([],g['relevant_windows']) for g in gt]);ts=[i/100 for i in range(101)];ys={}
   for k in [1,3]:
    pos=np.array([_compute_set_iou_score([w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'],max_pred_windows=10)[:k]],g['relevant_windows']) for p,g in zip(pred,gt)])
    ys[str(k)]=[round(float(100*np.where(scores>t,pos,neg).mean()),2) for t in ts]
   for t in [0.,.4,.5,.6,1.]:
    gated,_=prepare_submission_for_gmiou(pred,cls_threshold=t,max_pred_windows=10);official=compute_G_mIoU(gated,gt,k_list=(1,3))
    for k in [1,3]:assert official[f'G-mIoU@{k}']==ys[str(k)][int(round(t*100))]
   row=dict(backbone=name,stage=stage,split=split,mAP=metrics['brief']['mAP'],G1_mean=sum(ys['1'][j] for j in [40,50,60])/3,G3_mean=sum(ys['3'][j] for j in [40,50,60])/3,AUROC=metrics['brief']['AUROC'],best_epoch=best['epoch'],reused=name=='moment_detr')
   for t in [.4,.5,.6]:row['RejF1_'+str(t)]=metrics['brief']['Rej-F1@'+str(t)]
   summary.append(row);curves.extend(dict(backbone=name,stage=stage,split=split,threshold=t,G1=ys['1'][i],G3=ys['3'][i]) for i,t in enumerate(ts))
   subsets={}
   for label,truth in [('short',[g for g in gt if g['relevant_windows'] and max(e-s for s,e in g['relevant_windows'])<=10]),('multi',[g for g in gt if len(g['relevant_windows'])>1])]:subsets[label]=dict(rows=len(truth),mAP=compute_mAP([byid[g['qid']] for g in truth],truth,num_workers=1) if truth else None)
   details[name][stage+'/'+split]=dict(metrics=metrics,subsets=subsets,curve=ys)
   dst=OUT/'predictions'/name/stage/split;dst.mkdir(parents=True,exist_ok=True)
   with gzip.GzipFile(filename=str(dst/'predictions.jsonl.gz'),mode='wb',mtime=0) as h:h.write((p/'predictions.jsonl').read_bytes())
   for file in ['metrics.json','complete.json']:shutil.copy2(p/file,dst/file)
   audits.append(dict(backbone=name,stage=stage,split=split,rows=count,checkpoint_sha256=m['best_sha256'],metrics_recomputed=True,sweep_verified=True))
   print('VERIFIED',name,stage,split,flush=True)
 diag=RUN/name/'diagnostics/clip_val.json'
 if diag.exists():details[name]['clip_val']=json.loads(diag.read_text())
for row in summary:
 base=next(x for x in summary if x['backbone']==row['backbone'] and x['stage']=='gmr' and x['split']==row['split'])
 for metric in ['mAP','G1_mean','G3_mean']:row['delta_'+metric+'_vs_aux_gmr']=row[metric]-base[metric]
# Historical branch baselines remain explicitly separate from common-budget runs.
historical=list(csv.DictReader((R/'results/test_thresholds_11_backbones/METRICS.csv').open()))
references=[]
for row in summary:
 if row['stage']!='geq_full':continue
 split='standard' if row['split']=='test_standard' else 'full'
 old=next(x for x in historical if x['backbone']==row['backbone'] and x['split']==split and x['method']=='gmr')
 references.append(dict(backbone=row['backbone'],split=row['split'],historical_protocol_group=old['protocol_group'],historical_max_video_tokens=old['max_video_tokens'],historical_GMR_mAP=float(old['mAP']),historical_GMR_G1=100*float(old['G1_mean']),new_E3_GEQ_mAP=row['mAP'],new_E3_GEQ_G1=row['G1_mean'],delta_map_descriptive=row['mAP']-float(old['mAP']),delta_G1_descriptive=row['G1_mean']-100*float(old['G1_mean']),scope='Historical cross-protocol reference, not matched-source training control'))
with (OUT/'HISTORICAL_BASELINE_REFERENCE.csv').open('w') as h:
 w=csv.DictWriter(h,fieldnames=list(references[0]));w.writeheader();w.writerows(references)
for file,rows in [('METRICS.csv',summary),('CURVES.csv',curves)]:
 with (OUT/file).open('w') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
dump_json(OUT/'DETAILS.json',details);dump_json(OUT/'AUDIT.json',dict(passed=True,source=f['source'],commit=f['commit'],stages=stage_records,evaluations=audits,new_training_stages=20,new_test_evaluations=40,reused_moment_evaluations=4,test_used_for_selection=False))
lines=['# 11-backbone E3 GMR → GEQ+aux','', 'Shared protocol: Standard Train/Val, seed2023,128 video tokens, GMR Val mAP selection patience20; GEQ Val mean G1(.4/.5/.6) selection patience10; both max200. Balanced BCE+.2 Dice auxiliary coefficient.1 in BOTH stages, inherited unfrozen query-aware head. Native per-backbone losses preserved.', '', '| Backbone | Split | Aux GMR mAP | GEQ+aux mAP | GEQ+aux G1 | GEQ+aux G3 |','|---|---|---:|---:|---:|---:|']
for name in NAMES:
 for split in ['test_standard','test_full']:
  a=next(x for x in summary if x['backbone']==name and x['split']==split and x['stage']=='gmr');b=next(x for x in summary if x['backbone']==name and x['split']==split and x['stage']=='geq_full')
  lines.append(f"| {name} | {split} | {a['mAP']:.2f} | {b['mAP']:.2f} | {b['G1_mean']:.2f} | {b['G3_mean']:.2f} |")
lines+=['','Moment-DETR E3 was already complete and is reused with its ORIGINAL source/hash provenance; not retrained under the new source. Ten other families trained under the new frozen source. Comparisons against GMR in METRICS.csv use each method\'s AUXILIARY-trained Stage1, not the original no-aux baseline. Historical no-aux baseline11 results stay separate under results/test_thresholds_11_backbones and retain old75-token/patience6 versus newer128-token/patience10 differences; do not interpret them as matched-budget controls.', '', 'Standard Test1036, Full-Decodable2970/2986. All new Val models fixed before any new formal Test. Fixed thresholds for selection; full101-point sweeps descriptive only. Positive mAP ungated. Short means all GT windows<=10s. Both Test sets overlap; single-seed observations are not proof of robust causal gains.']
lines+=['', 'HISTORICAL_BASELINE_REFERENCE.csv records descriptive increments against the no-aux GMR baseline from the specified 11-backbone branch, including its original protocol group/token limit. These baselines are not retrained or silently replaced.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('PASS: 11 methods,22 stage results,44 Test evaluations (Moment reused).')
