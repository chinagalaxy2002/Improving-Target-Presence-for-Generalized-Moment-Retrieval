"""Audit fixed checkpoints, recompute Test metrics, report paired results."""
import csv,json
from pathlib import Path
import torch
from soccergmr import checkpoint as ck
from soccergmr.config import load_config,stage_config_identity
from soccergmr.data import SoccerDataset
from soccergmr.backbones.registry import get_spec
from soccergmr.evaluation import evaluate
from soccergmr.evaluation.official.metrics import compute_mAP
from soccergmr.io import dump_json,read_jsonl,sha256,fingerprint
from soccergmr.selection import score
R=Path(__file__).resolve().parents[1];RUN=R/'runs/clip_v2';OUT=R/'results/clip_v2';OUT.mkdir(parents=True,exist_ok=True)
freeze=json.loads((RUN/'FREEZE.json').read_text());assert ck.source_identity()==freeze['source']
for p,h in freeze['files'].items():assert sha256(R/p)==h
frozen_choices=json.loads((RUN/'VAL_FROZEN.json').read_text());rows=[];details={};audit={}
for i in range(1,7):
 n=f'E{i}';c=load_config(R/f'configs/experiments/{n}.json');spec=get_spec('moment_detr');details[n]={}
 assert ck.asset_identity([SoccerDataset(c,s,spec) for s in ['train','val']])==freeze['train_val_assets']
 assert ck.asset_identity([SoccerDataset(c,s,spec) for s in ['test_standard','test_full']])==freeze['test_assets']
 for stage in ['gmr','geq_full']:
  d=RUN/n/stage;m=json.loads((d/'complete.json').read_text());assert m==frozen_choices[n][stage]
  expected=fingerprint(dict(config=stage_config_identity(c,stage),source=freeze['source'],train_val_assets=freeze['train_val_assets'],stage=stage))
  best=ck.load(d/'best.pt',expected,stage);last=ck.load(d/'last.pt',expected,stage);hist=read_jsonl(d/'history.jsonl')
  assert sha256(d/'best.pt')==m['best_sha256'] and last['history']==hist
  assert [h['epoch'] for h in hist]==list(range(1,last['epoch']+1))
  assert max(hist,key=lambda h:h['selection_score'])['epoch']==best['epoch']==m['best_epoch']
  assert score(stage,evaluate(read_jsonl(d/'best_val_predictions.jsonl'),SoccerDataset(c,'val',spec).data))==m['best_score']
  if stage=='geq_full':
   assert not any(k.startswith('clip_head.') for k in best['model'])
   assert len(best['optimizer']['param_groups'])==2 and all(h['clip_loss']==0 for h in hist)
   assert best['baseline_sha256']==sha256(RUN/n/'gmr/best.pt')
  else:assert any(k.startswith('clip_head.') for k in best['model'])==(i!=6)
  patience=c['protocol']['patience' if stage=='gmr' else 'geq_patience']
  assert (m['reason']=='patience' and last['epoch']-best['epoch']==patience) or (m['reason']=='max_epochs' and last['epoch']==200)
  for split,count in [('test_standard',1036),('test_full',2970)]:
   part=RUN/n/(stage+'_test')/split;marker=json.loads((part/'complete.json').read_text());pred=read_jsonl(part/'predictions.jsonl');gt=SoccerDataset(c,split,spec).data
   assert marker['rows']==len(pred)==count
   assert (part/'complete.json').stat().st_mtime_ns>=(RUN/'VAL_FROZEN.json').stat().st_mtime_ns
   for f,k in [('predictions.jsonl','predictions_sha256'),('metrics.json','metrics_sha256')]:assert sha256(part/f)==marker[k]
   computed=evaluate(pred,gt);metrics=json.loads((part/'metrics.json').read_text())
   assert metrics['provenance']['checkpoint_sha256']==m['best_sha256'] and not metrics['provenance']['test_used_for_selection']
   for key in ['brief','gmiou','mAP_detail']:assert computed[key]==metrics[key]
   byid={p['qid']:p for p in pred};subsets={}
   for label,truth in [('short',[g for g in gt if g['relevant_windows'] and max(e-s for s,e in g['relevant_windows'])<=10]),('multi',[g for g in gt if len(g['relevant_windows'])>1])]:subsets[label]=dict(rows=len(truth),metrics=compute_mAP([byid[g['qid']] for g in truth],truth,num_workers=1) if truth else None)
   row=dict(experiment=n,method=c['clip']['method'],stage=stage,split=split,mAP=metrics['brief']['mAP'],G1_mean=sum(metrics['gmiou'][str(t)]['G-mIoU@1'] for t in [.4,.5,.6])/3,G3_mean=sum(metrics['gmiou'][str(t)]['G-mIoU@3'] for t in [.4,.5,.6])/3,AUROC=metrics['brief']['AUROC'],best_epoch=best['epoch'])
   historical_map={'gmr':{'test_standard':6.76,'test_full':6.33},'geq_full':{'test_standard':6.82,'test_full':6.01}}
   row['delta_map_historical_C']=row['mAP']-historical_map[stage][split]
   row['delta_G1_historical_C']=row['G1_mean']-{'test_standard':52.60,'test_full':48.91}[split] if stage=='geq_full' else None
   row['reference_scope']='historical C, different source; not a matched-source control'
   for t in [.4,.5,.6]:row['RejF1_'+str(t)]=metrics['brief']['Rej-F1@'+str(t)]
   rows.append(row);details[n][stage+'/'+split]=dict(metrics=metrics,subsets=subsets);audit[n+'/'+stage+'/'+split]=dict(rows=count,checkpoint_sha256=m['best_sha256'])
 diag=json.loads((RUN/n/'diagnostics/clip_val.json').read_text());assert diag['checkpoint_sha256']==sha256(RUN/n/'gmr/best.pt')
 details[n]['clip_val']=diag
with (OUT/'METRICS.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
dump_json(OUT/'DETAILS.json',details);dump_json(OUT/'AUDIT.json',dict(passed=True,source=freeze['source'],commit=freeze['commit'],test_evaluations=audit,training_stages=12,test_used_for_selection=False))
lines=['# Clip V2: E1–E6','', 'All runs: seed 2023; GMR-only auxiliary loss; unchanged head-free GEQ. All Val selections frozen before any formal Test.', '', '| Method | Split | GMR mAP | GEQ mAP | GEQ G1 | GEQ G3 |','|---|---|---:|---:|---:|---:|']
for n in [f'E{i}' for i in range(1,7)]:
 for split in ['test_standard','test_full']:
  a=next(r for r in rows if r['experiment']==n and r['stage']=='gmr' and r['split']==split);b=next(r for r in rows if r['experiment']==n and r['stage']=='geq_full' and r['split']==split)
  lines.append(f"| {n} | {split} | {a['mAP']:.2f} | {b['mAP']:.2f} | {b['G1_mean']:.2f} | {b['G3_mean']:.2f} |")
lines+=['','Old C is a historical reference (Standard/Full GEQ mAP 6.82/6.01; G1 52.60/48.91). Per user instruction it was not retrained under this source. Cross-source/single-seed differences do not establish robust gains.', '', 'Clip diagnostics, per-tIoU mAP and short/multi-GT subsets are in DETAILS.json. AUPRC and probability gaps are diagnostics, never checkpoint selection criteria. Short means every GT window ≤10 seconds. Full-Decodable covers 2970/2986 queries; Standard Test covers 1036. The two Test sets overlap.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('Final audit PASS: 12 training stages; 24 Test evaluations.')
