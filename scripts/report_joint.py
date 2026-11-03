import csv,json
from pathlib import Path
import torch
from soccergmr import checkpoint as ck
from soccergmr.config import load_config,stage_config_identity
from soccergmr.backbones.registry import get_spec
from soccergmr.data import SoccerDataset
from soccergmr.evaluation import evaluate
from soccergmr.evaluation.official.metrics import compute_mAP
from soccergmr.io import dump_json,sha256,read_jsonl,fingerprint
from soccergmr.selection import score
R=Path(__file__).resolve().parents[1];RUN=R/'runs/clip_joint';OLD=Path('/home/guoxiangyu/SoccerGMR-GEQ-clip-v2');OUT=R/'results/clip_joint';OUT.mkdir(parents=True,exist_ok=True)
f=json.loads((RUN/'FREEZE.json').read_text());assert ck.source_identity()==f['source']
for p,h in f['files'].items():assert sha256(R/p)==h
choices=json.loads((RUN/'VAL_FROZEN.json').read_text());rows=[];details={};audits={}
for i in range(1,7):
 n=f'E{i}';d=RUN/n/'geq_full';cfg=load_config(R/f'configs/experiments/{n}.json');spec=get_spec('moment_detr')
 assert ck.asset_identity([SoccerDataset(cfg,s,spec) for s in ['train','val']])==f['train_val_assets']
 assert ck.asset_identity([SoccerDataset(cfg,s,spec) for s in ['test_standard','test_full']])==f['test_assets']
 m=json.loads((d/'complete.json').read_text());assert m==choices[n]['geq_full'] and sha256(d/'best.pt')==m['best_sha256']
 identity=fingerprint(dict(config=stage_config_identity(cfg,'geq_full'),source=f['source'],train_val_assets=f['train_val_assets'],stage='geq_full'))
 best=ck.load(d/'best.pt',identity,'geq_full');last=ck.load(d/'last.pt',identity,'geq_full');hist=read_jsonl(d/'history.jsonl')
 parent=ck.load(RUN/n/'gmr/best.pt');old=ck.load(OLD/f'runs/clip_v2/{n}/gmr/best.pt')
 assert sha256(parent['migration']['original_path'])==parent['migration']['original_sha256']
 assert all(torch.equal(v,parent['model'][k]) for k,v in old['model'].items())
 assert best['baseline_sha256']==sha256(RUN/n/'gmr/best.pt')
 assert last['history']==hist and list(range(1,last['epoch']+1))==[h['epoch'] for h in hist]
 assert max(hist,key=lambda h:h['selection_score'])['epoch']==best['epoch']==m['best_epoch']
 assert all(h['clip_loss']>0 for h in hist) and len(last['optimizer']['param_groups'])==(2 if i==6 else 3)
 if i<6:assert any(not torch.equal(v,last['model'][k]) for k,v in parent['model'].items() if k.startswith('clip_head.'))
 else:assert not any(k.startswith('clip_head.') for k in last['model'])
 assert last['epoch']-best['epoch']==10 or last['epoch']==200
 assert score('geq_full',evaluate(read_jsonl(d/'best_val_predictions.jsonl'),SoccerDataset(cfg,'val',spec).data))==m['best_score']
 details[n]={}
 for split,count in [('test_standard',1036),('test_full',2970)]:
  path=RUN/n/'geq_full_test'/split;marker=json.loads((path/'complete.json').read_text());pred=read_jsonl(path/'predictions.jsonl');gt=SoccerDataset(cfg,split,spec).data
  assert marker['rows']==len(pred)==count and (path/'complete.json').stat().st_mtime_ns>=(RUN/'VAL_FROZEN.json').stat().st_mtime_ns
  for file,key in [('predictions.jsonl','predictions_sha256'),('metrics.json','metrics_sha256')]:assert sha256(path/file)==marker[key]
  metrics=json.loads((path/'metrics.json').read_text());computed=evaluate(pred,gt)
  for key in ['brief','gmiou','mAP_detail']:assert computed[key]==metrics[key]
  assert metrics['provenance']['checkpoint_sha256']==m['best_sha256'] and not metrics['provenance']['test_used_for_selection']
  previous=json.loads((OLD/f'runs/clip_v2/{n}/geq_full_test'/split/'metrics.json').read_text());g1=sum(metrics['gmiou'][str(t)]['G-mIoU@1'] for t in [.4,.5,.6])/3;oldg1=sum(previous['gmiou'][str(t)]['G-mIoU@1'] for t in [.4,.5,.6])/3
  row=dict(experiment=n,split=split,mAP=metrics['brief']['mAP'],G1_mean=g1,G3_mean=sum(metrics['gmiou'][str(t)]['G-mIoU@3'] for t in [.4,.5,.6])/3,AUROC=metrics['brief']['AUROC'],delta_map_previous_E=metrics['brief']['mAP']-previous['brief']['mAP'],delta_G1_previous_E=g1-oldg1,best_epoch=best['epoch'])
  rows.append(row);byid={p['qid']:p for p in pred};subsets={}
  for label,truth in [('short',[g for g in gt if g['relevant_windows'] and max(e-s for s,e in g['relevant_windows'])<=10]),('multi',[g for g in gt if len(g['relevant_windows'])>1])]:subsets[label]=dict(rows=len(truth),mAP=compute_mAP([byid[g['qid']] for g in truth],truth,num_workers=1) if truth else None)
  details[n][split]=dict(metrics=metrics,subsets=subsets);audits[n+'/'+split]=dict(rows=count,checkpoint_sha256=m['best_sha256'])
 diag=json.loads((RUN/n/'diagnostics/clip_val.json').read_text());assert diag['checkpoint_sha256']==m['best_sha256'];details[n]['clip_val']=diag
with (OUT/'METRICS.csv').open('w') as file:
 w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
dump_json(OUT/'DETAILS.json',details);dump_json(OUT/'AUDIT.json',dict(passed=True,source=f['source'],commit=f['commit'],parents_not_retrained=True,training_stages=6,test_evaluations=audits,test_used_for_selection=False))
lines=['# GEQ joint auxiliary finetuning','', 'Same E1–E6 GMR parents, inherited unfrozen heads (E6 native saliency), auxiliary coefficient .1. Compared against previous head-free GEQ. This round intentionally changes GEQ training objective; native GEQ architecture/localization/gating stays unchanged.','', '| Method | Split | mAP | delta previous E | G1 | delta previous E |','|---|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['experiment']} | {r['split']} | {r['mAP']:.2f} | {r['delta_map_previous_E']:+.2f} | {r['G1_mean']:.2f} | {r['delta_G1_previous_E']:+.2f} |")
lines+=['','All Val choices fixed before Test. Single seed 2023. Standard1036 / Full-Decodable2970 of2986. Fixed thresholds .4/.5/.6; no Test tuning. Source/config parent migrations retain original hashes and bit-identical GMR tensors; no old GMR was retrained.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print('PASS: six joint GEQ stages, twelve fixed Test evaluations.')
