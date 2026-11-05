#!/usr/bin/env python3
"""Discover FFN columns on Train discovery and test activation suppression on held-out videos."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from core import read_jsonl,sha256,write_json,video_partition,label,distribution
p=argparse.ArgumentParser();p.add_argument('--legacy-code',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--train',required=True);p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--layer',type=int,default=-1);p.add_argument('--columns',type=int,default=16);p.add_argument('--strengths',type=float,nargs='+',default=[.25,.5,1.])
a=p.parse_args();sys.path.insert(0,str(Path(a.legacy_code).resolve()))
from common import seed
from adapters import Adapter
from model import JointGMR
seed();torch.set_num_threads(2)
source_sha=sha256(a.checkpoint);analysis=json.loads((Path(a.analysis)/'analysis.json').read_text());assert analysis['checkpoint_sha256']==source_sha
assert sha256(Path(a.analysis)/'direction.npz')==analysis['direction_sha256']
u=np.load(Path(a.analysis)/'direction.npz')['direction'];rows=read_jsonl(Path(a.train)/'rows.jsonl')
adapter=Adapter('flashvtg',a.device);model=JointGMR(adapter.base,'flashvtg',False).to(a.device);saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
data=adapter.dataset('train');assert len(rows)==len(data)
layer=a.layer%len(model.geq.video_reads);proj=model.geq.video_reads[layer].feedforward[3];W=proj.weight.detach().cpu().numpy();assert W.shape==(256,512)
features=[];indices=[i for i,r in enumerate(rows) if video_partition(r['vid'])=='discovery' and label(r,analysis['tau'],analysis['hard_null_threshold']) in ('B','C')]
def capture(mod,args):features.append(args[0].detach().cpu().numpy().reshape(-1))
hook=proj.register_forward_pre_hook(capture)
with torch.inference_mode():
 for n,i in enumerate(indices):
  meta,inputs,_=adapter.collate([data[i]]);assert meta[0]['qid']==rows[i]['qid'];model(**inputs)
  if n%200==0:print('discovery',n,len(indices),flush=True)
hook.remove();F=np.stack(features);assert F.shape==(len(indices),512)
y=np.array([label(rows[i],analysis['tau'],analysis['hard_null_threshold'])=='B' for i in indices]);assert y.sum()>=2 and (~y).sum()>=2
sensitivity=u@W;importance=np.abs((F[y].mean(0)-F[~y].mean(0))*sensitivity);cols=np.argsort(-importance)[:a.columns].copy();assert len(set(cols.tolist()))==a.columns
unit=torch.as_tensor(u,dtype=proj.weight.dtype,device=a.device);columns=torch.as_tensor(cols,dtype=torch.long,device=a.device)
coeff=unit@proj.weight[:,columns];records=[]
strength=[0.]+a.strengths
current=[0.]
def intervene(module,args,output):
 if current[0]==0:return output
 inp=args[0];component=(inp.index_select(-1,columns)*coeff).sum(dim=-1,keepdim=True)
 return output-current[0]*component*unit
hook=proj.register_forward_hook(intervene)
dev=[i for i,r in enumerate(rows) if video_partition(r['vid'])=='development']
with torch.inference_mode():
 for n,i in enumerate(dev):
  meta,inputs,_=adapter.collate([data[i]]);assert meta[0]['qid']==rows[i]['qid'];values=[]
  for alpha in strength:
   current[0]=alpha;result=model(**inputs);values.append(float(result['pred_exist_logits'].sigmoid().item()))
  records.append(dict(qid=rows[i]['qid'],scores=values))
  if n%200==0:print('development',n,len(dev),flush=True)
hook.remove();scores=np.array([r['scores'] for r in records]);base=scores[:,0]>analysis['tau'];rr=[rows[i] for i in dev];gt=np.array([r['presence'] for r in rr]);B=np.array([label(r,analysis['tau'],analysis['hard_null_threshold'])=='B' for r in rr]);C=np.array([label(r,analysis['tau'],analysis['hard_null_threshold'])=='C' for r in rr]);results=[]
for j,alpha in enumerate(a.strengths,1):
 gate=scores[:,j]>analysis['tau'];rescued=int((B&~base&gate).sum());fp=int((~gt&gate).sum());fn=int((gt&~gate).sum());hard=int((C&gate).sum());ok=rescued>0 and fp<=int((~gt&base).sum()) and fn<=int((gt&~base).sum()) and hard<=int((C&base).sum())
 results.append(dict(strength=alpha,rescued=rescued,fp=fp,fn=fn,hard_null_false_accepts=hard,pass_causal_gate=bool(ok),B_logit_probability_change=distribution((scores[B,j]-scores[B,0]).tolist()),C_logit_probability_change=distribution((scores[C,j]-scores[C,0]).tolist())))
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
write_json(out/'causal.json',dict(source_sha256=source_sha,direction_sha256=analysis['direction_sha256'],layer=layer,columns=cols.tolist(),column_selection='largest absolute B-minus-C activation mean difference times direction projection of output-projection column, Train discovery only',discovery_B=int(y.sum()),discovery_C=int((~y).sum()),development_queries=len(dev),baseline=dict(fp=int((~gt&base).sum()),fn=int((gt&~base).sum()),hard_null_false_accepts=int((C&base).sum())),intervention='subtract strength * u * (u.T @ W[:,columns] @ hidden[columns]) at FFN output; no weights modified',results=results,passed=any(r['pass_causal_gate'] for r in results),independent_development_videos=True,gradient_free=True))
with (out/'causal_scores.jsonl').open('x') as f:
 for row in records:f.write(json.dumps(row)+'\n')
assert sha256(a.checkpoint)==source_sha
print(json.dumps(results),flush=True)
