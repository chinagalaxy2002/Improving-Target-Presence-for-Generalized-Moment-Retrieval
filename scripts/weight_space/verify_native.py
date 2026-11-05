#!/usr/bin/env python3
"""Re-run every edited head against the identical E1 model inputs, checking native tensors."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from core import read_jsonl,sha256,write_json
p=argparse.ArgumentParser();p.add_argument('--legacy-code',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--edits',required=True);p.add_argument('--val',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--limit',type=int,help='Smoke check only')
a=p.parse_args();sys.path.insert(0,str(Path(a.legacy_code).resolve()))
from common import seed
from adapters import Adapter
from model import JointGMR
seed();torch.set_num_threads(2)
source_sha=sha256(a.checkpoint);index=json.loads((Path(a.edits)/'index.json').read_text());assert index['source_sha256']==source_sha
adapter=Adapter('flashvtg',a.device);model=JointGMR(adapter.base,'flashvtg',False).to(a.device);original=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(original['model'],strict=True);model.eval().requires_grad_(False)
weights={};wk='geq.global_head.weight';bk='geq.global_head.bias';changed_keys=set()
for name in index['candidates']:
 f=Path(a.edits)/name;manifest=json.loads((f/'manifest.json').read_text());assert sha256(f/'edited.ckpt')==manifest['checkpoint_sha256'];state=torch.load(f/'edited.ckpt',map_location='cpu',weights_only=False)['model']
 assert state.keys()==original['model'].keys()
 changed={k for k in state if not torch.equal(state[k],original['model'][k])}
 assert all(k.startswith('geq.') for k in changed),(name,changed)
 weights[name]={k:state[k].to(a.device) for k in changed};changed_keys.update(changed)
 del state
base_values={k:original['model'][k].to(a.device) for k in changed_keys}
live_state=model.state_dict()
def apply(name):
 for k,v in base_values.items():live_state[k].copy_(v)
 for k,v in weights[name].items():live_state[k].copy_(v)
rows=read_jsonl(Path(a.val)/'rows.jsonl');data=adapter.dataset('val');assert len(rows)==len(data)
counts={name:0 for name in weights};max_error={name:0. for name in weights}
def compare(x,y,path='native'):
 if torch.is_tensor(x):
  if not torch.is_tensor(y) or not torch.equal(x,y):raise AssertionError(f'native tensor changed: {path}')
 elif isinstance(x,dict):
  assert x.keys()==y.keys(),path
  for k in x:compare(x[k],y[k],path+'.'+str(k))
 elif isinstance(x,(list,tuple)):
  assert len(x)==len(y),path
  for i,(v,z) in enumerate(zip(x,y)):compare(v,z,path+f'[{i}]')
 else:
  assert x==y,(path,x,y)
with torch.inference_mode():
 for i in range(min(a.limit,len(data)) if a.limit is not None else len(data)):
  meta,inputs,_=adapter.collate([data[i]]);assert meta[0]['qid']==rows[i]['qid']
  apply('unedited')
  baseline=model(**inputs);baseline_logit=baseline['pred_exist_logits'].item()
  for name in weights:
   apply(name)
   result=model(**inputs);compare(baseline['native_output'],result['native_output']);counts[name]+=1
  if i%100==0:print('verified',i,'/',len(data),flush=True)
assert sha256(a.checkpoint)==source_sha
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
write_json(out/'native_invariance.json',{'passed':True,'split':'Standard Val','rows':next(iter(counts.values())),'limited':a.limit is not None,'candidates':counts,'source_sha256':source_sha,'equal_native_output':'all tensor leaves exactly equal for same query and input; all base tensors identical','candidate_export_note':'historical post-processing scales candidate scores by existence, so official evaluation freezes original E1 candidate export','forward_no_grad':True})
print('native invariance passed',len(data),len(weights),flush=True)
