#!/usr/bin/env python3
"""Native FlashVTG export with original E1 candidate fusion and edited GEQ gate.

The historical exporter scales candidate confidences by the presence probability.
This bridge supplies the original head probability to that exporter, then replaces
only the exported presence value with the edited head probability. Localization and
its legacy post-processing therefore remain identical to the unedited reference.
"""
import argparse,json,sys
from pathlib import Path
import torch
from core import sha256,write_json
p=argparse.ArgumentParser();p.add_argument('--legacy-code',required=True);p.add_argument('--reference',required=True);p.add_argument('--edited',required=True);p.add_argument('--split',choices=('train','val'),required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:1')
a=p.parse_args();sys.path.insert(0,str(Path(a.legacy_code).resolve()))
from common import seed
from adapters import Adapter
from model import JointGMR
from evaluation import predict
seed();torch.set_num_threads(2)
ref=torch.load(a.reference,map_location='cpu',weights_only=False);candidate=torch.load(a.edited,map_location='cpu',weights_only=False)
assert candidate['weight_space_edit']['parent_sha256']==sha256(a.reference)
for k in ref['model']:
 if not k.startswith('geq.'):assert torch.equal(ref['model'][k],candidate['model'][k])
adapter=Adapter('flashvtg',a.device);model=JointGMR(adapter.base,'flashvtg',False).to(a.device);model.load_state_dict(candidate['model'],strict=True);model.eval().requires_grad_(False)
w=ref['model']['geq.global_head.weight'].to(a.device);b=ref['model']['geq.global_head.bias'].to(a.device)
changed=[k for k in ref['model'] if not torch.equal(ref['model'][k],candidate['model'][k])]
if any(not k.startswith('geq.') for k in changed):raise ValueError('Only GEQ edits are allowed')
nonhead=any(k not in ('geq.global_head.weight','geq.global_head.bias') for k in changed)
reference_geq={k[4:]:v.to(a.device) for k,v in ref['model'].items() if k.startswith('geq.')}
edited_geq={k[4:]:v.to(a.device) for k,v in candidate['model'].items() if k.startswith('geq.')}
class FrozenFusion(torch.nn.Module):
 def __init__(self,model):
  super().__init__();self.model=model;self.h=None;self.edited=[];self.handle=model.geq.global_head.register_forward_pre_hook(self.capture)
 def capture(self,module,args):self.h=args[0]
 def forward(self,**inputs):
  if nonhead:
   self.model.geq.load_state_dict(reference_geq,strict=True)
   reference=self.model(**inputs)['pred_exist_logits'].detach()
   self.model.geq.load_state_dict(edited_geq,strict=True)
  output=self.model(**inputs);self.edited.extend(output['pred_exist_logits'].detach().cpu().reshape(-1).tolist())
  fixed=reference.reshape(-1) if nonhead else torch.nn.functional.linear(self.h,w,b).reshape(-1);self.h=None
  return {**output,'pred_exist_logits':fixed,'pred_exist_score':fixed.sigmoid()}
bridge=FrozenFusion(model).to(a.device);data=adapter.dataset(a.split)
with torch.inference_mode():pred,_=predict(adapter,bridge,data)
bridge.handle.remove();assert len(pred)==len(bridge.edited)
for row,logit in zip(pred,bridge.edited):
 row['pred_exist_logit_raw']=logit;row['pred_exist_score']=float(torch.tensor(logit).sigmoid())
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
with (out/'predictions.jsonl').open('x') as f:
 for row in pred:f.write(json.dumps(row)+'\n')
write_json(out/'audit.json',dict(split=a.split,rows=len(pred),reference_sha256=sha256(a.reference),edited_sha256=sha256(a.edited),prediction_sha256=sha256(out/'predictions.jsonl'),candidate_fusion='original E1 GEQ logit',presence='edited GEQ logit',gradient_free=True))
