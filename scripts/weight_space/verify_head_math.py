#!/usr/bin/env python3
"""Check captured-activation analytic logits against actual edited forward passes."""
import argparse,json,sys
from pathlib import Path
import torch
from core import sha256,write_json
p=argparse.ArgumentParser();p.add_argument('--legacy-code',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--edits',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--samples',type=int,default=16)
a=p.parse_args();sys.path.insert(0,str(Path(a.legacy_code).resolve()))
from common import seed
from adapters import Adapter
from model import JointGMR
seed();torch.set_num_threads(2);source=sha256(a.checkpoint);index=json.loads((Path(a.edits)/'index.json').read_text());assert index['source_sha256']==source
adapter=Adapter('flashvtg',a.device);model=JointGMR(adapter.base,'flashvtg',False).to(a.device);state=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model'];model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
w0=state['geq.global_head.weight'].to(a.device);b0=state['geq.global_head.bias'].to(a.device);weights={}
for name in index['candidates']:
 m=json.loads((Path(a.edits)/name/'manifest.json').read_text());assert m['kind']!='ffn';path=Path(a.edits)/name/'delta.pt';assert sha256(path)==m['delta_sha256'];delta=torch.load(path,map_location=a.device,weights_only=True)
 weights[name]=(w0+delta.get('geq.global_head.weight',torch.zeros_like(w0)),b0+delta.get('geq.global_head.bias',torch.zeros_like(b0)))
data=adapter.dataset('val');captured=[]
def hook(module,args):captured.append(args[0].clone())
handle=model.geq.global_head.register_forward_pre_hook(hook);maxerror=0.
with torch.inference_mode():
 for i in range(min(a.samples,len(data))):
  _,inputs,_=adapter.collate([data[i]]);model.geq.global_head.weight.copy_(w0);model.geq.global_head.bias.copy_(b0);out=model(**inputs);h=captured.pop();baseline=out['pred_exist_logits']
  for name,(w,b) in weights.items():
   model.geq.global_head.weight.copy_(w);model.geq.global_head.bias.copy_(b);out=model(**inputs);hh=captured.pop();assert torch.equal(h,hh)
   expected=baseline+torch.nn.functional.linear(h,w-w0,b-b0).reshape(-1)
   error=float((out['pred_exist_logits']-expected).abs().max());maxerror=max(error,maxerror);assert error<3e-5,(name,i,error)
handle.remove();assert sha256(a.checkpoint)==source
out=Path(a.output);out.mkdir(parents=True,exist_ok=False);write_json(out/'head_math.json',{'passed':True,'samples':min(a.samples,len(data)),'candidates':len(weights),'maximum_logit_error':maxerror,'input_activation_identical':True,'gradient_free':True,'source_sha256':source})
print('head math passed',maxerror)
