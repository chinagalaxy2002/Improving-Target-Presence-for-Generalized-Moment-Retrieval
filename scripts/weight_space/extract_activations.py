#!/usr/bin/env python3
"""Extract FlashVTG E1 native predictions and GEQ global-head input, Train/Val only."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from core import sha256, temporal_iou, write_json

p=argparse.ArgumentParser()
p.add_argument('--legacy-code',required=True,help='stage_1_geq/code directory')
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',required=True)
p.add_argument('--split',choices=['train','val'],required=True)
p.add_argument('--device',default='cuda:0')
p.add_argument('--limit',type=int,default=None,help='Smoke tests only')
a=p.parse_args()
output=Path(a.output); output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(Path(a.legacy_code).resolve()))
from common import ROOT, seed
from adapters import Adapter
from model import JointGMR
from evaluation import predict
seed();torch.set_num_threads(2)
before=sha256(a.checkpoint)
adapter=Adapter('flashvtg',a.device)
model=JointGMR(adapter.base,'flashvtg',False).to(a.device)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
assert saved.get('setting')=='E1'
model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
activations=[]
def hook(module,args):
    v=args[0].detach().cpu().numpy()
    assert v.ndim==2 and v.shape[1]==256
    activations.extend(v.copy())
handle=model.geq.global_head.register_forward_pre_hook(hook)
data=adapter.dataset(a.split)
with torch.inference_mode(): predictions,_=predict(adapter,model,data,limit=a.limit)
handle.remove()
assert sha256(a.checkpoint)==before
meta=data.data[:len(predictions)]
assert len(predictions)==len(activations)==len(meta)
rows=[]
for g,pred in zip(meta,predictions):
    assert g['qid']==pred['qid']
    # Official scoring uses the exported, ranked pre-existence candidates.
    windows=pred.get('pred_relevant_windows_pre_exist',pred['pred_relevant_windows'])
    top=windows[0] if windows else None
    rows.append(dict(qid=g['qid'],vid=g['vid'],presence=bool(g['relevant_windows']),
        top1_window=top[:2] if top else None,top1_iou=temporal_iou(top,g['relevant_windows']),
        candidate_score=float(top[2]) if top else 0.,logit=pred['pred_exist_logit_raw'],probability=pred['pred_exist_score']))
np.save(output/'activations.npy',np.stack(activations))
for name,items in [('rows',rows),('predictions',predictions),('ground_truth',meta)]:
    with (output/f'{name}.jsonl').open('x') as f:
        for row in items:f.write(json.dumps(row,allow_nan=False)+'\n')
write_json(output/'audit.json',dict(split=a.split,rows=len(rows),limited=a.limit is not None,
 checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=before,baseline_sha256=adapter.manifest['sha256'],
 label_path=str(ROOT/'data'/f'{a.split}.jsonl'),label_sha256=sha256(ROOT/'data'/f'{a.split}.jsonl'),
 output_sha256={f:sha256(output/f) for f in ('activations.npy','rows.jsonl','predictions.jsonl','ground_truth.jsonl')},
 input_width=256,forward_no_grad=True,legacy_e1=True))
print(json.dumps({'split':a.split,'rows':len(rows),'output':str(output)}))
