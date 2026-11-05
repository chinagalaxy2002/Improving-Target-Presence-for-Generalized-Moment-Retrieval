#!/usr/bin/env python3
"""Compare actual bridge logits with cached-activation analytic head logits."""
import argparse
from pathlib import Path
import numpy as np
import torch
from core import read_jsonl,write_json,sha256
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--bridge',required=True);p.add_argument('--delta',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.reference);rows=read_jsonl(r/'rows.jsonl');pred=read_jsonl(Path(a.bridge)/'predictions.jsonl');h=np.load(r/'activations.npy');delta=torch.load(a.delta,map_location='cpu',weights_only=True)
assert set(delta)<={'geq.global_head.weight','geq.global_head.bias'} and len(rows)==len(pred)==len(h)
assert all(x['qid']==y['qid'] for x,y in zip(rows,pred))
w=delta.get('geq.global_head.weight',torch.zeros(1,256)).numpy().reshape(-1);b=float(delta.get('geq.global_head.bias',torch.zeros(1)).item())
expected=np.array([x['logit'] for x in rows])+h@w+b;actual=np.array([x['pred_exist_logit_raw'] for x in pred]);error=float(np.max(np.abs(expected-actual)))
expected_prob=1/(1+np.exp(-expected));actual_prob=np.array([x['pred_exist_score'] for x in pred]);changes={str(t):int(((expected_prob>t)!=(actual_prob>t)).sum()) for t in (.4,.5,.6)}
assert error<3e-5 and not any(changes.values()),(error,changes)
write_json(a.output,{'passed':True,'rows':len(rows),'max_logit_absolute_error':error,'threshold_decision_differences':changes,'delta_sha256':sha256(a.delta),'bridge_prediction_sha256':sha256(Path(a.bridge)/'predictions.jsonl')})
print('analytic check',len(rows),error,changes)
