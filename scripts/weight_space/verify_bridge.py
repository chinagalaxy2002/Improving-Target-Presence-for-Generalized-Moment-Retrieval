#!/usr/bin/env python3
"""Verify a fixed-candidate export byte-for-byte at query level and via official mAP."""
import argparse,sys
from pathlib import Path
from core import read_jsonl,write_json,sha256
p=argparse.ArgumentParser();p.add_argument('--legacy-code',required=True);p.add_argument('--reference',required=True,help='Baseline extraction directory');p.add_argument('--bridge',required=True,help='infer_fixed_candidates output');a=p.parse_args()
sys.path.insert(0,str(Path(a.legacy_code).resolve()));from common import OLD
sys.path.insert(0,str(OLD));from official_eval.eval_main import evaluate_gmr
r=Path(a.reference);d=Path(a.bridge);baseline=read_jsonl(r/'predictions.jsonl');edited=read_jsonl(d/'predictions.jsonl');gt=read_jsonl(r/'ground_truth.jsonl');assert len(baseline)==len(edited)==len(gt)
for x,y,z in zip(baseline,edited,gt):
 assert x['qid']==y['qid']==z['qid']
 assert x['pred_relevant_windows']==y['pred_relevant_windows'] and x['pred_relevant_windows_pre_exist']==y['pred_relevant_windows_pre_exist'],x['qid']
x=evaluate_gmr(baseline,gt,map_num_workers=0,verbose=False)['brief']['mAP'];y=evaluate_gmr(edited,gt,map_num_workers=0,verbose=False)['brief']['mAP'];assert x==y
write_json(d/'verification.json',dict(passed=True,queries=len(baseline),candidate_and_order_exact=True,unedited_map=x,edited_map=y,source_prediction_sha256=sha256(r/'predictions.jsonl'),edited_prediction_sha256=sha256(d/'predictions.jsonl')))
print('bridge verified',len(baseline),'mAP',x)
