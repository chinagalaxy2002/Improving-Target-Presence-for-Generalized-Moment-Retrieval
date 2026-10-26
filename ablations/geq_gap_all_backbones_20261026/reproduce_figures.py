#!/usr/bin/env python3
"""Portable figure replay from compact saved actual scores and ground truth."""
import csv,gzip,json,hashlib
from pathlib import Path
from report import figures,EXP

def main():
    inputs=EXP/'evidence/figure_inputs.json.gz'
    with gzip.open(inputs,'rt') as f:bundle=json.load(f)
    records=[]
    for row in csv.DictReader((EXP/'reports/all_metrics.csv').open()):records.append({k:v if k in ['backbone','method','split'] else float(v) for k,v in row.items()})
    preds={}
    for name,values in bundle['scores'].items():
        bb,method,split=name.split('|');gt=bundle['ground_truth'][split];mp={p['qid']:p for p in values};assert len(mp)==len(values)==len(gt) and set(mp)=={g['qid'] for g in gt};preds[bb,method,split]=([mp[g['qid']] for g in gt],gt)
    figures(records,preds)
    print('Reproduced six PDF/SVG/PNG figure sets from actual stored scores; no checkpoint or GPU required.')
if __name__=='__main__':main()
