#!/usr/bin/env python3
"""Portable package audit. No original absolute paths, models, CUDA or network."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--all-files',action='store_true');args=p.parse_args()
    manifest=read(ROOT/'PACKAGE_SHA256.json');checked=0
    for name,entry in manifest.items():
        if not args.all_files and entry['derived']:continue
        assert sha(ROOT/name)==entry['sha256'],name
        checked+=1
    datasets=read(ROOT/'data/DATASETS.json')
    for name in ('standard_train','standard_val','standard_test','full_decodable','full_original'):
        d=datasets[name];assert sha(ROOT/d['labels'])==d['sha256']
    models=read(ROOT/'data/CHECKPOINTS.json');index={m['id']:m for m in models};assert len(index)==24
    runs=read(ROOT/'data/RUNS.json');assert len(runs)==48
    results={(r['test'],r['backbone'],r['method']):r for r in read(ROOT/'data/RESULTS.json')}
    assert len(results)==48
    for r in runs:
        assert index[r['model_id']]['sha256']==r['checkpoint_sha256']
        path=ROOT/r['predictions'];assert sha(path)==r['predictions_gzip_sha256']
        raw=gzip.decompress(path.read_bytes());assert hashlib.sha256(raw).hexdigest()==r['predictions_uncompressed_sha256']
        preds=[json.loads(l) for l in raw.decode().splitlines() if l.strip()]
        gt=[json.loads(l) for l in (ROOT/r['gt']).read_text().splitlines() if l.strip()]
        idx={d['qid']:d for d in gt}
        assert len(preds)==len({p['qid'] for p in preds})==len(gt)==(1036 if r['test']=='standard' else 2970)
        assert {p['qid'] for p in preds}==set(idx)
        assert all((p['vid'],p['query'])==(idx[p['qid']]['vid'],idx[p['qid']]['query']) for p in preds)
        result=results[(r['test'],r['backbone'],r['method'])]
        assert result['checkpoint_sha256']==r['checkpoint_sha256']
        for k,v in r['expected_published'].items():assert abs(result['values'][k]-v)<1e-9,(r['model_id'],k)
    audit=read(ROOT/'AUDIT.json');assert audit['passed'] and len(audit['frozen_checks'])==16
    assert all(c['candidates_exact'] and c['all_localization_metrics_exact'] for c in audit['frozen_checks'])
    print(json.dumps(dict(passed=True,checked_packaged_files=checked,checkpoints=24,formal_records=48,
                          standard_rows_each=1036,full_decodable_rows_each=2970,original_full_rows=2986,
                          frozen_checks=16,original_local_files_required=False,training=False,model_inference=False)))

if __name__=='__main__':main()
