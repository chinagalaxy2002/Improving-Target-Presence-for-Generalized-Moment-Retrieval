#!/usr/bin/env python3
"""Bounded subprocess smoke checks; E1 phase finishes before E2 starts."""
import concurrent.futures, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BACKBONES=['moment_detr','qd_detr','cg_detr','eatr','flashvtg']

def main():
    for setting in ('E1','E2'):
        # Each fixed worker owns one slot; queue provides exactly two slots/GPU.
        slots=[(0,['moment_detr','flashvtg']),(0,['qd_detr']),(1,['cg_detr']),(1,['eatr'])]
        def worker(slot):
            gpu,backbones=slot
            for b in backbones:
                log=ROOT/f'logs/verified_preflight_{b}_{setting}.log'
                with log.open('x') as f:
                    rc=subprocess.call([sys.executable,str(ROOT/'run.py'),'--backbone',b,'--setting',setting,'--preflight-only'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT)
                print(json.dumps(dict(backbone=b,setting=setting,returncode=rc,log=str(log))),flush=True)
                if rc:raise RuntimeError(f'Preflight failed: {b}/{setting}: {log}')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(worker,slots):pass
    report={}
    for b in BACKBONES:
        e1=json.loads((ROOT/f'preflights/{b}/E1/CONFIG.json').read_text());e2=json.loads((ROOT/f'preflights/{b}/E2/CONFIG.json').read_text())
        assert e1['shared_initial_tensor_sha256']==e2['shared_initial_tensor_sha256']
        assert e1['initialization']['sha256']==e2['initialization']['sha256']
        report[b]=dict(shared_initial_tensors_equal=True,source_checkpoint_sha256=e1['initialization']['sha256'],batch_size=e1['initialization']['batch_size'])
    path=ROOT/'reports/ALL_PREFLIGHTS.json'
    path.write_text(json.dumps(dict(passed=True,configurations=10,finished=time.time(),initialization_pair_audit=report),indent=2)+'\n')
    print('ALL_TEN_CONFIGURATIONS_VERIFIED',flush=True)

if __name__=='__main__':main()
