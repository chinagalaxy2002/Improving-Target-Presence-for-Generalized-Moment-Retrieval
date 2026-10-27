#!/usr/bin/env python3
"""Durable two-GPU queue: <=2 child jobs per GPU, E1/E2 stage barrier."""
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BACKBONES=['moment_detr','qd_detr','cg_detr','eatr','flashvtg']

def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args()
    if a.status:
        print((ROOT/'QUEUE.json').read_text() if (ROOT/'QUEUE.json').exists() else 'Queue not launched');return
    if a.launch:
        assert not (ROOT/'QUEUE.json').exists(), 'Queue exists; inspect before attempting a new launch'
        log=(ROOT/'logs/queue.log').open('a')
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        print(json.dumps(dict(queue_pid=proc.pid,log=str(ROOT/'logs/queue.log'))));return
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (ROOT/'QUEUE.json').exists(), 'Never overwrite an existing queue record'
    jobs=[]
    for setting in ('E1','E2'):
        for b in BACKBONES:
            pre=ROOT/f'preflights/{b}/{setting}/PREFLIGHT.json'
            assert pre.exists() and json.loads(pre.read_text())['passed'], f'Unverified configuration: {b}/{setting}'
            jobs.append(dict(backbone=b,setting=setting,status='pending',command=[sys.executable,str(ROOT/'run.py'),'--backbone',b,'--setting',setting],preflight=str(pre)))
    state=dict(pid=os.getpid(),started=time.time(),max_tasks_per_gpu=2,gpus=[0,1],phase_barrier='E1 all complete before E2',jobs=jobs,status='running')
    processes={};logs={}
    while True:
        for i,proc in list(processes.items()):
            rc=proc.poll()
            if rc is None:continue
            job=jobs[i];complete=ROOT/f"runs/{job['backbone']}/{job['setting']}/COMPLETE.json"
            job.update(status='complete' if rc==0 and complete.exists() else 'failed',returncode=rc,finished=time.time())
            logs.pop(i).close();del processes[i]
        e1=[j for j in jobs if j['setting']=='E1']
        phase='E2' if all(j['status']=='complete' for j in e1) else 'E1'
        if not any(j['status']=='pending' and j['setting']==phase for j in jobs) and not processes:
            if any(j['status']=='failed' for j in jobs):
                state['status']='failed'
                for j in jobs:
                    if j['status']=='pending':j['status']='blocked_by_E1_failure'
            else:state['status']='complete'
            state['finished']=time.time();dump(ROOT/'QUEUE.json',state)
            # Partial/missing rows stay missing; reporting cannot invent results.
            result=subprocess.run([sys.executable,str(ROOT/'summarize.py')],cwd=ROOT)
            state['report_returncode']=result.returncode;dump(ROOT/'QUEUE.json',state);return
        for i,job in enumerate(jobs):
            if job['status']!='pending' or job['setting']!=phase:continue
            counts={g:sum(jobs[k]['gpu']==g for k in processes) for g in (0,1)}
            available=[g for g in (0,1) if counts[g]<2]
            if not available:break
            gpu=min(available,key=lambda g:(counts[g],g))
            logpath=ROOT/f"logs/{job['backbone']}_{job['setting']}.log"
            logs[i]=logpath.open('x')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            proc=subprocess.Popen(job['command'],env=env,cwd=ROOT,stdout=logs[i],stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            processes[i]=proc;job.update(status='running',gpu=gpu,pid=proc.pid,started=time.time(),log=str(logpath))
            print(json.dumps(job),flush=True)
        assert all(sum(jobs[k]['gpu']==g for k in processes)<=2 for g in (0,1))
        dump(ROOT/'QUEUE.json',state)
        time.sleep(5)

if __name__=='__main__':main()
