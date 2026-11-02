#!/usr/bin/env python3
"""Locked evaluation queue. Subprocesses are local workers, not AI agents."""
import fcntl
import os
import subprocess
import sys
import time
from common import *

def event(**data):
    data['time']=time.strftime('%Y-%m-%d %H:%M:%S')
    with (HERE/'queue_events.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
    print(json.dumps(data),flush=True)

def gpu_jobs(active):
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines()
    mapping={uuid.strip():int(i) for i,uuid in (l.split(',') for l in lines)}
    jobs={i:set() for i in mapping.values()}
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()
    for l in lines:
        if not l.strip():continue
        uuid,pid=l.split(',');jobs[mapping[uuid.strip()]].add(int(pid))
    # Reserve slots for queued workers that have not initialized CUDA yet.
    for p,gpu,_,_ in active: jobs[gpu].add(p.pid)
    return jobs

def stage(indices,phase,parallel):
    pending=list(indices);active=[];failed=[]
    while pending or active:
        for item in list(active):
            p,gpu,index,stream=item
            if p.poll() is not None:
                stream.close();active.remove(item)
                event(action='finished',phase=phase,index=index,gpu=gpu,pid=p.pid,returncode=p.returncode)
                if p.returncode: failed.append(index)
        if failed:
            pending=[]
            if not active: raise RuntimeError(f'{phase} failed: {failed}; inspect logs; no completion claimed')
        if pending and len(active)<parallel:
            jobs=gpu_jobs(active)
            if sum(map(len,jobs.values()))<4:
                choices=[i for i in jobs if len(jobs[i])<2]
                if choices:
                    gpu=min(choices,key=lambda i:(len(jobs[i]),i));index=pending.pop(0)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
                    entry=CHECKPOINTS['checkpoints'][index]
                    path=HERE/'logs'/f'{phase}_{entry["backbone"]}_{entry["method"]}.log';path.parent.mkdir(exist_ok=True)
                    stream=path.open('a')
                    p=subprocess.Popen([sys.executable,'-B',str(HERE/'eval_only.py'),'--index',str(index),'--phase',phase],env=env,cwd=HERE,stdout=stream,stderr=subprocess.STDOUT)
                    active.append((p,gpu,index,stream))
                    event(action='launched',phase=phase,index=index,gpu=gpu,pid=p.pid,all_gpu_jobs={str(k):sorted(v) for k,v in jobs.items()})
        dump(HERE/'QUEUE_STATUS.json',dict(phase=phase,pending=pending,active=[dict(pid=p.pid,gpu=g,index=i) for p,g,i,_ in active],failed=failed))
        if pending or active: time.sleep(2)

def main():
    with (HERE/'TASK.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from audit_inputs import audit
        audit()
        entries=CHECKPOINTS['checkpoints']
        stage(range(24),'preflight',1)
        event(action='ALL_24_STRICT_LOAD_AND_INFERENCE_PREFLIGHTS_PASSED')
        parents=[i for i,e in enumerate(entries) if e['method'] in ('gmr','multi')]
        stage(parents,'formal',4)
        event(action='ALL_8_FULL_TEST_PARENTS_COMPLETED')
        stage([i for i in range(24) if i not in parents],'formal',4)
        subprocess.run([sys.executable,'-B',str(HERE/'build_report.py')],check=True,cwd=HERE)
        event(action='ALL_24_FORMAL_FULL_DECODABLE_TESTS_COMPLETED_AND_AUDITED')

if __name__=='__main__':main()
