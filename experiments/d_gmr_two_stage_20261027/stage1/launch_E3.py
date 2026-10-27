#!/usr/bin/env python3
"""Independent five-job E3 queue, at most two child tasks on each GPU."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BACKBONES=['moment_detr','qd_detr','cg_detr','eatr','flashvtg']

def dump(p,x):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args()
    state_path=ROOT/'QUEUE_E3.json'
    if a.status:print(state_path.read_text() if state_path.exists() else 'E3 queue not launched');return
    if a.launch:
        assert not state_path.exists()
        with (ROOT/'logs/queue_E3.log').open('x') as f:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print('E3_QUEUE_PID',process.pid);return
    lock=(ROOT/'queue_E3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not state_path.exists()
    jobs=[]
    for b in BACKBONES:
        gate=json.loads((ROOT/f'preflights/{b}/E3/PREFLIGHT.json').read_text());assert gate['passed']
        jobs.append(dict(backbone=b,setting='E3',status='pending'))
    state=dict(pid=os.getpid(),status='running',started=time.time(),max_tasks_per_gpu=2,jobs=jobs);processes={};logs={}
    while True:
        for i,process in list(processes.items()):
            rc=process.poll()
            if rc is None:continue
            jobs[i].update(status='complete' if rc==0 and (ROOT/f"runs/{jobs[i]['backbone']}/E3/COMPLETE.json").exists() else 'failed',returncode=rc,finished=time.time())
            logs.pop(i).close();del processes[i]
        for i,j in enumerate(jobs):
            if j['status']!='pending':continue
            counts={g:sum(jobs[k]['gpu']==g for k in processes) for g in (0,1)}
            available=[g for g in (0,1) if counts[g]<2]
            if not available:break
            g=min(available,key=lambda g:(counts[g],g));logs[i]=(ROOT/f"logs/{j['backbone']}_E3.log").open('x')
            cmd=[sys.executable,str(ROOT/'run_E3.py'),'--backbone',j['backbone']]
            process=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),PYTHONDONTWRITEBYTECODE='1'),stdout=logs[i],stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            processes[i]=process;j.update(status='running',gpu=g,pid=process.pid,started=time.time(),command=cmd);print(json.dumps(j),flush=True)
        assert all(sum(jobs[i]['gpu']==g for i in processes)<=2 for g in (0,1))
        if not processes and all(j['status']!='pending' for j in jobs):
            state.update(status='complete' if all(j['status']=='complete' for j in jobs) else 'failed',finished=time.time());dump(state_path,state)
            rc=subprocess.call([sys.executable,str(ROOT/'report_E3.py')],cwd=ROOT)
            state['report_returncode']=rc;dump(state_path,state);return
        dump(state_path,state);time.sleep(5)

if __name__=='__main__':main()
