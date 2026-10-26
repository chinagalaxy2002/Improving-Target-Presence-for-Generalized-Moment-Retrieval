#!/usr/bin/env python3
"""Six background workers, three/GPU; preflight then fresh training, Val-only selection."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
EXP=Path(__file__).resolve().parent

def dump(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--worker');args=p.parse_args()
    if args.launch:
        assert not (EXP/'JOB_DRIVER.json').exists()
        with (EXP/'logs/driver.log').open('x') as f:proc=subprocess.Popen([sys.executable,str(EXP/'launch.py')],cwd=EXP,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        dump(EXP/'JOB_DRIVER.json',dict(pid=proc.pid,max_tasks_per_gpu=3,total_trials=6));print(proc.pid);return
    if args.worker:
        for flag in [['--preflight-only'],[]]:subprocess.run([sys.executable,str(EXP/'search_train.py'),'--trial',args.worker]+flag,cwd=EXP,check=True)
        return
    plan=json.loads((EXP/'SEARCH.json').read_text());uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines();busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
    assert len(uuids)>=2 and all(not any(u.strip() in line for line in busy) for u in uuids[:2]),'Wait for empty GPUs before reserving three workers each'
    workers=[]
    for i,trial in enumerate(plan['trials']):
        gpu=i%2
        with (EXP/f'logs/{trial}.log').open('x') as f:proc=subprocess.Popen([sys.executable,str(EXP/'launch.py'),'--worker',trial],cwd=EXP,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        workers.append((trial,gpu,proc))
    while True:
        state=[dict(trial=n,gpu=g,pid=p.pid,status='running' if p.poll() is None else 'val_complete' if p.returncode==0 else 'failed',exit_code=p.returncode) for n,g,p in workers]
        dump(EXP/'QUEUE.json',dict(max_tasks_per_gpu=3,entries=state));dump(EXP/'STATUS.json',dict(status='running',workers=state,automatic_val_selection_then_test=True))
        if all(p.poll() is not None for _,_,p in workers):break
        time.sleep(10)
    if any(p.returncode for _,_,p in workers):dump(EXP/'STATUS.json',dict(status='failed',workers=state));raise RuntimeError('Worker failed; inspect individual logs. No selection performed.')
    dump(EXP/'STATUS.json',dict(status='selecting_on_val_then_test'))
    subprocess.run([sys.executable,str(EXP/'report.py')],cwd=EXP,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0'),check=True)
if __name__=='__main__':main()
