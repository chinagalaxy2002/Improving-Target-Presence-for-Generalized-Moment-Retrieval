#!/usr/bin/env python3
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def dump(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--launch',action='store_true');ap.add_argument('--worker');args=ap.parse_args()
 if args.launch:
  assert not (ROOT/'JOB_DRIVER.json').exists()
  with (ROOT/'logs/driver.log').open('x') as f:p=subprocess.Popen([sys.executable,str(ROOT/'launch.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
  dump(ROOT/'JOB_DRIVER.json',dict(pid=p.pid,new_training=4,max_tasks_per_gpu=2,automatic_val_test_report_figures=True));print(p.pid);return
 if args.worker:
  for flag in [['--preflight-only'],[]]:subprocess.run([sys.executable,str(ROOT/'run.py'),'--backbone',args.worker]+flag,check=True,cwd=ROOT)
  return
 uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines();busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines();assert len(uuids)>=2 and not any(u.strip() in l for u in uuids[:2] for l in busy),'Expected empty GPUs before reserving two tasks each'
 workers=[]
 for i,bb in enumerate(['moment_detr','qd_detr','cg_detr','eatr']):
  with (ROOT/f'logs/{bb}.log').open('x') as f:p=subprocess.Popen([sys.executable,str(ROOT/'launch.py'),'--worker',bb],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i%2),PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT)
  workers.append((bb,i%2,p))
 while True:
  entries=[dict(backbone=b,gpu=g,pid=p.pid,status='running' if p.poll() is None else 'complete' if p.returncode==0 else 'failed',exit_code=p.returncode) for b,g,p in workers];dump(ROOT/'QUEUE.json',dict(max_tasks_per_gpu=2,entries=entries));dump(ROOT/'STATUS.json',dict(status='running',entries=entries))
  if all(p.poll() is not None for _,_,p in workers):break
  time.sleep(10)
 if any(p.returncode for _,_,p in workers):dump(ROOT/'STATUS.json',dict(status='failed',entries=entries));raise RuntimeError('Failed worker; see preserved logs')
 dump(ROOT/'STATUS.json',dict(status='reporting'))
 try:subprocess.run([sys.executable,str(ROOT/'report.py')],cwd=ROOT,check=True)
 except Exception:dump(ROOT/'STATUS.json',dict(status='report_failed'));raise
if __name__=='__main__':main()
