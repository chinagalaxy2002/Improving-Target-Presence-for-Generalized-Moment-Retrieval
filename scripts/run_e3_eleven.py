"""Six independent fixed GPU slots; all Val choices finalized before Test."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from soccergmr import checkpoint as ck
from soccergmr.config import load_config
from soccergmr.data import SoccerDataset
from soccergmr.backbones.registry import get_spec
from soccergmr.io import dump_json,sha256
R=Path(__file__).resolve().parents[1];RUN=R/'runs/e3_eleven/formal';NAMES=['ld_detr','taskweave','bm_detr','uvcom','eatr','bam_detr','cg_detr','tr_detr','qd_detr','sim_detr']
GPU0={'ld_detr','bm_detr','eatr','cg_detr','qd_detr'}

def main():
 os.chdir(R);lock=(RUN/'.queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 frozen=json.loads((RUN/'FREEZE.json').read_text());workers={};completed=[];failure=[]
 state=dict(reused=['moment_detr'],status='running',pid=os.getpid(),source=frozen['source'],commit=frozen['commit'],running={},completed=completed,failures=failure,phase='training',max_tasks_per_gpu=3)
 env={**os.environ,'PYTHONPATH':str(R/'src'),'OMP_NUM_THREADS':'2','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
 def gate():
  assert ck.source_identity()==frozen['source'],'Training source changed'
  for p,h in frozen['files'].items():assert sha256(R/p)==h,'Frozen file changed: '+p
 def cfg(n):return R/'configs/experiments'/f'{n}.json'
 def complete(n,k):
  d=RUN/n/k
  if k in ('gmr','geq_full'):
   if not (d/'complete.json').exists():return False
   m=json.loads((d/'complete.json').read_text());assert m['source']==frozen['source'] and sha256(d/'best.pt')==m['best_sha256']
   if k=='geq_full':assert m['baseline_sha256']==sha256(RUN/n/'gmr/best.pt')
   return True
  if k=='diagnostics':return False # repeat read-only diagnostics on restart
  for split,count in [('test_standard',1036),('test_full',2970)]:
   part=d/split
   if not (part/'complete.json').exists():return False
   m=json.loads((part/'complete.json').read_text());assert m['rows']==count
   assert all(sha256(part/f)==m[key] for f,key in [('predictions.jsonl','predictions_sha256'),('metrics.json','metrics_sha256')])
  return True
 def capacity():
  info=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
  gpus={int(i):(u.strip(),int(m)) for line in info.splitlines() for i,u,m in [line.split(',')]};counts={g:set() for g in gpus};uu={v[0]:g for g,v in gpus.items()}
  apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
  for line in apps.splitlines():
   if line.strip():u,pid=line.split(',');counts[uu[u.strip()]].add(int(pid))
  for proc,gpu,log in workers.values():counts[gpu].add(proc.pid)
  state['resources']={str(g):dict(pids=sorted(counts[g]),free_mib=gpus[g][1]) for g in gpus}
  return {g:max(0,3-len(counts[g])) if gpus[g][1]>=6000 else 0 for g in gpus}
 def publish():
  state['running']={n+'/'+k:dict(pid=p.pid,gpu=g) for (n,k),(p,g,l) in workers.items()};dump_json(RUN/'queue_state.json',state)
 def group(phase,jobs,dependencies):
  state['phase']=phase;done=set();pending=[]
  for job in jobs:
   if complete(*job):done.add(job);completed.append('/'.join(job))
   else:pending.append(job)
  publish()
  while pending or workers:
   for job,(p,g,l) in list(workers.items()):
    code=p.poll()
    if code is None:continue
    l.close();del workers[job]
    if code or (job[1]!='diagnostics' and not complete(*job)):failure.append(dict(job=job,exit_code=code))
    else:done.add(job);completed.append('/'.join(job));print('DONE',job,flush=True)
   gate()
   if failure:
    publish()
    if not workers:raise RuntimeError('Workers failed: '+repr(failure))
    time.sleep(2);continue
   slots=capacity()
   for n,k in list(pending):
    gpu=0 if n in GPU0 else 1
    if not slots[gpu] or not all(d in done for d in dependencies((n,k))):continue
    c=load_config(cfg(n));splits=['test_standard','test_full'] if k.endswith('_test') else ['train','val'];expected=frozen['test_assets'] if k.endswith('_test') else frozen['train_val_assets']
    assert ck.asset_identity([SoccerDataset(c,s,get_spec(c['backbone'])) for s in splits])==expected,'Assets changed'
    slots=capacity()
    if not slots[gpu]:continue
    if k in ('gmr','geq_full'):
     cmd=[sys.executable,'-m','soccergmr','train','--config',str(cfg(n)),'--stage',k,'--output',str(RUN/n/k)]
     if k=='geq_full':cmd+=['--baseline',str(RUN/n/'gmr/best.pt')]
     if (RUN/n/k/'last.pt').exists():cmd+=['--resume']
    elif k=='diagnostics':cmd=[sys.executable,str(R/'scripts/diagnose_e3_eleven.py'),'--config',str(cfg(n)),'--root',str(RUN/n)]
    else:cmd=[sys.executable,'-m','soccergmr','test','--config',str(cfg(n)),'--checkpoint',str(RUN/n/k[:-5]/'best.pt'),'--output',str(RUN/n/k)]
    log=(RUN/'logs'/f'{n}_{k}.log').open('a');log.write('\nCOMMAND '+repr(cmd)+'\n');log.flush()
    p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env={**env,'CUDA_VISIBLE_DEVICES':str(gpu)},pass_fds=(lock.fileno(),));workers[n,k]=(p,gpu,log);pending.remove((n,k));slots[gpu]-=1
    print('START',n,k,p.pid,'GPU',gpu,flush=True)
   publish()
   if pending or workers:time.sleep(2)
 try:
  gate()
  # Prioritize each method's GEQ as soon as its GMR finishes.
  jobs=[(n,k) for n in NAMES for k in ('gmr','geq_full')]
  group('training',jobs,lambda j:[] if j[1]=='gmr' else [(j[0],'gmr')])
  choices={}
  for n in NAMES:
   choices[n]={k:json.loads((RUN/n/k/'complete.json').read_text()) for k in ('gmr','geq_full')}
  choices['moment_detr']=json.loads((RUN/'moment_detr/REUSE.json').read_text())
  barrier=RUN/'VAL_FROZEN.json'
  if barrier.exists():assert json.loads(barrier.read_text())==choices,'Frozen Val choices changed'
  else:dump_json(barrier,choices)
  group('fixed_evaluation',[(n,k) for n in NAMES for k in ('diagnostics','gmr_test','geq_full_test')],lambda j:[])
  with (RUN/'report.log').open('a') as log:
   code=subprocess.call([sys.executable,str(R/'scripts/report_e3_eleven.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
  if code:raise RuntimeError('Report/audit failed; see report.log')
  state.update(status='complete',test_used_for_selection=False);publish()
 except BaseException as e:
  state.update(status='failed',error=repr(e));publish();raise
if __name__=='__main__':main()
