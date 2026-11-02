"""Persistent locked per-backbone DAG; all CUDA work shares two slots per GPU."""
import os,sys,fcntl,subprocess,time,signal
from paths import *

def worker_arguments(backbone,directory):
 if (backbone,directory)==('eatr','multi_positive_only'):
  return ['--backbone','eatr','--method','multi','--output-suffix','_positive_only','--auxiliary-query-scope','positive_only_v1']
 return ['--backbone',backbone,'--method',directory]

def compute():
 r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
 uuid={x.split(',')[1].strip():int(x.split(',')[0]) for x in r.stdout.splitlines()}
 r=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
 counts={0:set(),1:set()}
 for x in r.stdout.splitlines():
  if ',' not in x:continue
  u,p=map(str.strip,x.split(','))
  if u in uuid:counts[uuid[u]].add(int(p))
 return counts

class AdoptedProcess:
 def __init__(self,pid,directory,completion):self.pid=pid;self.directory=directory;self.completion=completion
 def poll(self):
  try:
   stat=Path(f'/proc/{self.pid}/stat').read_text().split(') ',1)[1].split()[0]
   if stat!='Z':return None
  except FileNotFoundError:pass
  pause=self.directory/'PAUSED_FOR_RECOVERY.json'
  if pause.exists() and json.loads(pause.read_text()).get('pid')==self.pid:return 0
  return 0 if self.completion.exists() else 1
 def terminate(self):
  try:os.kill(self.pid,signal.SIGTERM)
  except ProcessLookupError:pass

def main():
 lock=(OUT/'logs/dispatcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 active={};failures={};stopping=False
 previous=json.loads((OUT/'logs/QUEUE.json').read_text()) if (OUT/'logs/QUEUE.json').exists() else {}
 failures=dict(previous.get('failures',{}))
 for key,job in previous.get('active',{}).items():
  pid=job['pid'];b,m=key.split('/')[:2]
  try:
   command=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
   expected=[str(KIT/'runner.py').encode()]+[x.encode() for x in worker_arguments(b,m)]
   if not all(x in command for x in expected):continue
  except FileNotFoundError:continue
  directory=OUT/'runs'/b/m;parts=key.split('/');completion=directory/('PROFILE.json' if parts[2]=='profile' else 'TRAINED.json') if parts[2]!='test' else directory/parts[3]/'COMPLETE.json'
  active[key]=(AdoptedProcess(pid,directory,completion),job['gpu'],open(job['log'],'a'))
  print('ADOPT_EXISTING',key,pid,flush=True)
 retry_path=OUT/'logs/RETRY_REQUESTS.json'
 retry=json.loads(retry_path.read_text()) if retry_path.exists() else {}
 for key,reason in retry.items():
  if key in failures:
   print('RETRY_AFTER_VERIFIED_FIX',key,reason,flush=True);failures.pop(key)

 def stop(s,f):
  nonlocal stopping;stopping=True
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 while True:
  for key,(p,gpu,f) in list(active.items()):
   rc=p.poll()
   if rc is not None:
    f.close();del active[key]
    if rc:
     failures[key]=failures.get(key,0)+1
     dump(OUT/'logs'/('failure_'+key.replace('/','_')+'.json'),dict(key=key,returncode=rc,attempts=failures[key],log=f.name,time=time.time()))
  policy_path=OUT/'logs/scheduling_policy.json'
  policy=json.loads(policy_path.read_text()) if policy_path.exists() else {}
  held_tasks=set(policy.get('held_tasks',[]))
  defer_flash=policy.get('flash_last',False) and any(not (OUT/'runs'/b/m/'TRAINED.json').exists() for b in BACKBONES if b!='flashvtg' for m in METHODS)
  candidates=[]
  for b in BACKBONES:
   if b=='flashvtg' and defer_flash:continue
   if not (OUT/'reports/DEBUG24_READY.json').exists():continue
   if not (OUT/'reports'/f'resume_audit_{b}.json').exists():continue
   for m in METHODS:
    if f'{b}/{m}' in held_tasks:continue
    d=OUT/'runs'/b/m;parent=OUT/'runs'/b/('multi' if m.startswith('multi') else 'gmr')
    if not (d/'TRAINED.json').exists():
     if '_geq_' in m and not ((parent/'TRAINED.json').exists() and (OUT/'runs'/b/'gmr/TRAINED.json').exists()):continue
     candidates.append((b,m,'train',None))
    else:
     if not (d/'PROFILE.json').exists():candidates.append((b,m,'profile',None))
     for s in ('test_id','test_ood'):
      if not (d/s/'COMPLETE.json').exists():
       if m.endswith('frozen') and not (parent/s/'COMPLETE.json').exists():continue
       candidates.append((b,m,'test',s))
  extra_enabled=policy.get('train_eatr_multi_positive_only',False)
  extra_done=(OUT/'runs/eatr/multi_positive_only/TRAINED.json').exists()
  if extra_enabled and not extra_done and 'eatr/multi_positive_only' not in held_tasks:
   assert (OUT/'runs/eatr/multi_positive_only/DUAL_VARIANT_AUTHORIZATION.json').exists()
   candidates.append(('eatr','multi_positive_only','train',None))
  # Fill ready GEQ branches before starting another backbone; do not globally barrier stages.
  def dispatch_priority(x):
   recovery=('/'.join((x[0],x[1],x[2],x[3] or ''))) not in retry
   if policy.get('prioritize_training',False):rank=0 if x[2]=='train' and '_geq_' in x[1] else 1 if x[2]=='train' else 2 if x[2]=='profile' else 3
   else:rank=0 if x[2]=='train' and '_geq_' in x[1] else 1 if x[2]=='profile' else 2 if x[2]=='test' else 3
   return recovery,rank,not (OUT/'runs'/x[0]/x[1]/'last.ckpt').exists()
  candidates.sort(key=dispatch_priority)
  counts=compute()
  if not stopping:
   for b,m,mode,split in candidates:
    key='/'.join((b,m,mode,split or ''))
    if failures.get(key,0)>=3 or any(k.startswith(f'{b}/{m}/') for k in active):continue
    occupation={g:len(counts[g]|{p.pid for p,gg,_ in active.values() if gg==g}) for g in (0,1)}
    gpu=min(occupation,key=occupation.get)
    if occupation[gpu]>=2:break
    log=(OUT/'logs'/f'{b}_{m}_{mode}_{split or ""}.log').open('a')
    cmd=[sys.executable,str(KIT/'runner.py'),*worker_arguments(b,m),'--mode',mode]+(['--split',split] if split else [])
    p=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    active[key]=(p,gpu,log);print('DISPATCH',key,gpu,p.pid,flush=True)
  if stopping:
   for p,_,_ in active.values():p.terminate()
   if not active:return
  trained=sum((OUT/'runs'/b/m/'TRAINED.json').is_file() for b in BACKBONES for m in METHODS)
  tested=sum((OUT/'runs'/b/m/s/'COMPLETE.json').is_file() for b in BACKBONES for m in METHODS for s in ('test_id','test_ood'))
  profiled=sum((OUT/'runs'/b/m/'PROFILE.json').exists() for b in BACKBONES for m in METHODS)
  additional_training={'eatr/multi_positive_only':dict(trained=extra_done)} if extra_enabled else {}
  dump(OUT/'logs/QUEUE.json',dict(scheduling_policy=policy,additional_training=additional_training,flash_deferred=defer_flash,profiled=profiled,required_profiles=24,pid=os.getpid(),time=time.time(),trained=trained,required_training=24,tested=tested,required_testing=48,active={k:dict(pid=p.pid,gpu=g,log=f.name) for k,(p,g,f) in active.items()},failures=failures))
  if trained==24 and tested==48 and profiled==24 and (not extra_enabled or extra_done):
   dump(OUT/'logs/GPU_WORK_FINISHED.json',dict(time=time.time(),trained=24,tested=48,additional_training=additional_training));return
  time.sleep(10)
if __name__=='__main__':main()
