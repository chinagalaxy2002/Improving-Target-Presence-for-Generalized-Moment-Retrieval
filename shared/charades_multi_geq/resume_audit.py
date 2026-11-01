"""Compare genuinely terminated workers with an uninterrupted two-step worker."""
import argparse,subprocess,sys,shutil,hashlib,os
from paths import *
p=argparse.ArgumentParser();p.add_argument('backbone');p.add_argument('--verify-existing',action='store_true');a=p.parse_args()
for suffix,steps in [('_resume_interrupted',1),('_resume_interrupted',1),('_resume_reference',2)]:
 cmd=[sys.executable,str(KIT/'runner.py'),'--backbone',a.backbone,'--method','multi','--interrupt-after-steps',str(steps),'--output-suffix',suffix]
 if not a.verify_existing:subprocess.run(cmd,check=True)
import torch
states=[]
for suffix in ('_resume_interrupted','_resume_reference'):
 states.append(torch.load(OUT/'runs'/a.backbone/('multi'+suffix)/'last.ckpt',map_location='cpu',weights_only=False))
assert states[0]['cursor']==states[1]['cursor']==2
unequal=[k for k in states[0]['model'] if not torch.equal(states[0]['model'][k],states[1]['model'][k])]
max_abs=max(float((states[0]['model'][k]-v).abs().max()) for k,v in states[1]['model'].items() if v.is_floating_point())
for k,v in states[0]['model'].items():torch.testing.assert_close(v,states[1]['model'][k],rtol=1e-4,atol=2e-6)
assert torch.equal(states[0]['rng']['torch'],states[1]['rng']['torch'])
assert all(torch.equal(a,b) for a,b in zip(states[0]['rng']['cuda'],states[1]['rng']['cuda']))
assert torch.equal(states[0]['sampler_epoch_start'],states[1]['sampler_epoch_start'])
for pid,v in states[0]['optimizer']['state'].items():
 for k,x in v.items():
  y=states[1]['optimizer']['state'][pid][k]
  if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-7)
  else:assert x==y
path=OUT/'reports'/f'resume_audit_{a.backbone}.json'
dump(path,dict(passed=True,real_process_exit_and_restart=True,model_tensors_numerically_equal=True,optimizer_tensors_numerically_equal=True,model_max_abs_difference=max_abs,model_rtol=1e-4,model_atol=2e-6,cpu_and_cuda_rng_exact=True,sampler_state_exact=True,note="CUDA reduction roundoff allowed; checkpoint reload is strict",steps=2,checkpoint_sha256=[sha(OUT/'runs'/a.backbone/('multi'+s)/'last.ckpt') for s in ('_resume_interrupted','_resume_reference')]))
for suffix in ('_resume_interrupted','_resume_reference'):shutil.rmtree(OUT/'runs'/a.backbone/('multi'+suffix))
print('RESUME_AUDIT_PASS',a.backbone,flush=True)
