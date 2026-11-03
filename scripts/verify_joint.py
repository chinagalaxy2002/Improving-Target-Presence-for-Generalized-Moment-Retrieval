import copy,json,os,signal
from pathlib import Path
import torch
from soccergmr import checkpoint as ck,train as training
from soccergmr.config import load_config
from soccergmr.io import dump_json
from soccergmr.test import test
from migrate_joint import migrate
R=Path(__file__).resolve().parents[1];OUT=R/'runs/clip_joint/acceptance';OLD=Path('/home/guoxiangyu/SoccerGMR-GEQ-clip-v2/runs/clip_v2/acceptance')
def main():
 proof={}
 for i in range(1,7):
  n=f'E{i}';raw=json.loads((OLD/(n+'.json')).read_text());raw['clip']['geq_weight']=.1;path=OUT/(n+'.json');dump_json(path,raw);c=load_config(path)
  parent=migrate(OLD/n/'gmr/best.pt',c,OUT/n/'gmr');initial=ck.load(parent)['model']
  oldsave=ck.save
  def save(p,payload):
   if payload['stage']=='geq_full':
    assert len(payload['optimizer']['param_groups'])==(2 if i==6 else 3)
    if payload['epoch']==0:
     assert all(torch.equal(v,payload['model'][('base.'+k) if i==6 else k].cpu()) for k,v in initial.items())
   return oldsave(p,payload)
  ck.save=save;training.train(c,OUT/n/'geq_full','geq_full','cuda:0',parent);ck.save=oldsave
  last=ck.load(OUT/n/'geq_full/last.pt');assert all(h['clip_loss']>0 for h in last['history'])
  if i<6:
   assert any(not torch.equal(v,last['model'][k]) for k,v in initial.items() if k.startswith('clip_head.'))
  else:assert not any(k.startswith('clip_head.') for k in last['model']) and not torch.equal(initial['saliency_proj.weight'],last['model']['base.saliency_proj.weight'])
  if i in [1,6]:
   original=training.evaluate
   def stop(*a,**kw):
    result=original(*a,**kw);os.kill(os.getpid(),signal.SIGTERM);return result
   training.evaluate=stop;res=training.train(c,OUT/n/'resumed','geq_full','cuda:0',parent);training.evaluate=original;assert res['completed_epoch']==1
   training.train(c,OUT/n/'resumed','geq_full','cuda:0',parent,resume=True);other=ck.load(OUT/n/'resumed/last.pt')
   assert all(torch.equal(v,other['model'][k]) for k,v in last['model'].items()) and last['selector']==other['selector']
   for k,v in last['optimizer']['state'].items():
    for key,t in v.items():assert torch.equal(t,other['optimizer']['state'][k][key])
  test(c,OUT/n/'geq_full/best.pt',OUT/n/'test','cuda:0')
  proof[n]=dict(migration_exact=True,head_inherited=i<6,head_or_saliency_updated=True,auxiliary_positive=True,optimizer_groups=2 if i==6 else 3,strict_test_roundtrip=True,resume_exact=i in [1,6])
  print(n,'PASS',flush=True)
 dump_json(OUT/'complete.json',dict(passed=True,source=ck.source_identity(),proofs=proof,kind='bounded compatibility check; no formal GMR retraining'))
if __name__=='__main__':main()
