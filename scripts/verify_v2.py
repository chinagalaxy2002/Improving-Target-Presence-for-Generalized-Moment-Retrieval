"""Bounded real-asset acceptance; scores never used as experimental evidence."""
import importlib.util,json,os,signal
from pathlib import Path
import torch
from soccergmr import checkpoint as ck,train as training
from soccergmr.backbones.registry import build,get_spec
from soccergmr.clip import wrap_gmr,load_native_gmr,clip_auxiliary_loss
from soccergmr.config import load_config
from soccergmr.data import SoccerDataset,collate,move
from soccergmr.geq import GEQFull
from soccergmr.io import dump_json,dump_jsonl,read_jsonl
from soccergmr.test import test
R=Path(__file__).resolve().parents[1];OUT=R/'runs/clip_v2/acceptance';DEVICE='cuda:0'

def main():
 OUT.mkdir(parents=True,exist_ok=True);cfg=load_config(R/'configs/experiments/E1.json');splits={}
 for split,path in cfg['data']['splits'].items():
  rows=read_jsonl(path)
  selected=[next(r for r in rows if len(r['relevant_windows'])>1),next(r for r in rows if not r['relevant_windows']),next(r for r in rows if len(r['relevant_windows'])==1),max(rows,key=lambda r:r['duration'])]
  p=OUT/(split+'.jsonl');dump_jsonl(p,list({r['qid']:r for r in selected}.values()));splits[split]=str(p)
 proof={};spec=get_spec('moment_detr')
 for name in ['E1','E2','E3','E4','E5','E6']:
  raw=json.loads((R/'configs/experiments'/f'{name}.json').read_text());raw['data']['splits']=splits;raw['data']['full_label']='BOUNDED compatibility check, not benchmark'
  raw['protocol'].update(max_epochs=2,geq_max_epochs=2,batch_size=2,eval_batch_size=2,workers=0)
  path=OUT/(name+'.json');dump_json(path,raw);c=load_config(path);dataset=SoccerDataset(c,'train',spec)
  base,criterion=build(c,DEVICE);model=wrap_gmr(base,spec,c).to(DEVICE);model.eval()
  nul=next(i for i,r in enumerate(dataset.data) if not r['relevant_windows'])
  for indices in [range(len(dataset)),[nul,nul]]:
   _,x,y=collate([dataset[i] for i in indices]);x,y=move(x,DEVICE),move(y,DEVICE)
   native=base(**x);out=model(**x);assert torch.equal(native['pred_logits'],out['pred_logits'])
   native_loss=criterion(native,y)[0];assert torch.equal(native_loss,criterion(out.get('native_output',out),y)[0])
   z=out['saliency_scores'] if c['clip']['method']=='native' else out['clip_logits']
   loss,_=clip_auxiliary_loss(z,y,x['src_vid_mask'],c['clip']);model.zero_grad(set_to_none=True);loss.backward()
   assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   assert sum(float(p.grad.abs().sum()) for n,p in base.named_parameters() if p.grad is not None and 'encoder' in n)>0
  if hasattr(model,'close'):model.close()
  del model,base,out,native
  # Exact interrupted vs continuous Stage-1 resume includes auxiliary state.
  gmr=OUT/name/'gmr';training.train(c,gmr,'gmr',DEVICE)
  if name in ['E1','E6']:
   ev=training.evaluate
   def stop(*a,**kw):
    value=ev(*a,**kw);os.kill(os.getpid(),signal.SIGTERM);return value
   training.evaluate=stop
   result=training.train(c,OUT/name/'resumed','gmr',DEVICE);assert result['completed_epoch']==1
   training.evaluate=ev;training.train(c,OUT/name/'resumed','gmr',DEVICE,resume=True)
   a=ck.load(gmr/'last.pt');b=ck.load(OUT/name/'resumed/last.pt')
   assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
   assert a['selector']==b['selector']
   for k,v in a['optimizer']['state'].items():
    for key,t in v.items():assert torch.equal(t,b['optimizer']['state'][k][key])
  initial=ck.load(gmr/'best.pt')
  # Same seeds and same native state: old original GEQ vs head-free v2 GEQ.
  reference=Path('/home/guoxiangyu/SoccerGMR-GEQ/src/soccergmr/geq.py')
  ms=importlib.util.spec_from_file_location('soccergmr.reference_geq',reference);old=importlib.util.module_from_spec(ms);ms.loader.exec_module(old)
  ck.seed_all(2023);b,_=build(c,DEVICE);b=load_native_gmr(b,spec,c,initial['model']);new=GEQFull(b,spec,clip_head=False).to(DEVICE)
  ck.seed_all(2023);b,_=build(c,DEVICE);native_state={k.removeprefix('base.'):v for k,v in initial['model'].items() if not k.startswith('clip_head.')} if name!='E6' else initial['model']
  b.load_state_dict(native_state,strict=True);ref=old.GEQFull(b,spec).to(DEVICE)
  assert set(new.state_dict())==set(ref.state_dict()) and not any(k.startswith('clip_head') for k in new.state_dict())
  assert all(torch.equal(v,ref.state_dict()[k]) for k,v in new.state_dict().items())
  new.train();ref.train();ck.seed_all(91);o1=new(**x);ck.seed_all(91);o2=ref(**x)
  assert torch.equal(o1['geq_logit'],o2['geq_logit']) and torch.equal(o1['pred_spans'],o2['pred_spans'])
  new.close();ref.close();del new,ref
  # Inspect actual trainer epoch-zero transfer and groups before updates.
  save=ck.save
  def checked(path,payload):
   if payload['stage']=='geq_full':
    assert not any(k.startswith('clip_head.') for k in payload['model'])
    assert len(payload['optimizer']['param_groups'])==2
    if payload['epoch']==0:assert all(torch.equal(v,payload['model']['base.'+k].cpu()) for k,v in native_state.items())
   return save(path,payload)
  ck.save=checked
  training.train(c,OUT/name/'geq_full','geq_full',DEVICE,gmr/'best.pt');ck.save=save
  last=ck.load(OUT/name/'geq_full/last.pt');assert all(h['clip_loss']==0 for h in last['history'])
  test(c,gmr/'best.pt',OUT/name/'gmr_test',DEVICE);test(c,OUT/name/'geq_full/best.pt',OUT/name/'geq_test',DEVICE)
  proof[name]=dict(native_loss_equal=True,encoder_clip_gradient=True,all_null_finite=True,original_geq_exact=True,head_discarded=True,geq_auxiliary_zero=True,strict_checkpoint_roundtrip=True,resume_exact=name in ['E1','E6'])
  print(name,'PASS',flush=True)
 dump_json(OUT/'complete.json',dict(passed=True,source=ck.source_identity(),proofs=proof,kind='bounded acceptance, NOT benchmark performance'))
if __name__=='__main__':main()
