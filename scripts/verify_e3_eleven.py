"""Per-family real data acceptance; fresh process per native namespace."""
import argparse,json,os,signal
from pathlib import Path
import torch
from soccergmr import checkpoint as ck,train as training
from soccergmr.config import load_config
from soccergmr.clip import wrap_gmr,clip_auxiliary_loss
from soccergmr.geq import GEQFull
from soccergmr.backbones.registry import build,get_spec,forward,encoder_layers
from soccergmr.data import SoccerDataset,collate,move
from soccergmr.io import dump_json,dump_jsonl,read_jsonl
from soccergmr.test import test
R=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--backbone',required=True);a=p.parse_args();name=a.backbone;out=R/'runs/e3_eleven/acceptance'/name;out.mkdir(parents=True,exist_ok=True)
 raw=json.loads((R/'configs/experiments'/f'{name}.json').read_text());splits={}
 for split,path in raw['data']['splits'].items():
  rows=read_jsonl(path);positive=[r for r in rows if r['relevant_windows']];selected=[next(r for r in positive if len(r['relevant_windows'])>1),next(r for r in rows if not r['relevant_windows']),next(r for r in positive if len(r['relevant_windows'])==1),max(rows,key=lambda r:r['duration'])]
  selected.append(next(r for r in positive if r['vid']!=selected[0]['vid']));dest=out/(split+'.jsonl');dump_jsonl(dest,list({r['qid']:r for r in selected}.values()));splits[split]=str(dest)
 raw['data']['splits']=splits;raw['data']['full_label']='Bounded acceptance only; not performance results';raw['protocol'].update(max_epochs=2,geq_max_epochs=2,batch_size=2,eval_batch_size=2,workers=0)
 cfgpath=out/'config.json';dump_json(cfgpath,raw);c=load_config(cfgpath);spec=get_spec(name);ds=SoccerDataset(c,'train',spec);base,criterion=build(c,'cuda:0');model=wrap_gmr(base,spec,c).to('cuda:0');null=next(i for i,r in enumerate(ds.data) if not r['relevant_windows']);gradproof=[]
 for epoch in ([0,100] if name=='taskweave' else [0]):
  for indices in [[0,1],[null,null]]:
   metas,x,y=collate([ds[i] for i in indices]);x,y=move(x,'cuda:0'),move(y,'cuda:0');model.train();model.zero_grad(set_to_none=True)
   o=forward(model,name,x,y,metas,epoch=epoch,batch_idx=1,train_loader_length=2);native,_=criterion(o['native_output'],y,epoch);aux,_=clip_auxiliary_loss(o['clip_logits'],y,x['src_vid_mask'],c['clip'])
   assert o['clip_logits'].shape==x['src_vid_mask'].shape
   aux.backward(retain_graph=True)
   encoder=sum(float(p.grad.abs().sum()) for layer in encoder_layers(base,name) for p in layer.parameters() if p.grad is not None);assert encoder>0
   model.zero_grad(set_to_none=True);(native+.1*aux).backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
   gradproof.append(dict(epoch=epoch,all_null=indices[0]==indices[1],native=float(native.detach()),aux=float(aux.detach()),encoder_gradient=encoder))
 # Eval must not mutate native state (especially LD queues/momentum encoders).
 model.eval();before={k:v.detach().clone() for k,v in model.state_dict().items()}
 with torch.no_grad():forward(model,name,x,metas=metas)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
 model.close();del model,base,before,o,native,aux
 # Exact interrupt/resume check at Stage1.
 ev=training.evaluate
 def stop(*args,**kw):
  v=ev(*args,**kw);os.kill(os.getpid(),signal.SIGTERM);return v
 training.evaluate=stop;inter=training.train(c,out/'gmr','gmr','cuda:0');training.evaluate=ev;assert inter['completed_epoch']==1
 training.train(c,out/'gmr','gmr','cuda:0',resume=True);training.train(c,out/'continuous','gmr','cuda:0')
 saved=ck.load(out/'gmr/last.pt');ref=ck.load(out/'continuous/last.pt');assert saved['selector']==ref['selector'] and all(torch.equal(v,ref['model'][k]) for k,v in saved['model'].items())
 for k,v in saved['optimizer']['state'].items():
  for key,t in v.items():assert torch.equal(t,ref['optimizer']['state'][k][key])
 initial=ck.load(out/'gmr/best.pt')['model'];oldsave=ck.save
 def checked(path,payload):
  if payload['stage']=='geq_full':
   assert len(payload['optimizer']['param_groups'])==3
   if payload['epoch']==0:assert all(torch.equal(v,payload['model'][k].cpu()) for k,v in initial.items())
  return oldsave(path,payload)
 ck.save=checked;training.train(c,out/'geq_full','geq_full','cuda:0',out/'gmr/best.pt');ck.save=oldsave
 final=ck.load(out/'geq_full/last.pt');assert all(h['clip_loss']>0 for h in final['history']);assert any(not torch.equal(v,final['model'][k]) for k,v in initial.items() if k.startswith('clip_head.'))
 test(c,out/'gmr/best.pt',out/'gmr_test','cuda:0');test(c,out/'geq_full/best.pt',out/'geq_full_test','cuda:0')
 dump_json(out/'complete.json',dict(passed=True,backbone=name,source=ck.source_identity(),gradient_proofs=gradproof,eval_state_unchanged=True,exact_resume=True,optimizer_exact=True,head_inheritance=True,head_updates=True,bounded_test_roundtrip=True,kind='bounded functional validation; not benchmark scores'))
 print(name,'PASS',flush=True)
if __name__=='__main__':main()
