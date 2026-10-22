#!/usr/bin/env python3
"""Isolated Table 2 Moment-DETR/Moment-DETR-GMR single-seed deployment."""
import argparse, gc, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'training/moment_detr_gmr'))
from config import BaseOptions
from dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from evaluate import setup_model
from models.moment_detr_gmr.utils.span_utils import span_cxw_to_xx
sys.path.insert(0,'/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared')
from official_eval.eval_main import evaluate_gmr
SHARED=Path('/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval')
OUT=ROOT

def dump(path, value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,default=str)+'\n')
def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True,warn_only=True)
def worker_seed(worker_id):
 seed=torch.initial_seed()%2**32; np.random.seed(seed); random.seed(seed)
def options(variant):
 m=BaseOptions('moment_detr','soccer_gmr','clip_slowfast'); m.parse(); o=m.option
 o.update(seed=2026,lr=3e-5,use_exist_head=variant=='gmr',exist_loss_coef=1.,exist_gate_thd=.4,
          max_q_l=32,max_v_l=75,clip_length=2,max_windows=10000,num_workers=4,
          results_dir=str(OUT/variant),device='cuda',mr_only=True,augmentation=False,
          qcd=False,query_swap=False,background_replacement=False,synthetic_negatives=False,
          checkpoint_selection='validation positive-query mAP; test only after selection frozen')
 o.train_path=str(SHARED/'data/label/Standard/train.jsonl'); o.eval_path=str(SHARED/'data/label/Standard/val.jsonl')
 o.v_feat_dirs=[str(SHARED/'Soccergmr/clip'),str(SHARED/'Soccergmr/slowfast')]
 o.t_feat_dir=str(SHARED/'Soccergmr/clip_text')
 assert not any(o[k] for k in ('augmentation','qcd','query_swap','background_replacement','synthetic_negatives'))
 return o

def dataset(o,split):
 d=StartEndDataset(dset_name='soccer_gmr',domain=None,data_path=str(SHARED/f'data/label/Standard/{split}.jsonl'),
  v_feat_dirs=o.v_feat_dirs,q_feat_dir=o.t_feat_dir,max_q_l=o.max_q_l,max_v_l=o.max_v_l,
  clip_len=o.clip_length,max_windows=o.max_windows,mr_only=True,keep_empty_gt=True,load_labels=True)
 expected=[json.loads(x) for x in Path(d.data_path).read_text().splitlines() if x.strip()]
 assert len(d)==len(expected),'Missing features: never silently drop an annotated example'
 return d

def loader(o,d,train=False):
 return DataLoader(d,batch_size=o.bsz if train else o.eval_bsz,shuffle=train,
   num_workers=o.num_workers,collate_fn=start_end_collate,worker_init_fn=worker_seed,
   generator=torch.Generator().manual_seed(o.seed),pin_memory=True)

def losses(criterion,outputs,targets):
 # Paper Sec4.1: null-set Lvmr has no gradient; Lexist supervises all examples.
 pos=targets['exist_label'].bool() if 'pred_exist_logits' in outputs else torch.ones_like(targets['exist_label'],dtype=torch.bool)
 ids=pos.nonzero().flatten().tolist()
 z=outputs['pred_logits'].sum()*0.+outputs['pred_spans'].sum()*0.
 total=z; values={}
 if ids:
  p_outputs={k:([ {kk:vv[pos] for kk,vv in a.items()} for a in v] if k=='aux_outputs' else v[pos]) for k,v in outputs.items()}
  p_targets={'span_labels':[targets['span_labels'][i] for i in ids],'exist_label':targets['exist_label'][pos]}
  ld=criterion(p_outputs,p_targets)
  for k,v in ld.items():
   if k.startswith('loss_exist'): continue
   if k in criterion.weight_dict:
    total=total+v*criterion.weight_dict[k]; values[k]=float(v.detach()) if torch.is_tensor(v) else float(v)
 if 'pred_exist_logits' in outputs:
  le=torch.nn.functional.binary_cross_entropy_with_logits(outputs['pred_exist_logits'],targets['exist_label'])
  total=total+le; values['loss_exist']=float(le.detach())
 values['total']=float(total.detach())
 return total,values

@torch.no_grad()
def predict(model,o,d):
 model.eval(); predictions=[]
 for metas,data in loader(o,d):
  inputs,_=prepare_batch_inputs(data,o.device); output=model(**inputs)
  spans=span_cxw_to_xx(output['pred_spans']).cpu()
  scores=output['pred_logits'].softmax(-1)[...,0].cpu()
  existence=output.get('pred_exist_logits'); existence=existence.sigmoid().cpu() if existence is not None else None
  for i,meta in enumerate(metas):
   # Same span transform as released Moment-DETR, bounded by each video's duration.
   seconds=(spans[i]*meta['duration']).clamp(0,meta['duration'])
   seconds=torch.round(seconds/o.clip_length)*o.clip_length
   seconds=seconds.clamp(0,meta['duration'])
   windows=sorted(torch.cat([seconds,scores[i,:,None]],-1).tolist(),key=lambda w:w[2],reverse=True)
   entry={'qid':meta['qid'],'query':meta['query'],'vid':meta['vid'],'pred_relevant_windows':windows}
   if existence is not None: entry['pred_exist_score']=float(existence[i])
   assert len(windows)==o.num_queries and all(len(w)==3 and np.isfinite(w).all() for w in windows)
   predictions.append(entry)
 assert {p['qid'] for p in predictions}=={m['qid'] for m in d.data}
 return predictions

def save_preds(path,preds):
 path.write_text(''.join(json.dumps(p)+'\n' for p in preds))

def smoke(variant):
 o=options(variant); seed_all(o.seed); o.num_workers=0
 output=OUT/variant; output.mkdir(parents=True,exist_ok=True); dump(output/'effective_config.json',dict(o))
 d=dataset(o,'train'); val=dataset(o,'val')
 indices=[next(i for i,m in enumerate(d.data) if m['relevant_windows']),next(i for i,m in enumerate(d.data) if not m['relevant_windows'])]
 model,criterion,opt,_=setup_model(o); torch.cuda.reset_peak_memory_stats()
 reports=[]
 for ix in indices:
  batch=start_end_collate([d[ix]]); inputs,targets=prepare_batch_inputs(batch[1],o.device)
  model.train(); opt.zero_grad(); outputs=model(**inputs); loss,values=losses(criterion,outputs,targets)
  assert torch.isfinite(loss); loss.backward()
  grads=[p.grad for p in model.parameters() if p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads)
  nonzero=sum(bool(g.abs().max()>0) for g in grads)
  assert nonzero>0
  opt.step(); reports.append({'qid':batch[0][0]['qid'],'positive':bool(targets['exist_label'].item()),'loss':values,'nonzero_grad_tensors':nonzero})
 # Full configured batch ensures the peak memory measurement covers training bsz8.
 mixed=[d[indices[j%2]] for j in range(o.bsz)]
 inputs,targets=prepare_batch_inputs(start_end_collate(mixed)[1],o.device)
 opt.zero_grad(); loss,_=losses(criterion,model(**inputs),targets); loss.backward(); opt.step()
 # Check strict checkpoint loading, including adapter keys.
 ckpt=output/'smoke_checkpoint.pt'; torch.save({'model':model.state_dict()},ckpt)
 model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)['model'],strict=True)
 val.data=[next(m for m in val.data if m['relevant_windows']),next(m for m in val.data if not m['relevant_windows'])]
 preds=predict(model,o,val); save_preds(output/'smoke_val_predictions.jsonl',preds)
 metrics=evaluate_gmr(preds,val.data,cls_thresholds=(.4,),map_num_workers=1,verbose=False)
 peak=torch.cuda.max_memory_allocated()/2**20
 dump(output/'smoke.json',{'passed':True,'variant':variant,'samples':reports,'peak_allocated_mib':peak,'batch_size_measured':o.bsz,'validation_metrics':metrics,'strict_checkpoint_reload':True})
 print('SMOKE_PASSED',variant,'peak_mib',peak,flush=True)
 del model,criterion,opt; gc.collect(); torch.cuda.empty_cache()

def train(variant):
 o=options(variant); seed_all(o.seed); output=OUT/variant; output.mkdir(parents=True,exist_ok=True)
 assert json.loads((output/'smoke.json').read_text())['passed']
 dump(output/'effective_config.json',dict(o)); dump(output/'environment.json',{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'visible_gpus':os.getenv('CUDA_VISIBLE_DEVICES'),'command':sys.argv,'seed':2026})
 tr=dataset(o,'train'); va=dataset(o,'val'); ld=loader(o,tr,True)
 model,criterion,optimizer,scheduler=setup_model(o); best=-float('inf'); bad=0; start=0
 last=output/'last.ckpt'
 if last.exists():
  state=torch.load(last,map_location='cpu',weights_only=False); model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler']); start=state['epoch']+1; best=state['best']; bad=state['bad']
  random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng']); ld.generator.set_state(state['loader_rng'])
 for epoch in range(start,o.n_epoch):
  model.train(); criterion.train(); begun=time.time()
  for step,(_,data) in enumerate(ld):
   inputs,targets=prepare_batch_inputs(data,o.device); optimizer.zero_grad(); loss,values=losses(criterion,model(**inputs),targets)
   assert torch.isfinite(loss),values; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),o.grad_clip,error_if_nonfinite=True); optimizer.step()
   if step%20==0: print(json.dumps({'variant':variant,'epoch':epoch+1,'step':step,'loss':values,'elapsed':time.time()-begun}),flush=True)
  scheduler.step(); preds=predict(model,o,va); save_preds(output/'latest_val_predictions.jsonl',preds)
  metrics=evaluate_gmr(preds,va.data,cls_thresholds=(.4,),map_num_workers=1,verbose=False); score=metrics['brief']['mAP']; improved=score>best
  bad=0 if improved else bad+1; best=max(best,score)
  state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'bad':bad,'opt':dict(o),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':ld.generator.get_state()}
  torch.save(state,last)
  if improved:
   torch.save(state,output/'best.ckpt'); save_preds(output/'best_val_predictions.jsonl',preds); dump(output/'best_val_metrics.json',metrics)
  dump(output/'progress.json',{'status':'TRAINING','epoch':epoch+1,'best_val_map':best,'val_map':score,'early_stop_counter':bad})
  print('VALIDATION',variant,epoch+1,json.dumps(metrics['brief']),flush=True)
  if bad>=o.max_es_cnt: break
 # Freeze provenance before loading the test split for the first time.
 bestpath=output/'best.ckpt'; sha=hashlib.sha256(bestpath.read_bytes()).hexdigest()
 dump(output/'frozen_selection.json',{'checkpoint_sha256':sha,'criterion':o.checkpoint_selection,'threshold':.4,'best_val_map':best})
 model.load_state_dict(torch.load(bestpath,map_location='cpu',weights_only=False)['model'],strict=True)
 te=dataset(o,'test'); preds=predict(model,o,te); save_preds(output/'test_predictions.jsonl',preds)
 metrics=evaluate_gmr(preds,te.data,cls_thresholds=(.4,),map_num_workers=1,verbose=False); dump(output/'test_metrics.json',metrics)
 dump(output/'progress.json',{'status':'COMPLETED','epoch':epoch+1,'best_val_map':best,'checkpoint_sha256':sha})
 del model,criterion,optimizer; gc.collect(); torch.cuda.empty_cache()

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['smoke','train'],required=True); p.add_argument('--variant',choices=['original','gmr','both'],default='both'); args=p.parse_args()
 for v in ['original','gmr'] if args.variant=='both' else [args.variant]:
  (smoke if args.mode=='smoke' else train)(v)
