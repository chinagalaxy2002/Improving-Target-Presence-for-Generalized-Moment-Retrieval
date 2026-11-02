#!/usr/bin/env python3
"""Native EaTR and paper-reconstructed GMR adapter on Soccer-GMR Standard."""
import argparse,gc,hashlib,json,os,random,sys,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from config import BaseOptions
from models.build_model import build_model
from models.module import MLP
from utils.span_utils import span_xx_to_cxw,span_cxw_to_xx
sys.path.insert(0,'/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared')
from official_eval.eval_main import evaluate_gmr
SOURCE=Path('/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval')
OUT=ROOT
COMMIT='384f09396a32741a73106d2e147bde54cbcce48f'
def dump(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,default=str)+'\n')
def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True,warn_only=True)
def worker_seed(w):
 s=torch.initial_seed()%2**32; random.seed(s); np.random.seed(s)
def options(variant):
 b=BaseOptions(); b.initialize(); o=b.parser.parse_args([])
 o.dset_name='soccer_gmr'; o.device=torch.device('cuda'); o.seed=2026; o.lr=3e-5
 o.v_feat_dim=2818; o.t_feat_dim=512; o.ctx_mode='video_tef'; o.max_windows=10000
 o.use_exist_head=variant=='gmr'; o.exist_loss_coef=1.; o.exist_gate_thd=.4
 o.txt_drop_ratio=0.; o.augmentation=False; o.qcd=False; o.query_swap=False
 o.background_replacement=False; o.synthetic_negatives=False
 o.saliency_labels='not supplied by Soccer-GMR Standard; native criterion returns zero'
 o.feature_order=['slowfast2304','clip512','tef2']; o.feature_stride_seconds=2
 o.source_commit=COMMIT; o.checkpoint_selection='validation positive-query mAP'
 assert o.bsz==32 and o.n_epoch==200 and o.lr_drop==150
 assert not any((o.txt_drop_ratio,o.augmentation,o.qcd,o.query_swap,o.background_replacement,o.synthetic_negatives))
 return o
class SoccerDataset(Dataset):
 def __init__(self,o,split):
  self.o=o; self.data=[json.loads(l) for l in (SOURCE/f'data/label/Standard/{split}.jsonl').read_text().splitlines() if l.strip()]
 def __len__(self): return len(self.data)
 def __getitem__(self,i):
  m=self.data[i]; o=self.o; vs=[]
  for kind in ('slowfast','clip'):
   with np.load(SOURCE/f'Soccergmr/{kind}/{m["vid"]}.npz') as z: v=z['features'][:o.max_v_l].astype(np.float32)
   v=v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-5); vs.append(v)
  length=min(map(len,vs)); v=torch.from_numpy(np.concatenate([a[:length] for a in vs],axis=-1))
  st=torch.arange(length,dtype=torch.float32)/length; v=torch.cat([v,torch.stack([st,st+1/length],-1)],-1)
  with np.load(SOURCE/f'Soccergmr/clip_text/qid{m["qid"]}.npz') as z: q=z['last_hidden_state'][:o.max_q_l].astype(np.float32)
  q=torch.from_numpy(q/(np.linalg.norm(q,axis=-1,keepdims=True)+1e-5))
  spans=span_xx_to_cxw(torch.tensor(m['relevant_windows'],dtype=torch.float32).reshape(-1,2)/(length*o.clip_length))
  assert v.shape[-1]==2818 and q.shape[-1]==512 and len(spans)==len(m['relevant_windows'])
  return m,q,v,spans
def collate(batch):
 metas,qs,vs,spans=zip(*batch)
 qmask=pad_sequence([torch.ones(len(q),dtype=torch.bool) for q in qs],batch_first=True)
 vmask=pad_sequence([torch.ones(len(v),dtype=torch.bool) for v in vs],batch_first=True)
 inputs={'src_txt':pad_sequence(qs,batch_first=True),'src_txt_mask':qmask,'src_vid':pad_sequence(vs,batch_first=True),'src_vid_mask':vmask}
 return list(metas),inputs,{'span_labels':[{'spans':s} for s in spans],'exist_label':torch.tensor([bool(len(s)) for s in spans],dtype=torch.float32)}
def cuda_batch(batch):
 metas,inputs,targets=batch
 return metas,{k:v.cuda(non_blocking=True) for k,v in inputs.items()},{'span_labels':[{'spans':s['spans'].cuda()} for s in targets['span_labels']],'exist_label':targets['exist_label'].cuda()}
def loader(o,d,train=False):
 return DataLoader(d,batch_size=o.bsz if train else o.eval_bsz,shuffle=train,num_workers=o.num_workers,collate_fn=collate,pin_memory=True,worker_init_fn=worker_seed,generator=torch.Generator().manual_seed(o.seed))
def setup(o):
 model,criterion=build_model(o)
 if o.use_exist_head: model.exist_head=MLP(o.hidden_dim,o.hidden_dim,1,2)
 model.cuda(); criterion.cuda()
 optimizer=torch.optim.AdamW(model.parameters(),lr=o.lr,weight_decay=o.wd)
 scheduler=torch.optim.lr_scheduler.StepLR(optimizer,o.lr_drop)
 return model,criterion,optimizer,scheduler
def subset(outputs,keep):
 ids=keep.nonzero().flatten().tolist(); out={}
 for k,v in outputs.items():
  if k=='aux_outputs': out[k]=[{kk:vv[keep] for kk,vv in a.items()} for a in v]
  elif k=='pseudo_event_spans': out[k]=[v[i] for i in ids]
  else: out[k]=v[keep]
 return out
def loss_fn(criterion,outputs,targets,epoch=0):
 keep=targets['exist_label'].bool() if 'pred_exist_logits' in outputs else torch.ones_like(targets['exist_label'],dtype=torch.bool)
 ids=keep.nonzero().flatten().tolist(); total=outputs['pred_logits'].sum()*0+outputs['pred_spans'].sum()*0; values={}
 if ids:
  td={'span_labels':[targets['span_labels'][i] for i in ids]}
  ld=criterion(subset(outputs,keep),td,epoch)
  for k,v in ld.items():
   if k in criterion.weight_dict:
    assert not torch.is_tensor(v) or torch.isfinite(v).all(),(k,v)
    total=total+v*criterion.weight_dict[k]; values[k]=float(v.detach()) if torch.is_tensor(v) else float(v)
 if 'pred_exist_logits' in outputs:
  le=torch.nn.functional.binary_cross_entropy_with_logits(outputs['pred_exist_logits'],targets['exist_label'])
  total=total+le; values['loss_exist']=float(le.detach())
 values['total']=float(total.detach()); return total,values
@torch.no_grad()
def predict(model,o,d):
 model.eval(); preds=[]
 for raw in loader(o,d):
  metas,inputs,_=cuda_batch(raw); outputs=model(**inputs)
  spans=span_cxw_to_xx(outputs['pred_spans']).cpu(); scores=outputs['pred_logits'].softmax(-1)[...,0].cpu()
  existence=outputs.get('pred_exist_logits'); existence=existence.sigmoid().cpu() if existence is not None else None
  for i,m in enumerate(metas):
   # Original EaTR clip_ts then round_multiple; upper bound follows each duration.
   sec=(spans[i]*m['duration']).clamp(0,m['duration']); sec=torch.round(sec/o.clip_length)*o.clip_length
   sec=sec.clamp(0,m['duration']); windows=sorted(torch.cat([sec,scores[i,:,None]],-1).tolist(),key=lambda a:a[2],reverse=True)
   windows=[[float(f'{x:.4f}') for x in w] for w in windows]
   p={'qid':m['qid'],'vid':m['vid'],'query':m['query'],'pred_relevant_windows':windows}
   if existence is not None: p['pred_exist_score']=float(existence[i])
   assert len(windows)==o.num_queries and np.isfinite(windows).all(); preds.append(p)
 assert {x['qid'] for x in preds}=={m['qid'] for m in d.data}
 return preds
def save_preds(path,preds): path.write_text(''.join(json.dumps(p)+'\n' for p in preds))
def smoke(variant):
 o=options(variant); o.num_workers=0; seed_all(o.seed); out=OUT/variant; out.mkdir(parents=True,exist_ok=True)
 dump(out/'effective_config.json',vars(o)); tr=SoccerDataset(o,'train'); va=SoccerDataset(o,'val')
 indices=[next(i for i,m in enumerate(tr.data) if m['relevant_windows']),next(i for i,m in enumerate(tr.data) if not m['relevant_windows'])]
 model,criterion,opt,_=setup(o); torch.cuda.reset_peak_memory_stats(); report=[]
 for ix in indices:
  metas,inputs,t=cuda_batch(collate([tr[ix]])); model.train(); opt.zero_grad(); loss,vals=loss_fn(criterion,model(**inputs),t)
  assert torch.isfinite(loss); loss.backward(); grads=[p.grad for p in model.parameters() if p.grad is not None]
  assert grads and all(torch.isfinite(g).all() for g in grads); nonzero=sum(bool(g.abs().max()>0) for g in grads); assert nonzero
  opt.step(); report.append({'qid':metas[0]['qid'],'positive':bool(t['exist_label'].item()),'loss':vals,'nonzero_grad_tensors':nonzero})
 # Measure actual full configured batch and maximum supported sequence lengths.
 examples=[tr[indices[i%2]] for i in range(o.bsz)]
 longest=max(range(len(tr)),key=lambda i:tr.data[i]['duration']); examples[0]=tr[longest]
 _,inputs,t=cuda_batch(collate(examples)); opt.zero_grad(); loss,_=loss_fn(criterion,model(**inputs),t); loss.backward(); opt.step()
 ckpt=out/'smoke_checkpoint.pt'; torch.save({'model':model.state_dict()},ckpt); model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)['model'],strict=True)
 va.data=[next(m for m in va.data if m['relevant_windows']),next(m for m in va.data if not m['relevant_windows'])]
 preds=predict(model,o,va); save_preds(out/'smoke_val_predictions.jsonl',preds)
 metrics=evaluate_gmr(preds,va.data,cls_thresholds=(.4,),map_num_workers=1,verbose=False)
 peak=torch.cuda.max_memory_allocated()/2**20
 dump(out/'smoke.json',{'passed':True,'variant':variant,'samples':report,'peak_allocated_mib':peak,'batch_size_measured':o.bsz,'strict_checkpoint_reload':True,'validation_metrics':metrics})
 print('SMOKE_PASSED',variant,'peak_mib',peak,flush=True); del model,criterion,opt; gc.collect(); torch.cuda.empty_cache()
def train(variant):
 o=options(variant); seed_all(o.seed); out=OUT/variant; out.mkdir(parents=True,exist_ok=True)
 assert json.loads((out/'smoke.json').read_text())['passed']
 dump(out/'effective_config.json',vars(o)); dump(out/'environment.json',{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'visible_gpus':os.getenv('CUDA_VISIBLE_DEVICES'),'command':sys.argv,'source_commit':COMMIT,'seed':2026})
 tr=SoccerDataset(o,'train'); va=SoccerDataset(o,'val'); ld=loader(o,tr,True); model,criterion,opt,scheduler=setup(o)
 best=-float('inf'); bad=0; start=0; last=out/'last.ckpt'
 if last.exists():
  s=torch.load(last,map_location='cpu',weights_only=False); model.load_state_dict(s['model'],strict=True); opt.load_state_dict(s['optimizer']); scheduler.load_state_dict(s['scheduler']); start=s['epoch']+1; best=s['best']; bad=s['bad']
  random.setstate(s['python_rng']); np.random.set_state(s['numpy_rng']); torch.set_rng_state(s['torch_rng']); torch.cuda.set_rng_state_all(s['cuda_rng']); ld.generator.set_state(s['loader_rng'])
 for epoch in range(start,o.n_epoch):
  model.train(); criterion.train(); begun=time.time(); dump(out/'progress.json',{'status':'TRAINING','epoch':epoch+1,'best_val_map':best})
  for step,raw in enumerate(ld):
   _,inputs,t=cuda_batch(raw); opt.zero_grad(); loss,values=loss_fn(criterion,model(**inputs),t,epoch); assert torch.isfinite(loss),values
   loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),o.grad_clip,error_if_nonfinite=True); opt.step()
   if step%20==0: print(json.dumps({'variant':variant,'epoch':epoch+1,'step':step,'loss':values,'elapsed':time.time()-begun}),flush=True)
  scheduler.step(); preds=predict(model,o,va); save_preds(out/'latest_val_predictions.jsonl',preds)
  metrics=evaluate_gmr(preds,va.data,cls_thresholds=(.4,),map_num_workers=1,verbose=False); score=metrics['brief']['mAP']; improved=score>best; best=max(best,score); bad=0 if improved else bad+1
  state={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'bad':bad,'config':vars(o),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':ld.generator.get_state()}
  torch.save(state,last)
  if improved: torch.save(state,out/'best.ckpt'); save_preds(out/'best_val_predictions.jsonl',preds); dump(out/'best_val_metrics.json',metrics)
  dump(out/'progress.json',{'status':'TRAINING','epoch':epoch+1,'best_val_map':best,'val_map':score,'early_stop_counter':bad}); print('VALIDATION',variant,epoch+1,json.dumps(metrics['brief']),flush=True)
  if o.max_es_cnt!=-1 and bad>o.max_es_cnt: break
 bestpath=out/'best.ckpt'; sha=hashlib.sha256(bestpath.read_bytes()).hexdigest(); dump(out/'frozen_selection.json',{'checkpoint_sha256':sha,'criterion':o.checkpoint_selection,'threshold':.4,'best_val_map':best})
 model.load_state_dict(torch.load(bestpath,map_location='cpu',weights_only=False)['model'],strict=True); te=SoccerDataset(o,'test'); preds=predict(model,o,te); save_preds(out/'test_predictions.jsonl',preds)
 metrics=evaluate_gmr(preds,te.data,cls_thresholds=(.4,),map_num_workers=1,verbose=False); dump(out/'test_metrics.json',metrics); dump(out/'progress.json',{'status':'COMPLETED','epoch':epoch+1,'best_val_map':best,'checkpoint_sha256':sha})
 del model,criterion,opt; gc.collect(); torch.cuda.empty_cache()
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['smoke','train'],required=True); p.add_argument('--variant',choices=['original','gmr','both'],default='both'); a=p.parse_args()
 for variant in ['original','gmr'] if a.variant=='both' else [a.variant]: (smoke if a.mode=='smoke' else train)(variant)
