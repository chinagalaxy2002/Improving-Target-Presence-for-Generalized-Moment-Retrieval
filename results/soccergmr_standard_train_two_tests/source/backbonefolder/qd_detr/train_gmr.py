#!/usr/bin/env python3
"""Locally reconstructed QD-DETR-GMR extension; separate from published Table2 targets."""
import os, sys, json, random, argparse, time, hashlib, subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'gmr_deployment'))
SHARED=Path('/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared')
sys.path.insert(0,str(SHARED))
from qd_detr.config import BaseOptions
from qd_detr.model import build_model
from qd_detr.span_utils import span_xx_to_cxw,span_cxw_to_xx
from official_eval.eval_main import evaluate_gmr
ASSETS=Path('/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr')
LABELS=ASSETS.parent/'data/label/Standard'
OUT=ROOT/'gmr'

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def seed_worker(worker):
 seed=torch.initial_seed()%2**32;np.random.seed(seed);random.seed(seed)

def norm(a):return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-5)
class Soccer(Dataset):
 def __init__(self,split):
  self.data=[json.loads(l) for l in (LABELS/f'{split}.jsonl').read_text().splitlines() if l.strip()]
  self.cache={}
 def __len__(self):return len(self.data)
 def __getitem__(self,i):
  d=self.data[i];vid=Path(d['vid']).stem
  if vid not in self.cache:
   fs=[np.load(ASSETS/f/f'{vid}.npz')['features'].astype(np.float32) for f in ('slowfast','clip')]
   n=min(map(len,fs));assert n>0 and n<=75
   self.cache[vid]=torch.from_numpy(np.concatenate([norm(f[:n]) for f in fs],-1))
  video=self.cache[vid];n=len(video)
  tef=torch.stack([torch.arange(n)/n,(torch.arange(n)+1)/n],-1)
  video=torch.cat([video,tef],-1)
  q=norm(np.load(ASSETS/'clip_text'/f'qid{d["qid"]}.npz')['last_hidden_state'][:32].astype(np.float32))
  windows=torch.tensor(d['relevant_windows'],dtype=torch.float32).reshape(-1,2)
  spans=span_xx_to_cxw(windows/float(d['duration']))
  sal=torch.zeros(n)
  for start,end in windows:
   a=max(0,min(n-1,int(start/float(d['duration'])*n)));b=max(a+1,min(n,int(np.ceil(float(end)/float(d['duration'])*n))))
   sal[a:b]=1
  pos=torch.where(sal>0)[0].tolist();neg=torch.where(sal==0)[0].tolist()
  p=random.choices(pos,k=2) if pos else [0,0];ng=random.choices(neg,k=2) if neg else p
  return d,torch.from_numpy(q),video,spans,sal,p,ng,float(bool(pos) and bool(neg))

def collate(items):
 meta,q,v,spans,sal,pos,neg,pm=zip(*items)
 qm=pad_sequence([torch.ones(len(x)) for x in q],batch_first=True);vm=pad_sequence([torch.ones(len(x)) for x in v],batch_first=True)
 inputs={'src_txt':pad_sequence(q,batch_first=True),'src_vid':pad_sequence(v,batch_first=True),'src_txt_mask':qm,'src_vid_mask':vm}
 target={'span_labels':[{'spans':s} for s in spans],'saliency_all_labels':pad_sequence(sal,batch_first=True),'saliency_pos_labels':torch.tensor(pos),'saliency_neg_labels':torch.tensor(neg),'saliency_pair_mask':torch.tensor(pm)}
 return list(meta),inputs,target

def move(inputs,targets,device):
 inputs={k:v.to(device) for k,v in inputs.items()}
 target={k:([{'spans':x['spans'].to(device)} for x in v] if k=='span_labels' else v.to(device)) for k,v in targets.items()}
 return inputs,target

def options():
 base=BaseOptions();base.initialize();o=base.parser.parse_args([])
 for k,v in dict(device='cuda:0',dset_name='soccer_gmr',seed=2026,lr=3e-5,n_epoch=200,lr_drop=400,bsz=32,eval_bsz=32,v_feat_dim=2818,t_feat_dim=512,num_workers=4,max_windows=None,max_es_cnt=200,augmentation=False,query_swap=False,synthetic_negatives=False,selection_metric='validation positive-query mAP',variant='gmr',existence_loss_coef=1.0,vmr_supervision='positive queries only',adapter='max final decoder slots -> Linear256 ReLU Linear1',temporal_normalization='duration',feature_order=['slowfast','clip','tef'],gmiou_cls_threshold=0.4,train_path=str(LABELS/'train.jsonl'),eval_path=str(LABELS/'val.jsonl'),v_feat_dirs=[str(ASSETS/'slowfast'),str(ASSETS/'clip')],t_feat_dir=str(ASSETS/'clip_text')).items():setattr(o,k,v)
 return o

def write_json(path,value):path.write_text(json.dumps(value,indent=2,default=str)+'\n')

def predictions(model,loader,device):
 model.eval();pred=[]
 with torch.no_grad():
  for meta,inputs,targets in loader:
   inputs,_=move(inputs,targets,device);out=model(**inputs)
   scores=out['pred_logits'].softmax(-1)[...,0];bounds=span_cxw_to_xx(out['pred_spans']).clamp(0,1)
   for d,b,s,exist in zip(meta,bounds,scores,out['pred_exist_logits'].sigmoid()):
    order=s.argsort(descending=True)
    # Official QD postprocessor clips then rounds to 2-second feature multiples.
    seconds=(torch.round(b*float(d['duration'])/2)*2).clamp(0,float(d['duration']))
    win=torch.cat([seconds,s[:,None]],-1)[order].cpu().tolist()
    win=[w[:2]+[round(w[2],4)] for w in win]
    pred.append({'qid':d['qid'],'query':d['query'],'vid':d['vid'],'pred_relevant_windows':win,'pred_exist_score':float(exist)})
 expected=([loader.dataset.dataset.data[i]['qid'] for i in loader.dataset.indices] if isinstance(loader.dataset,torch.utils.data.Subset) else [d['qid'] for d in loader.dataset.data])
 assert len(pred)==len(expected) and len({r['qid'] for r in pred})==len(expected) and {r['qid'] for r in pred}==set(expected),'Prediction query coverage mismatch'
 return pred

def loader(ds,o,training=False):
 g=torch.Generator().manual_seed(o.seed)
 return DataLoader(ds,batch_size=o.bsz if training else o.eval_bsz,shuffle=training,num_workers=o.num_workers,collate_fn=collate,worker_init_fn=seed_worker,generator=g,pin_memory=True,persistent_workers=False)

def train_step(model,criterion,optimizer,items,device,check_head=False):
 model.train();inputs,targets=move(items[1],items[2],device);optimizer.zero_grad(set_to_none=True)
 out=model(**inputs);losses=criterion(out,targets)
 loss=sum(losses[k]*v for k,v in criterion.weight_dict.items() if k in losses)
 assert torch.isfinite(loss),{k:float(v) for k,v in losses.items()}
 loss.backward()
 head_grads=[p.grad for p in model.exist_head.parameters()]
 assert all(g is not None and torch.isfinite(g).all() for g in head_grads),'Existence head gradients missing/nonfinite'
 if check_head:assert any(torch.count_nonzero(g).item()>0 for g in head_grads),'Existence head has zero gradient'
 grads=[p.grad for p in model.parameters() if p.grad is not None]
 assert grads and all(torch.isfinite(x).all() for x in grads)
 torch.nn.utils.clip_grad_norm_(model.parameters(),0.1);optimizer.step()
 return float(loss.detach()),{k:float(v) for k,v in losses.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['gmr'],default='gmr');p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-then-train',action='store_true');p.add_argument('--resume',type=Path);a=p.parse_args()
 OUT.mkdir(parents=True,exist_ok=True);o=options();seed_all(o.seed)
 if a.smoke or a.smoke_then_train:o.num_workers=0
 model,criterion=build_model(o);model.to(o.device)
 optimizer=torch.optim.AdamW(model.parameters(),lr=o.lr,weight_decay=o.wd);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,o.lr_drop,gamma=.1)
 train=Soccer('train');val=Soccer('val')
 assert not ({d['qid'] for d in train.data}&{d['qid'] for d in val.data})
 maxmom=max(len(d['relevant_windows']) for d in train.data);assert maxmom<=o.num_queries,('Not all targets fit native proposals',maxmom)
 if a.smoke or a.smoke_then_train:
  pidx=next(i for i,d in enumerate(train.data) if len(d['relevant_windows'])>=2);nidx=next(i for i,d in enumerate(train.data) if not d['relevant_windows'])
  batch=collate([train[pidx],train[nidx]])
  with torch.no_grad():
   for i in (pidx,nidx):
    _,x,y=collate([train[i]]);x,y=move(x,y,o.device);out=model(**x);assert torch.isfinite(out['pred_logits']).all();assert torch.isfinite(out['pred_spans']).all();assert torch.isfinite(out['pred_exist_logits']).all()
  loss,losses=train_step(model,criterion,optimizer,batch,o.device,check_head=True)
  assert criterion.last_positive_count==1,'VMR must see only the positive query'
  neg_loss,neg_losses=train_step(model,criterion,optimizer,collate([train[nidx],train[nidx]]),o.device,check_head=True)
  assert criterion.last_positive_count==0
  assert all(v==0 for k,v in neg_losses.items() if k!='loss_exist'),'Null queries must not supervise native VMR'
  # Estimate actual full batch peak, including backward + Adam states.
  full_loss,_=train_step(model,criterion,optimizer,collate([train[i] for i in range(o.bsz)]),o.device,check_head=True)
  checkpoint=OUT/'smoke_checkpoint.pt';torch.save({'model':model.state_dict()},checkpoint)
  reloaded,_=build_model(o);reloaded.to(o.device);reloaded.load_state_dict(torch.load(checkpoint,map_location=o.device,weights_only=False)['model'],strict=True)
  model.eval();reloaded.eval()
  with torch.no_grad():
   x,_=move(batch[1],batch[2],o.device);before=model(**x);after=reloaded(**x)
   assert all(torch.equal(before[k],after[k]) for k in ('pred_logits','pred_spans','pred_exist_logits'))
  del reloaded
  # Paper mask invariant: perturb only negative predictions; positive VMR loss stays exact.
  import copy
  with torch.no_grad():
   x,y=move(batch[1],batch[2],o.device);outputs=model(**x)
   original_losses=criterion(outputs,y)
   perturbed=copy.deepcopy(outputs)
   for name in ('pred_logits','pred_spans','saliency_scores'):
    perturbed[name][1]=1234
   for aux in perturbed.get('aux_outputs',[]):
    aux['pred_logits'][1]=1234;aux['pred_spans'][1]=1234
   changed_losses=criterion(perturbed,y)
   assert all(torch.equal(original_losses[k],changed_losses[k]) for k in original_losses),'Negative queries leaked into VMR'
   expected_bce=torch.nn.functional.binary_cross_entropy_with_logits(outputs['pred_exist_logits'],outputs['pred_exist_logits'].new_tensor([1.,0.]))
   assert torch.equal(original_losses['loss_exist'],expected_bce),'Existence BCE must see all samples'
   slots=torch.randn(2,10,o.hidden_dim,device=o.device)
   assert torch.equal(model.exist_head(slots),model.exist_head(slots.flip(1))),'Adapter must max pool final decoder slots'
  vi=[next(i for i,d in enumerate(val.data) if d['relevant_windows']),next(i for i,d in enumerate(val.data) if not d['relevant_windows'])]
  subset=torch.utils.data.Subset(val,vi);pred=predictions(model,loader(subset,o),o.device)
  for d,r in zip([val.data[i] for i in vi],pred):
   assert d['qid']==r['qid'] and len(r['pred_relevant_windows'])==o.num_queries
   assert all(0<=s<=e<=d['duration'] and 0<=c<=1 for s,e,c in r['pred_relevant_windows'])
  metrics=evaluate_gmr(pred,[val.data[i] for i in vi],map_num_workers=0,verbose=False)
  result={'status':'PASSED','strict_checkpoint_reload':'PASSED exact logits, spans, existence logits','existence_head_gradient':'PASSED finite and nonzero','positive_only_vmr':'PASSED mixed and all-negative batches','paper_mask_invariant':'PASSED negative-prediction perturbation leaves VMR unchanged; BCE includes both queries','max_pool_invariant':'PASSED slot permutation','mixed_loss':loss,'all_negative_loss':neg_loss,'full_batch_loss':full_loss,'losses':losses,'negative_losses':neg_losses,'parameters':sum(p.numel() for p in model.parameters()),'peak_vram_mib':torch.cuda.max_memory_allocated()/2**20,'reserved_vram_mib':torch.cuda.max_memory_reserved()/2**20,'training_count':len(train),'validation_count':len(val),'maximum_moments':maxmom,'validation_metrics':metrics,'positive_qid':train.data[pidx]['qid'],'negative_qid':train.data[nidx]['qid']}
  write_json(OUT/'smoke.json',result);print(json.dumps(result,indent=2),flush=True)
  if not a.smoke_then_train:return
  print('SMOKE_GATE_PASSED: restarting seed2026 scratch initialization for full GMR training',flush=True)
  # Reinitialize from scratch in this same scheduler-tracked PID after the smoke gate.
  del model,criterion,optimizer,scheduler,before,after,outputs,perturbed
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();o=options();seed_all(o.seed)
  model,criterion=build_model(o);model.to(o.device)
  optimizer=torch.optim.AdamW(model.parameters(),lr=o.lr,weight_decay=o.wd);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,o.lr_drop,gamma=.1)

 config=vars(o);write_json(OUT/'effective_config.json',config)
 import importlib.metadata,platform
 write_json(OUT/'environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}})
 provenance={'source':json.loads((ROOT/'official_snapshot/SOURCE.json').read_text()),'python':sys.executable,'torch':torch.__version__,'numpy':np.__version__,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'command':sys.argv,'seed_python_numpy_torch_dataloader':o.seed,'criterion':'max validation positive-query mAP; G-mIoU monitored at frozen threshold0.4; test held out until final checkpoint freeze','initialization':'scratch','native_architecture':'QD-DETR PReLU transformer + local paper-style max decoder-slot GMR MLP','published_qd_gmr_table2_row':False,'all_training_queries':len(train),'all_training_moments':sum(len(d['relevant_windows']) for d in train.data),'all_validation_queries':len(val),'maximum_moments':maxmom}
 provenance['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'gmr_deployment').rglob('*.py'))+[Path(__file__)]}
 write_json(OUT/'provenance.json',provenance)
 start=0;best=-float('inf');best_epoch=-1;loader_rng=None
 if a.resume:
  ck=torch.load(a.resume,map_location=o.device,weights_only=False);model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);start=ck['epoch']+1;best=ck['best'];best_epoch=ck['best_epoch'];random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in ck['rng']['cuda']]);loader_rng=ck['rng'].get('dataloader')
 tl=loader(train,o,True);vl=loader(val,o)
 if loader_rng is not None:tl.generator.set_state(loader_rng.cpu())
 print(f'TRAINING_STARTED QD-DETR-GMR seed={o.seed} epochs={o.n_epoch} bsz={o.bsz} train={len(train)} val={len(val)} device={o.device}',flush=True)
 for epoch in range(start,o.n_epoch):
  begin=time.time();total=0
  for step,items in enumerate(tl):
   loss,_=train_step(model,criterion,optimizer,items,o.device);total+=loss
   if step%20==0:print(json.dumps({'phase':'train','epoch':epoch,'step':step,'loss':loss,'peak_vram_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)
  scheduler.step();pred=predictions(model,vl,o.device);metrics=evaluate_gmr(pred,val.data,map_num_workers=0,verbose=False);score=metrics['brief']['mAP'];improved=score>best
  if improved:best=score;best_epoch=epoch
  record={'epoch':epoch,'train_loss':total/len(tl),'validation':metrics['brief'],'best_val_map':best,'best_epoch':best_epoch,'elapsed_sec':time.time()-begin}
  with (OUT/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True)
  ck={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'config':config,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'dataloader':tl.generator.get_state()}}
  torch.save(ck,OUT/'last.ckpt')
  if improved:
   torch.save(ck,OUT/'best.ckpt');write_json(OUT/'best_val_metrics.json',metrics)
   with (OUT/'best_val_predictions.jsonl').open('w') as f:
    for r in pred:f.write(json.dumps(r)+'\n')
  if epoch-best_epoch>=o.max_es_cnt:break
 write_json(OUT/'checkpoint_frozen.json',{'checkpoint':str(OUT/'best.ckpt'),'sha256':hashlib.sha256((OUT/'best.ckpt').read_bytes()).hexdigest(),'epoch':best_epoch,'validation_mAP':best,'threshold':.4})
 print('TRAINING_COMPLETED checkpoint frozen; starting one final test inference',flush=True)
 ck=torch.load(OUT/'best.ckpt',map_location=o.device,weights_only=False);model.load_state_dict(ck['model'],strict=True);test=Soccer('test');pred=predictions(model,loader(test,o),o.device)
 with (OUT/'test_predictions.jsonl').open('w') as f:
  for r in pred:f.write(json.dumps(r)+'\n')
 metrics=evaluate_gmr(pred,test.data,map_num_workers=0,verbose=True);write_json(OUT/'test_metrics.json',metrics)
if __name__=='__main__':main()
