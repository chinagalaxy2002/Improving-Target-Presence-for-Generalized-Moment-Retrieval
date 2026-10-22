"""CG-DETR-GMR requested extension; locally reconstructed paper existence adapter."""
import os,sys,json,random,time,argparse,subprocess,hashlib,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'gmr_source'))
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from cg_detr.config import BaseOptions
from cg_detr.model import build_model,SetCriterion
from cg_detr.span_utils import span_xx_to_cxw,span_cxw_to_xx
GMR=Path('/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval')
sys.path.insert(0,'/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared')
from official_eval.eval_main import evaluate_gmr
OUT=ROOT/'gmr'

def seed_all():
 random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def json_write(path,data):path.write_text(json.dumps(data,indent=2,default=str)+'\n')
def jsonl_write(path,data):path.write_text(''.join(json.dumps(x)+'\n' for x in data))

def config():
 b=BaseOptions();b.initialize();a=b.parser.parse_args([])
 a.dset_name='soccer_gmr';a.device=torch.device('cuda:0');a.v_feat_dim=2818;a.t_feat_dim=512
 a.max_v_l=75;a.max_q_l=32;a.bsz=32;a.eval_bsz=32;a.lr=3e-5;a.n_epoch=200;a.lr_drop=400
 a.seed=2026;a.max_es_cnt=200;a.eval_epoch=1;a.clip_length=2.;a.num_workers=0
 a.disable_synthetic_negatives=True;a.exist_loss_coef=1.0;a.gmr_threshold=0.4;a.variant="cg_detr_gmr_requested_extension"
 return a

class Soccer(Dataset):
 def __init__(self,split):
  self.path=GMR/'data/label/Standard'/f'{split}.jsonl'
  self.data=[json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]
 def __len__(self):return len(self.data)
 def __getitem__(self,i):
  d=self.data[i];feats=[]
  for name in ['slowfast','clip']:
   x=np.load(GMR/'Soccergmr'/name/f'{d["vid"]}.npz')['features'].astype('float32')
   assert len(x)<=75,(d['vid'],x.shape)
   x=x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-5);feats.append(x)
  n=min(map(len,feats));x=torch.from_numpy(np.concatenate([z[:n] for z in feats],-1))
  tef=torch.stack([torch.arange(n)/n,(torch.arange(n)+1)/n],-1)
  x=torch.cat([x,tef],-1)
  q=np.load(GMR/'Soccergmr/clip_text'/f'qid{d["qid"]}.npz')['last_hidden_state'][:32].astype('float32')
  q=torch.from_numpy(q/(np.linalg.norm(q,axis=-1,keepdims=True)+1e-5))
  wins=torch.tensor(d['relevant_windows'],dtype=torch.float32).reshape(-1,2)
  spans=span_xx_to_cxw(wins/float(d['duration']))
  labels=torch.zeros(n)
  for s,e in wins.tolist():labels[max(0,int(s/2)):min(n,max(int(s/2)+1,int(np.ceil(e/2))))]=1
  pos=torch.where(labels>0)[0].tolist();neg=torch.where(labels==0)[0].tolist()
  # Pair supervision only uses annotated foreground; natural negatives are handled separately.
  pi=random.choices(pos,k=2) if pos else [0,0]
  ni=random.choices(neg,k=2) if neg else pi
  return d,x,q,spans,labels,pi,ni

def collate(items):
 ds,x,q,spans,l,pi,ni=zip(*items)
 def pad(seq):
  y=torch.nn.utils.rnn.pad_sequence(seq,batch_first=True);m=torch.arange(y.shape[1])[None,:]<torch.tensor([len(z) for z in seq])[:,None];return y.cuda(),m.float().cuda()
 x,xm=pad(x);q,qm=pad(q);l,_=pad(l)
 # Passing vid=None explicitly disables synthetic query/video pairing without changing architecture.
 inputs=dict(src_vid=x,src_vid_mask=xm,src_txt=q,src_txt_mask=qm,vid=None,qid=[d['qid'] for d in ds])
 targets=dict(span_labels=[dict(spans=s.cuda()) for s in spans],saliency_all_labels=l,relevant_clips=l,saliency_pos_labels=torch.tensor(pi,device='cuda'),saliency_neg_labels=torch.tensor(ni,device='cuda'))
 return ds,inputs,targets

class SoccerCriterion(SetCriterion):
 def loss_spans(self,outputs,targets,indices):
  if sum(len(x[0]) for x in indices)==0:
   z=outputs['pred_spans'].sum()*0;return dict(loss_span=z,loss_giou=z)
  return super().loss_spans(outputs,targets,indices)
 def loss_labels(self,outputs,targets,indices,log=True):
  return super().loss_labels(outputs,targets,indices,log=False)
 def positive_view(self,outputs,targets):
  idx=torch.tensor([len(x['spans'])>0 for x in targets['span_labels']],device='cuda')
  b=len(idx)
  o={k:(v[idx] if torch.is_tensor(v) and v.ndim and v.shape[0]==b and k not in ['global_rep_tokens'] else v) for k,v in outputs.items()}
  t={k:([v[i] for i in range(b) if idx[i]] if k=='span_labels' else v[idx]) for k,v in targets.items()}
  return idx,o,t
 def loss_saliency(self,outputs,targets,indices,log=True):
  idx,o,t=self.positive_view(outputs,targets)
  loss=outputs['saliency_scores'].sum()*0
  if idx.any():
   # Upstream BCE expects unpadded attention probabilities. Evaluate each positive
   # at its real video length so padding cannot become a negative BCE input.
   pieces=[]
   for j in range(int(idx.sum())):
    n=int(o['video_mask'][j].sum())
    one={k:(v[j:j+1,:n] if k in ['saliency_scores','video_mask','t2vattnvalues'] else v) for k,v in o.items()}
    ot={k:(v[j:j+1,:n] if k in ['saliency_all_labels','relevant_clips'] else v[j:j+1] if torch.is_tensor(v) else v) for k,v in t.items()}
    pieces.append(super().loss_saliency(one,ot,None)['loss_saliency'])
   loss=torch.stack(pieces).mean()
  return dict(loss_saliency=loss)
 def loss_contrastive_moment_sentence(self,outputs,targets,indices,log=True):
  idx,o,t=self.positive_view(outputs,targets)
  if not idx.any():return dict(loss_ms_align=outputs['pred_logits'].sum()*0)
  return super().loss_contrastive_moment_sentence(o,t,None)
 def loss_moment2txt_sim_distill(self,outputs,targets,indices,log=True):
  idx,o,t=self.positive_view(outputs,targets)
  if not idx.any():return dict(loss_distill=outputs['pred_logits'].sum()*0)
  return super().loss_moment2txt_sim_distill(o,t,None)

class GMRCriterion(SoccerCriterion):
 def forward(self,outputs,targets):
  positive=torch.tensor([len(x['spans'])>0 for x in targets['span_labels']],device=outputs['pred_logits'].device)
  # BCE sees every naturally positive and null query. Native VMR and every
  # auxiliary decoder loss see positive queries only, following the paper.
  existence=F.binary_cross_entropy_with_logits(outputs['pred_exist_logit'],positive.float())
  if positive.any():
   b=len(positive)
   def subset(o):
    return {k:([subset(vv) for vv in v] if k=='aux_outputs' else v[positive] if torch.is_tensor(v) and v.ndim and v.shape[0]==b and k!='global_rep_tokens' else v) for k,v in o.items()}
   ot={k:([v[i] for i in range(b) if positive[i]] if k=='span_labels' else v[positive]) for k,v in targets.items()}
   native=super().forward(subset(outputs),ot)
  else:
   zero=outputs['pred_logits'].sum()*0
   native={k:zero for k in self.weight_dict if k!='loss_exist'}
  native['loss_exist']=existence
  return native

def setup():
 a=config();model,criterion=build_model(a);criterion.__class__=GMRCriterion
 criterion.weight_dict['loss_exist']=a.exist_loss_coef
 model=model.cuda();return a,model,criterion

def infer(model,loader):
 model.eval();pred=[]
 with torch.no_grad():
  for ds,inputs,t in loader:
   o=model(**inputs);scores=o['pred_logits'].softmax(-1)[...,0];spans=span_cxw_to_xx(o['pred_spans']).clamp(0,1)
   for d,ss,ww,exist in zip(ds,scores,spans,o['pred_exist_logit'].sigmoid()):
    order=ss.argsort(descending=True);times=(torch.round(ww[order]*d['duration']/2)*2).clamp(0,d['duration']);wins=torch.cat([times,ss[order,None]],-1).cpu().tolist()
    # Keep native proposal confidence for localization; explicit existence is gated by the official evaluator.
    pred.append(dict(qid=d['qid'],query=d['query'],vid=d['vid'],pred_relevant_windows=wins,pred_saliency_scores=[],pred_exist_score=float(exist)))
 expected=loader.dataset
 expected_qids={expected.dataset.data[i]['qid'] for i in expected.indices} if isinstance(expected,torch.utils.data.Subset) else {d['qid'] for d in expected.data}
 assert {d['qid'] for d in pred}==expected_qids,'Prediction qid coverage differs from split'
 assert len(pred)==len(loader.dataset),'Prediction count differs from dataset'
 assert len({d['qid'] for d in pred})==len(pred),'Duplicate predicted qid'
 return pred

def train_step(model,c,optimizer,batch,check_null_mask=False):
 _,inputs,t=batch;model.train();optimizer.zero_grad(set_to_none=True)
 o=model(**inputs,targets=t);losses=c(o,t);loss=sum(v*c.weight_dict[k] for k,v in losses.items() if k in c.weight_dict)
 if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {losses}')
 if check_null_mask:
  native=sum(v*c.weight_dict[k] for k,v in losses.items() if k in c.weight_dict and k!='loss_exist')
  # Check every batch-dependent VMR output used by native and auxiliary losses.
  tensors=[o[k] for k in ['pred_logits','pred_spans','saliency_scores','t2vattnvalues','memory_moment','nmmemory_moment','sentence_txt','sentence_dummy','moment2txt_similarity','cate_attn_weights','src_vid'] if torch.is_tensor(o.get(k)) and o[k].requires_grad]
  tensors += [a[k] for a in o.get('aux_outputs',[]) for k in ['pred_logits','pred_spans']]
  derivatives=torch.autograd.grad(native,tensors,retain_graph=True,allow_unused=True)
  null=torch.tensor([not len(x['spans']) for x in t['span_labels']],device='cuda')
  for output,derivative in zip(tensors,derivatives):
   if derivative is not None and output.shape[0]==len(null):
    assert torch.count_nonzero(derivative[null])==0,'Native VMR gradient reached natural null query'
  losses['null_vmr_gradient_max']=0.0
 loss.backward()
 grads=[p.grad for p in model.parameters() if p.grad is not None]
 assert grads and all(torch.isfinite(x).all() for x in grads),'Nonfinite gradients'
 head_grads=[p.grad for p in model.exist_head.parameters()]
 assert all(g is not None and torch.isfinite(g).all() for g in head_grads),'Missing/nonfinite existence head gradients'
 head_norm=sum(float(g.square().sum()) for g in head_grads)**0.5
 assert head_norm>0,'Zero existence head gradient'
 losses['exist_head_gradient_norm']=head_norm
 grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),c.args.grad_clip));optimizer.step()
 return float(loss.detach()),grad,{k:float(v) for k,v in losses.items()}

def smoke():
 seed_all();OUT.mkdir(parents=True,exist_ok=True);a,m,c=setup();tr=Soccer('train');val=Soccer('val')
 p=next(i for i,d in enumerate(tr.data) if d['relevant_windows']);n=next(i for i,d in enumerate(tr.data) if not d['relevant_windows'])
 optimizer=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.wd);torch.cuda.reset_peak_memory_stats();report={}
 short=next(i for i,d in enumerate(tr.data) if d['duration']<150 and d['relevant_windows']);multi=next(i for i,d in enumerate(tr.data) if len(d['relevant_windows'])>1)
 for name,ids in [('positive',[p]),('negative',[n]),('mixed_batch32',[p,n,short,multi]*8)]:
  loss,grad,losses=train_step(m,c,optimizer,collate([tr[i] for i in ids]),check_null_mask=True);report[name]=dict(loss=loss,gradient_norm=grad,losses=losses)
 selected=[next(i for i,d in enumerate(val.data) if d['relevant_windows']),next(i for i,d in enumerate(val.data) if not d['relevant_windows'])]
 loader=DataLoader(torch.utils.data.Subset(val,selected),batch_size=2,collate_fn=collate)
 pred=infer(m,loader);gt=[val.data[i] for i in selected]
 assert len(pred)==2 and all(len(x['pred_relevant_windows'])==10 and 0<=x['pred_exist_score']<=1 for x in pred)
 assert report['negative']['losses']['loss_span']==0 and report['negative']['losses']['loss_label']==0
 assert all(x['losses']['exist_head_gradient_norm']>0 for x in report.values())
 met=evaluate_gmr(pred,gt,map_num_workers=1,verbose=False)
 assert all(x['losses']['null_vmr_gradient_max']==0 for x in report.values())
 report.update(peak_memory_mib=torch.cuda.max_memory_allocated()/1024**2,validation_metrics=met,positive_qid=tr.data[p]['qid'],negative_qid=tr.data[n]['qid'])
 torch.save(dict(model=m.state_dict()),OUT/'smoke_reload.pt');m.load_state_dict(torch.load(OUT/'smoke_reload.pt',weights_only=False)['model'],strict=True)
 json_write(OUT/'smoke.json',report);jsonl_write(OUT/'smoke_val_predictions.jsonl',pred)
 print('GMR_SMOKE_PASSED',json.dumps(report),flush=True)

def train():
 seed_all();OUT.mkdir(parents=True,exist_ok=True);a,m,c=setup();tr=Soccer('train');val=Soccer('val')
 gen=torch.Generator().manual_seed(2026)
 loader=DataLoader(tr,batch_size=a.bsz,shuffle=True,collate_fn=collate,num_workers=0,generator=gen)
 vl=DataLoader(val,batch_size=a.eval_bsz,collate_fn=collate,num_workers=0)
 optimizer=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.wd);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,a.lr_drop)
 cfg=vars(a).copy();cfg.update(augmentation=dict(qcd=False,query_swap=False,background_replacement=False,synthetic_negatives=False,text_drop=0),checkpoint_selection='validation positive-query mAP, official GMR evaluator',feature_order=['slowfast2304','clip512','tef2'],all_natural_negatives=True,all_annotated_moments=True,gmr_adapter='max(last_decoder_slots)->Linear256/ReLU/Linear1',native_vmr_losses='positive-only including auxiliary decoder losses',existence_loss='BCEWithLogits all natural queries coef1',reconstruction_status='local extension; no published CG-DETR-GMR Table2 target')
 json_write(OUT/'effective_config.json',cfg)
 json_write(OUT/'provenance.json',dict(source_repository='https://github.com/wjun0830/CGDETR',source_commit=json.loads((ROOT/'gmr_source/SOURCE_PROVENANCE.json').read_text())['commit'],runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),model_sha256=hashlib.sha256((ROOT/'gmr_source/cg_detr/model.py').read_bytes()).hexdigest(),python=sys.version,torch=torch.__version__,numpy=np.__version__,command=sys.argv,cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),seed=2026,train_samples=len(tr),val_samples=len(val),test_during_training=False))
 best=-1.;stale=0;start=0
 if (OUT/'latest.pt').exists():
  ck=torch.load(OUT/'latest.pt',weights_only=False);m.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);start=ck['epoch']+1;best=ck['best'];stale=ck['stale'];random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);gen.set_state(ck['loader_rng'])
 print(f'TRAINING_STARTED variant=cg_detr_gmr train={len(tr)} val={len(val)} epoch={start} seed=2026',flush=True)
 for epoch in range(start,a.n_epoch):
  vals=[];begin=time.time()
  for step,batch in enumerate(loader):
   loss,grad,_=train_step(m,c,optimizer,batch);vals.append(loss)
   if step%25==0:print(f'epoch={epoch+1} step={step}/{len(loader)} loss={loss:.6f} gradient={grad:.6f}',flush=True)
  scheduler.step();pred=infer(m,vl);metrics=evaluate_gmr(pred,val.data,map_num_workers=1,verbose=False);score=metrics['brief']['mAP'];improved=score>best
  if improved:best=score;stale=0
  else:stale+=1
  ck=dict(model=m.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,best=best,stale=stale,config=cfg,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),loader_rng=gen.get_state())
  torch.save(ck,OUT/'latest.tmp');os.replace(OUT/'latest.tmp',OUT/'latest.pt')
  if improved:
   torch.save(ck,OUT/'best.tmp');os.replace(OUT/'best.tmp',OUT/'best.pt');jsonl_write(OUT/'best_val_predictions.jsonl',pred);json_write(OUT/'best_val_metrics.json',metrics)
  entry=dict(epoch=epoch+1,loss=np.mean(vals),validation=metrics['brief'],best_val_map=best,seconds=time.time()-begin)
  with (OUT/'history.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
  json_write(OUT/'status.json',entry);print(json.dumps(entry),flush=True)
  if stale>a.max_es_cnt:break
 # Freeze selection before loading test data; no test data are touched during optimization.
 ck=torch.load(OUT/'best.pt',weights_only=False);m.load_state_dict(ck['model'],strict=True)
 json_write(OUT/'frozen_checkpoint.json',dict(path=str(OUT/'best.pt'),epoch=ck['epoch']+1,sha256=hashlib.sha256((OUT/'best.pt').read_bytes()).hexdigest(),threshold=.4,selection='validation positive-query mAP',variant='cg_detr_gmr_requested_extension'))
 test=Soccer('test');pred=infer(m,DataLoader(test,batch_size=a.eval_bsz,collate_fn=collate));jsonl_write(OUT/'final_test_predictions.jsonl',pred);json_write(OUT/'final_test_metrics.json',evaluate_gmr(pred,test.data,map_num_workers=1,verbose=False))
 print('GMR_TRAINING_COMPLETED',flush=True)

if __name__=='__main__':
 torch.set_num_threads(4)
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-then-train',action='store_true');args=p.parse_args()
 if args.smoke or args.smoke_then_train:smoke()
 if args.smoke_then_train:
  # Replace the process so smoke GPU allocations/RNG cannot leak into training.
  os.execv(sys.executable,[sys.executable,'-u',str(Path(__file__).resolve())])
 elif not args.smoke:train()
