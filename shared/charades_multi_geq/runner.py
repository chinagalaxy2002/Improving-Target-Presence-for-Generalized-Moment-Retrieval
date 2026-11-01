"""Single task worker. Only Val selects checkpoints; tests are terminal exports."""
import argparse,fcntl,os,signal,time,copy,sys,math
from pathlib import Path
# No module named queue in this directory.
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from paths import *
from engine_utils import seed,save,rng,restore,state_hash,seed_worker
from adapter import Adapter
from geq_model import RouteGEQ
from metrics import measure
from validation_schedule import ensure_schedule,validation_due,advance_selection
STOP=False

def stop(signum,frame):
 global STOP;STOP=True

def loader(ad,split,training=False,generator=None):
 return DataLoader(ad.dataset(split),batch_size=ad.bs if training else (1 if ad.name=='flashvtg' else ad.bs),shuffle=training,collate_fn=list,num_workers=2,pin_memory=False,generator=generator,worker_init_fn=seed_worker,persistent_workers=False)

def build(backbone,method,auxiliary_query_scope=None):
 seed();ad=Adapter(backbone,method.startswith('multi'),auxiliary_query_scope=auxiliary_query_scope)
 base=ad.base;model=base;parent=None
 if '_geq_' in method:
  parent=OUT/'runs'/backbone/('multi' if method.startswith('multi') else 'gmr')
  state=torch.load(parent/'best.ckpt',map_location='cpu',weights_only=False)
  assert state['train_sha256']==sha(OUT/'data/train.jsonl') and state['seed']==2023
  base.load_state_dict(state['model'],strict=True)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(2023);model=RouteGEQ(base,backbone,method.rsplit('_',1)[1]).cuda()
  init=OUT/'runs'/backbone/'geq_init.pt'
  with init.with_suffix('.lock').open('a') as initialization_lock:
   fcntl.flock(initialization_lock,fcntl.LOCK_EX)
   if init.exists():model.geq.load_state_dict(torch.load(init,map_location='cpu',weights_only=False)['model'],strict=True)
   else:save(init,dict(model=model.geq.state_dict(),seed=2023,sha256=state_hash(model.geq)))
 groups=[dict(params=[p for p in base.parameters() if p.requires_grad],lr=1e-5 if parent else 3e-5)]
 if parent:groups.append(dict(params=list(model.geq.parameters()),lr=3e-5))
 optimizer=torch.optim.AdamW([g for g in groups if len(list(g['params']))],weight_decay=1e-4)
 return ad,model,optimizer,parent

def loss(ad,model,out,targets,metas,epoch):
 if isinstance(model,RouteGEQ):
  labels=targets.get('exist_label')
  if labels is None:labels=torch.tensor([bool(m['relevant_windows']) for m in metas],device='cuda')
  value=F.binary_cross_entropy_with_logits(out['pred_exist_logits'].reshape(-1),labels.float().reshape(-1))
  if model.mode=='full':value=value+ad.loss(out['native_output'],targets,metas,epoch)
  return value
 return ad.loss(out,targets,metas,epoch)

def export(ad,out,metas):
 native=out.get('native_output',out);result=[]
 for i,m in enumerate(metas):
  if ad.name=='flashvtg':
   windows=native['_out']['boundary'].detach().clone()
  else:
   center,width=native['pred_spans'][i].unbind(-1)
   xy=torch.stack((center-width/2,center+width/2),-1)*float(m['duration'])
   windows=torch.cat((xy,native['pred_logits'][i].softmax(-1)[:,:1]),-1)
  windows=ad.native.export_ranked_proposals(windows,m,apply_pat=False)
  logit=out['pred_exist_logits'].reshape(-1)[i]
  result.append(dict(qid=m['evaluation_id'],evaluation_id=m['evaluation_id'],original_qid=m['original_qid'],vid=m['vid'],query=m['query'],source_row_index=m['source_row_index'],pred_relevant_windows=windows,pred_exist_score=float(logit.sigmoid())))
 return result

def evaluate(ad,model,split,output,parent=None):
 model.eval();pred=[];begun=time.time();torch.cuda.synchronize();batches=loader(ad,split)
 fixed=None
 if isinstance(model,RouteGEQ) and model.mode=='frozen':
  fixed_path=parent/('best_val_predictions.jsonl' if split=='val' else f'{split}/predictions.jsonl')
  fixed=rows(fixed_path)
  parent_marker=json.loads((parent/'TRAINED.json').read_text())
  assert sha(parent/'best.ckpt')==parent_marker['checkpoint_sha256']
  parent_state=torch.load(parent/'best.ckpt',map_location='cpu',weights_only=False)['model']
  actual=model.base.state_dict()
  assert actual.keys()==parent_state.keys()
  assert all(torch.equal(v.detach().cpu(),parent_state[k]) for k,v in actual.items()),'Frozen trunk differs from selected parent'
  del parent_state,actual
 with torch.no_grad():
  for items in batches:
   metas,inputs,targets=ad.collate(items);out=model(**inputs)
   if fixed is None:
    values=export(ad,out,metas)
   else:
    # The frozen experiment uses the parent's saved candidate export as its
    # authoritative localization output; only the presence score is replaced.
    values=[]
    for i,meta in enumerate(metas):
     old=fixed[len(pred)+i]
     assert old['qid']==meta['evaluation_id'] and old['vid']==meta['vid'] and old['original_qid']==meta['original_qid']
     p=copy.deepcopy(old);p['pred_exist_score']=float(out['pred_exist_logits'].reshape(-1)[i].sigmoid())
     assert p['pred_relevant_windows']==old['pred_relevant_windows']
     values.append(p)
   pred.extend(values)
 torch.cuda.synchronize();seconds=time.time()-begun
 metrics=measure(pred,batches.dataset.data);metrics.update(inference_seconds=seconds,inference_ms_per_row=seconds*1000/len(pred),checkpoint_selection_split='val')
 if fixed is not None:
  old_metrics=json.loads((parent/('best_val_metrics.json' if split=='val' else f'{split}/metrics.json')).read_text())
  assert metrics['mAP']==old_metrics['mAP'];metrics['frozen_candidates_identical']=True
  assert [p['pred_relevant_windows'] for p in pred]==[p['pred_relevant_windows'] for p in fixed]
  metrics['parent_prediction_sha256']=sha(fixed_path);metrics['parent_checkpoint_sha256']=parent_marker['checkpoint_sha256']
  metrics['frozen_parent_tensors_bitwise_equal']=True
 jsonl(output/'predictions.jsonl',pred);dump(output/'metrics.json',metrics)
 return metrics,pred

def preflight(ad,model,optimizer,method,outdir):
 """All changes are restored, including RNG, optimizer and frozen buffers."""
 initial=copy.deepcopy(model.state_dict());initial_buffers={k:v.detach().clone() for k,v in model.named_buffers()};initial_criterion=copy.deepcopy(ad.native.criterion.state_dict());initial_rng=rng();initial_opt=copy.deepcopy(optimizer.state_dict());checks={}
 ds=ad.dataset('train');data=ds.data
 pos=next(i for i,r in enumerate(data) if r['relevant_windows']);neg=next(i for i,r in enumerate(data) if not r['relevant_windows'])
 longest=max(range(len(data)),key=lambda i:data[i]['duration'])
 tail=next(i for i,r in enumerate(data) if any(w[1]>2*json.loads((OUT/'data/audit.json').read_text())['video_lengths'][r['vid']] for w in r['relevant_windows']))
 cases={'mixed':[pos,neg],'all_null':[neg,neg],'longest_tail':[longest,tail],'physical_batch':[pos,neg]*(ad.bs//2)}
 # Real OOD negative input uses the training-independent source, no model selection.
 ood=ad.dataset('test_ood');oi=next(i for i,r in enumerate(ood.data) if not r['relevant_windows'])
 for case,indices in cases.items():
  model.train();optimizer.zero_grad(set_to_none=True);metas,inputs,targets=ad.collate([ds[i] for i in indices]);prediction=model(**inputs)
  value=loss(ad,model,prediction,targets,metas,1);assert torch.isfinite(value);value.backward()
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  if case=='mixed':
   active=[n for n,p in model.named_parameters() if p.grad is not None and bool(p.grad.abs().sum()>0)]
   if isinstance(model,RouteGEQ):
    assert any(n.startswith('geq.') for n in active)
    if model.mode=='full':
     assert any('transformer.encoder' in n for n in active)
     assert any('exist_head' in n for n in active),'Original existence BCE must remain active'
    else:assert not any(n.startswith('base.') for n in active)
   if method.startswith('multi') and not method.endswith('frozen'):
    assert any(any(k in n for k in ('ffns','private')) for n in active),active
    assert any('instructions' in n for n in active)
  optimizer.step()
  if isinstance(model,RouteGEQ) and case=='mixed':assert any(not torch.equal(v,model.state_dict()[k]) for k,v in initial.items() if k.startswith('geq.')),'GEQ optimizer must change GEQ parameters'
  checks[case]=dict(loss=float(value),finite=True)
 model.train();metas,inputs,targets=ad.collate([ood[oi]]);v=loss(ad,model,model(**inputs),targets,metas,1);assert torch.isfinite(v);checks['ood_negative']=float(v)
 if isinstance(model,RouteGEQ) and model.mode=='frozen':
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items() if k.startswith('base.'))
  assert all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in initial_buffers.items() if k.startswith('base.'))
  checks['frozen_parameters_buffers_unchanged']=True
 aux=model.base if isinstance(model,RouteGEQ) else model
 calls=[];handles=[]
 for name,module in aux.named_modules():
  if name.startswith(('ffns.','layers.')) or name.endswith('.private'):
   handles.append(module.register_forward_hook(lambda *args:calls.append(True)))
 model.eval();metas,inputs,targets=ad.collate([ds[pos]]);output=model(**inputs);assert 'o2m_outputs' not in output.get('native_output',output)
 assert not calls,'Auxiliary modules executed during eval'
 for handle in handles:handle.remove()
 checks['eval_auxiliary_forward_calls']=len(calls);checks['eval_has_no_auxiliary_outputs']=True
 eval_predictions=[];eval_gt=[]
 with torch.no_grad():
  for index in (pos,neg):
   meta,inp,tgt=ad.collate([ds[index]]);prediction=model(**inp);values=export(ad,prediction,meta)
   if isinstance(model,RouteGEQ) and model.mode=='frozen':
    native=ad.base(**inp);reference=export(ad,native,meta);assert values[0]['pred_relevant_windows']==reference[0]['pred_relevant_windows']
   eval_predictions.extend(values);eval_gt.extend(meta)
 eval_metrics=measure(eval_predictions,eval_gt);checks['native_export_and_metrics']=dict(rows=2,mAP=eval_metrics['mAP'],finite=True)
 checks['geq_parameters_changed']=isinstance(model,RouteGEQ)

 model.load_state_dict(initial,strict=True)
 for k,v in model.named_buffers():v.copy_(initial_buffers[k])
 ad.native.criterion.load_state_dict(initial_criterion,strict=True)
 optimizer.load_state_dict(initial_opt);restore(initial_rng);optimizer.zero_grad(set_to_none=True)
 # Strict serialized reload; interrupted-process resume is separately checked by the queue.
 save(outdir/'preflight_reload.pt',dict(model=initial,optimizer=initial_opt,rng=initial_rng))
 loaded=torch.load(outdir/'preflight_reload.pt',map_location='cpu',weights_only=False);model.load_state_dict(loaded['model'],strict=True)
 (outdir/'preflight_reload.pt').unlink();checks['strict_reload']=True;checks['initial_state_restored']=all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items())
 dump(outdir/'preflight.json',dict(passed=True,checks=checks))

def checkpoint(model,optimizer,epoch,cursor,generator_state,best,best_epoch,stale,best_above,train_sha,schedule=None):
 state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng(),sampler_epoch_start=generator_state,epoch=epoch,cursor=cursor,best=best,best_epoch=best_epoch,stale=stale,best_above=best_above,train_sha256=train_sha,seed=2023)
 if schedule is not None:state['validation_schedule']=schedule
 return state

def train(ad,model,optimizer,parent,outdir,args):
 train_sha=sha(OUT/'data/train.jsonl');generator=torch.Generator().manual_seed(2023)
 epoch=1;cursor=0;best=-math.inf;best_epoch=0;stale=0;best_above=False;resumed=None
 if (outdir/'last.ckpt').exists():
  resumed=torch.load(outdir/'last.ckpt',map_location='cpu',weights_only=False);assert resumed['train_sha256']==train_sha
  model.load_state_dict(resumed['model'],strict=True);optimizer.load_state_dict(resumed['optimizer']);restore(resumed['rng'])
  epoch=resumed['epoch'];cursor=resumed['cursor'];best=resumed['best'];best_epoch=resumed['best_epoch'];stale=resumed['stale'];best_above=resumed['best_above'];generator.set_state(resumed['sampler_epoch_start'])
 else:preflight(ad,model,optimizer,args.method,outdir)
 schedule=ensure_schedule(outdir,bool(parent),epoch)
 if schedule.get('defer_early_stop_until_epoch'):
  request=json.loads((outdir/'USER_CONTINUE_REQUEST.json').read_text())
  assert (ad.name,args.method)==('eatr','multi') and not parent
  assert request['authorized_by']=='user' and request['instruction']=='可以恢复原始全查询multi,继续训练。继续排队。'
  assert schedule['user_continue_request_sha256']==sha(outdir/'USER_CONTINUE_REQUEST.json')
  assert schedule['defer_early_stop_until_epoch']==request['defer_early_stop_until_epoch']==53
 if resumed is not None and 'validation_schedule' in resumed:assert resumed['validation_schedule']==schedule
 cfg=json.loads((outdir/'config.json').read_text());cfg['validation_schedule']=schedule;dump(outdir/'config.json',cfg)
 if resumed is not None:
  print('TRAIN_RESUME',json.dumps(dict(epoch=epoch,cursor=cursor,best_epoch=best_epoch,stale=stale,checkpoint_sha256=sha(outdir/'last.ckpt'),validation_interval_epochs=schedule['interval_epochs'],strict_model_optimizer_rng_sampler_restore=True)),flush=True)
 baseline=None;parent_values=None
 if parent:
  baseline=json.loads((OUT/'runs'/ad.name/'gmr/best_val_metrics.json').read_text())['selection_values']
  parent_values=json.loads((parent/'best_val_metrics.json').read_text())['selection_values']
 ceiling=20 if parent else 200
 for ep in range(epoch,ceiling+1):
  if stale>=schedule['patience_epochs'] and ep>schedule.get('defer_early_stop_until_epoch',0):break
  epoch_generator=generator.get_state();batches=loader(ad,'train',True,generator);model.train();begun=time.time();total=0.;steps=0
  torch.cuda.reset_peak_memory_stats()
  for step,items in enumerate(batches):
   if step<cursor:continue
   if resumed is not None:restore(resumed['rng']);resumed=None
   metas,inputs,targets=ad.collate(items);optimizer.zero_grad(set_to_none=True);value=loss(ad,model,model(**inputs),targets,metas,ep)
   assert torch.isfinite(value),f'Nonfinite loss epoch {ep} step {step}'
   value.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],.1);assert torch.isfinite(norm)
   optimizer.step();total+=float(value);steps+=1
   if (step+1)%100==0 or STOP or args.interrupt_after_steps and steps>=args.interrupt_after_steps:
    save(outdir/'last.ckpt',checkpoint(model,optimizer,ep,step+1,epoch_generator,best,best_epoch,stale,best_above,train_sha,schedule))
    diagnostics={'last_batch_loss_parts':{k:float(v) for k,v in getattr(ad,'last_loss_components',{}).items()}} if ad.name=='eatr' else {}
    dump(outdir/'status.json',dict(status='TRAINING',epoch=ep,step=step+1,total_steps=len(batches),pid=os.getpid(),train_loss=total/max(steps,1),**diagnostics))
    print(json.dumps(dict(epoch=ep,step=step+1,loss=total/max(steps,1),**diagnostics)),flush=True)
   if STOP or args.interrupt_after_steps and steps>=args.interrupt_after_steps:return
  cursor=0
  validated=validation_due(ep,best_epoch,ceiling,schedule);metrics=None;pred=None;score=None;above=False
  if validated:
   metrics,pred=evaluate(ad,model,'val',outdir/'current_val',parent)
   score=metrics['selection_mean'] if parent else metrics['mAP'];above=bool(parent) and all(a>b for a,b in zip(metrics['selection_values'],baseline))
  best,best_epoch,best_above,improved,tie_better=advance_selection(ep,score,above,best,best_epoch,best_above)
  stale=0 if improved else stale+1 if math.isfinite(best) else 0
  # At an epoch boundary, generator now stores next epoch's deterministic sampler state.
  state=checkpoint(model,optimizer,ep+1,0,generator.get_state(),best,best_epoch,stale,best_above,train_sha,schedule)
  save(outdir/'last.ckpt',state)
  if improved or tie_better:
   save(outdir/'best.ckpt',state);jsonl(outdir/'best_val_predictions.jsonl',pred);dump(outdir/'best_val_metrics.json',metrics)
  record=dict(epoch=ep,validated=validated,validation_interval_epochs=schedule['interval_epochs'],selection=score,best=best if math.isfinite(best) else None,best_epoch=best_epoch,stale=stale,seconds=time.time()-begun,train_loss=total/max(steps,1),peak_memory_bytes=torch.cuda.max_memory_allocated(),mAP=metrics['mAP'] if validated else None,val_values=metrics['selection_values'] if validated else None,gain_vs_gmr=[a-b for a,b in zip(metrics['selection_values'],baseline)] if parent and validated else None,gain_vs_parent=[a-b for a,b in zip(metrics['selection_values'],parent_values)] if parent and validated else None)
  with (outdir/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
  dump(outdir/'status.json',dict(status='TRAINING',pid=os.getpid(),**record));print('EPOCH',json.dumps(record),flush=True)
  if STOP:return
 assert (outdir/'best.ckpt').is_file()
 dump(outdir/'TRAINED.json',dict(best_epoch=best_epoch,score=best,checkpoint_sha256=sha(outdir/'best.ckpt'),selection='Val mean G-mIoU@1 at .4/.5/.6' if parent else 'Val positive mAP',seed=2023,validation_schedule_sha256=sha(outdir/'validation_schedule.json')))

def main():
 p=argparse.ArgumentParser();p.add_argument('--backbone',choices=BACKBONES,required=True);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--mode',choices=['train','preflight','test','profile'],default='train');p.add_argument('--split',choices=['test_id','test_ood']);p.add_argument('--interrupt-after-steps',type=int,default=0);p.add_argument('--output-suffix',default='');p.add_argument('--auxiliary-query-scope',choices=['all_queries','positive_only_v1']);args=p.parse_args()
 if args.auxiliary_query_scope=='positive_only_v1':assert (args.backbone,args.method,args.output_suffix)==('eatr','multi','_positive_only'),'Positive-only ablation must use its independent directory'
 torch.set_num_threads(4);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 outdir=OUT/'runs'/args.backbone/(args.method+args.output_suffix);outdir.mkdir(parents=True,exist_ok=True)
 with (outdir/'task.lock').open('a') as lock, (outdir/'.lock').open('a') as legacy_lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  fcntl.flock(legacy_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  ad,model,optimizer,parent=build(args.backbone,args.method,args.auxiliary_query_scope)
  dump(outdir/'config.json',dict(backbone=args.backbone,method=args.method,seed=2023,train_sha256=sha(OUT/'data/train.jsonl'),native=ad.native.cfg,initialization=ad.native.manifest,parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),parent_sha256=sha(parent/'best.ckpt') if parent else None,native_init_snapshot_sha256=sha(OUT/'runs'/args.backbone/'native_init.pt'),geq_init_snapshot_sha256=sha(OUT/'runs'/args.backbone/'geq_init.pt') if parent else None,amp=False,microbatch=ad.bs,accumulation=1))
  if (outdir/'validation_schedule.json').exists():
   cfg=json.loads((outdir/'config.json').read_text());cfg['validation_schedule']=json.loads((outdir/'validation_schedule.json').read_text());dump(outdir/'config.json',cfg)
  if args.mode=='preflight':preflight(ad,model,optimizer,args.method,outdir)
  elif args.mode=='train':
   if not (outdir/'TRAINED.json').exists():train(ad,model,optimizer,parent,outdir,args)
  elif args.mode=='profile':
   assert (outdir/'TRAINED.json').exists()
   if not (outdir/'PROFILE.json').exists():
    from profile_inference import profile
    profile(ad,model,outdir)
  else:
   assert args.split and (outdir/'TRAINED.json').exists();dest=outdir/args.split
   if (dest/'COMPLETE.json').exists():return
   state=torch.load(outdir/'best.ckpt',map_location='cpu',weights_only=False);model.load_state_dict(state['model'],strict=True)
   metrics,pred=evaluate(ad,model,args.split,dest,parent);assert len(pred)==7440
   dump(dest/'COMPLETE.json',dict(checkpoint_sha256=sha(outdir/'best.ckpt'),rows=len(pred),prediction_sha256=sha(dest/'predictions.jsonl'),metrics_sha256=sha(dest/'metrics.json')))
if __name__=='__main__':main()
