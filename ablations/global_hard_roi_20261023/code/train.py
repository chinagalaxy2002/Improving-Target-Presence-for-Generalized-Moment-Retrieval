"""Fresh two-model Train/Val entry. Does not run Test or resume archived state."""
import argparse,time
from pathlib import Path
from environment import ROOT,read_json,write_json,write_rows,seed_all,make_dataset
from api import build_model,job_for
import torch
from torch.utils.data import DataLoader
from training.flash_vtg_gmr.dataset import start_end_collate,prepare_batch_inputs
from models.flash_vtg_gmr.standalone_eval.eval import eval_submission
from prediction import predictions
from evaluation import Evaluation
from runtime import selection_score,training_mode
from model import compute_mcev_loss

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['E0','S4Fusion'],required=True);p.add_argument('--run-name',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 directory=(ROOT/'runs'/a.run_name).resolve();assert directory.is_relative_to((ROOT/'runs').resolve()),'Output must be inside this package runs/'
 if directory.exists():raise FileExistsError('Fresh run required; refusing to overwrite or resume')
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 job=job_for(a.model);model,opt,checks,criterion=build_model(a.model,a.device,initialize=True)
 frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
 active=[p for p in model.parameters() if p.requires_grad]
 groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and '.lora_' not in n],'lr':job['lr']}]
 adapters=[p for n,p in model.named_parameters() if p.requires_grad and '.lora_' in n]
 if adapters:groups.append({'params':adapters,'lr':job['lora_lr']})
 optimizer=torch.optim.AdamW(groups,weight_decay=job['wd'],foreach=False);scheduler=torch.optim.lr_scheduler.StepLR(optimizer,job['lr_drop'],gamma=.5)
 seed_all(2026);train,val=make_dataset(opt,'train'),make_dataset(opt,'val');assert len(train)==4138 and len(val)==465
 loader=DataLoader(train,batch_size=opt.bsz,shuffle=True,num_workers=0,collate_fn=start_end_collate,generator=torch.Generator().manual_seed(2026))
 seed_all(2026 if a.model=='E0' else 102026)
 directory.mkdir(parents=True);write_json(directory/'CONFIG.json',dict(model=a.model,job=job,train_count=len(train),val_count=len(val),selection='native pre-NMS Val mAP' if a.model=='E0' else 'Val omni_spectrum',trainable=sum(p.numel() for p in active),fresh_optimizer=True,Test_used=False));write_json(directory/'INITIALIZATION_CHECK.json',checks)
 best=0. if a.model=='E0' else -1.;stale=0
 for epoch in range(1,job['epochs']+1):
  if a.model=='E0':model.train()
  else:training_mode(model)
  sums={};start=time.time()
  for step,(metadata,batch) in enumerate(loader,1):
   features,targets=prepare_batch_inputs(batch,opt.device,non_blocking=False);targets.update(label=metadata,fps=torch.full((len(metadata),),1/opt.clip_length,device=opt.device))
   output=model(**features,targets=targets)
   if a.model=='E0':
    losses=criterion(metadata,output,targets);details={k:v for k,v in losses.items() if 'loss' in k and k in criterion.weight_dict};loss=sum(v*criterion.weight_dict[k] for k,v in details.items())
   else:loss,details=compute_mcev_loss(output,metadata,targets,job,model=model,model_inputs=features)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
   optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(active,job['grad_clip'],error_if_nonfinite=True);optimizer.step()
   for k,v in details.items():sums[k]=sums.get(k,0.)+float(v.detach())
   if step==1 or step%50==0:print(f'{a.model} epoch={epoch} batch={step}/{len(loader)} loss={float(loss.detach()):.5f}',flush=True)
  scheduler.step()
  for n,p_ in model.named_parameters():
   if n in frozen:assert torch.equal(p_.detach().cpu(),frozen[n]),n
  model.eval();pre,post=predictions(model,val,opt);report=Evaluation(pre,post,val.data).report('val',comprehensive=True)
  score=eval_submission(pre,val.data,verbose=False,match_number=True,full_only=opt.eval_full_only,mr_only=opt.mr_only)['brief']['MR-full-mAP'] if a.model=='E0' else selection_score(report)
  improved=score>best;stale=0 if improved else stale+1
  if improved:
   best=score;write_json(directory/'selected.json',dict(epoch=epoch,score=score,source='Val only'));write_json(directory/'validation_metrics.json',report);write_rows(directory/'validation_predictions.jsonl',post);write_rows(directory/'validation_predictions_pre_nms.jsonl',pre)
  state=dict(epoch=epoch,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),best_score=best,stale=stale)
  for filename in (['best.ckpt','latest.ckpt'] if improved else ['latest.ckpt']):
   path=directory/filename;temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)
  record=dict(epoch=epoch,score=score,best=best,improved=improved,stale=stale,seconds=time.time()-start,losses={k:v/len(loader) for k,v in sums.items()})
  with (directory/'trace.jsonl').open('a') as f:
   import json
   f.write(json.dumps(record)+'\n')
  print(f'VAL {a.model} epoch={epoch} score={score:.4f} best={best:.4f}',flush=True)
  if (stale>job['early_stop_patience'] if a.model=='E0' else stale>=job['early_stop_patience']):break
 write_json(directory/'COMPLETE.json',dict(state='training_complete',epochs=epoch,best=best,Test_used=False))
if __name__=='__main__':main()
