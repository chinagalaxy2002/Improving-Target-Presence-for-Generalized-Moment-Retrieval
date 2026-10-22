"""Read-only extraction verification; no optimizer, backward or training."""
import gc,argparse,json
from collections import Counter
from environment import ROOT,ASSETS,make_dataset,read_rows,read_json,write_json,digest
from api import build_model,load_selected,SHAS
import torch
from torch.utils.data import DataLoader
from training.flash_vtg_gmr.dataset import start_end_collate,prepare_batch_inputs

def main():
 p=argparse.ArgumentParser();p.add_argument('--full-dataset',action='store_true');a=p.parse_args();torch.set_num_threads(2)
 model,opt,checks,_=build_model('S4Fusion','cpu',initialize=True)
 expected=read_json(ROOT/'results/S4Fusion/INITIALIZATION_CHECK.json')
 assert checks['verified_shared_tensors']==expected['verified_shared_tensors']==212
 assert checks['tensor_sha256']==expected['tensor_sha256'],'Initialization changed'
 record={'initialization_all_319_state_hashes_match':len(checks['tensor_sha256']),'shared_E0_tensors':212,'models':{},'training_or_backward_executed':False}
 del model;gc.collect()
 for name in ['E0','S4Fusion']:
  model,opt,saved=load_selected(name,'cpu');states=model.state_dict()
  assert all(torch.equal(v.detach().cpu(),saved['model'][k].cpu()) for k,v in states.items())
  before={k:v.detach().cpu().clone() for k,v in states.items()};data=make_dataset(opt,'val')
  archived={r['qid']:r for r in read_rows(ROOT/'results'/name/'validation_predictions.jsonl')} if (ROOT/'results'/name/'validation_predictions.jsonl').exists() else None
  calls=Counter();hooks=[];global_logits=[]
  if name=='S4Fusion':
   params=read_json(ROOT/'results/S4Fusion/PARAMETERS.json');assert all(p.requires_grad==params[n]['requires_grad'] for n,p in model.named_parameters())
   assert sum(p.numel() for p in model.parameters() if p.requires_grad)==2496452
   for k in ['moment_cross_attn','moment_match_mlp','local_head','fusion_gate','refiner','extractor']:
    def hook(_m,_i,_o,key=k):calls[key]+=1
    hooks.append(getattr(model.base.exist_head,k).register_forward_hook(hook))
   def capture(_m,_i,o):global_logits.append(o.detach().cpu())
   hooks.append(model.base.exist_head.global_head.register_forward_hook(capture))
  selected=[next(i for i,r in enumerate(data.data) if bool(r['relevant_windows'])==positive) for positive in [False,True]]
  with torch.inference_mode():
   for i in selected:
    meta,batch=start_end_collate([data[i]]);features,targets=prepare_batch_inputs(batch,opt.device,non_blocking=False);targets=targets or {};targets.update(label=meta,fps=torch.full((len(meta),),1/opt.clip_length,device=opt.device));out=model(**features,targets=targets)
    assert torch.isfinite(out['pred_exist_logits']).all()
    if archived:
     actual=float(out['pred_exist_logits'].reshape(-1)[0]);expected_score=archived[meta[0]['qid']]['pred_exist_logit_raw'];assert abs(actual-expected_score)<.001,(name,actual,expected_score)
    if name=='S4Fusion':assert torch.equal(out['pred_exist_logits'].cpu().reshape(-1),global_logits[-1].reshape(-1))
  for h in hooks:h.remove()
  if name=='S4Fusion':
   assert all(calls[k]==0 for k in ['moment_cross_attn','moment_match_mlp','local_head','fusion_gate']);assert calls['refiner']==calls['extractor']==2
  assert all(torch.equal(v.detach().cpu(),before[k]) for k,v in model.state_dict().items())
  record['models'][name]={'strict_selected_state_tensors':len(states),'all_weights_unchanged':True,'Val_positive_and_negative_forward':'passed','archived_Val_logit_comparison':bool(archived),'branch_call_counts':{k:calls[k] for k in ['moment_cross_attn','moment_match_mlp','local_head','fusion_gate','refiner','extractor']} if name=='S4Fusion' else {}}
  del model,states,saved,before;gc.collect()
 train=make_dataset(opt,'train');assert [str(p) for p in opt.v_feat_dirs]==[str(ASSETS/'slowfast'),str(ASSETS/'clip')]
 if a.full_dataset:
  count=negative=0;seen=set()
  for metadata,batch in DataLoader(train,batch_size=8,shuffle=False,num_workers=0,collate_fn=start_end_collate):
   features,targets=prepare_batch_inputs(batch,opt.device,non_blocking=False);assert features['src_vid'].shape[-1]==2818 and features['src_txt'].shape[-1]==512
   count+=len(metadata);negative+=int((targets['exist_label']==0).sum());seen.update(r['qid'] for r in metadata)
  assert count==len(seen)==4138 and negative==2002
  record['full_Dataset_collate']={'rows':count,'empty_GT_retained':negative,'qid_unique':True,'video_order':'SlowFast2304 -> CLIP512 -> TEF2','text_dimension':512}
 else:record['full_Dataset_collate']='not requested'
 write_json(ROOT/'reports/READ_ONLY_PREFLIGHT.json',record);print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
