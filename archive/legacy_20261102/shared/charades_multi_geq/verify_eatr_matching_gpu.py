"""Train-only GPU equivalence and contended timing before activating a saved run."""
import argparse,copy,fcntl,importlib.util,json,statistics,time
import torch
from paths import OUT,dump,sha
from engine_utils import restore
from runner import build
from adapter import Adapter


def main():
 p=argparse.ArgumentParser();p.add_argument('--legacy-adapter',required=True);a=p.parse_args()
 d=OUT/'runs/eatr/multi'
 with (d/'task.lock').open('a') as lock,(d/'.lock').open('a') as legacy_lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(legacy_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  torch.set_num_threads(4)
  spec=importlib.util.spec_from_file_location('eatr_adapter_before_transfer',a.legacy_adapter)
  old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
  saved=torch.load(d/'last.ckpt',map_location='cpu',weights_only=False)
  checkpoint_sha=sha(d/'last.ckpt');ad,model,opt,parent=build('eatr','multi')
  model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);restore(saved['rng'])
  initial=copy.deepcopy(model.state_dict());ds=ad.dataset('train')
  positive=[i for i,r in enumerate(ds.data) if r['relevant_windows']]
  negative=[i for i,r in enumerate(ds.data) if not r['relevant_windows']]
  params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
  cases={};mixed=None
  for case,ids in [('mixed32',positive[:16]+negative[:16]),('all_negative32',negative[:32])]:
   metas,inputs,targets=ad.collate([ds[i] for i in ids]);model.train();opt.zero_grad(set_to_none=True)
   if case=='mixed32':mixed=(metas,inputs,targets)
   output=model(**inputs)
   new=Adapter.loss(ad,output,targets,metas,saved['epoch'])
   previous=old.Adapter.loss(ad,output,targets,metas,saved['epoch'])
   torch.testing.assert_close(new,previous,rtol=0,atol=0)
   ga=torch.autograd.grad(new,[v for _,v in params],retain_graph=True,allow_unused=True)
   gb=torch.autograd.grad(previous,[v for _,v in params],allow_unused=True)
   maximum=0.;active=0
   for (name,_),x,y in zip(params,ga,gb):
    assert (x is None)==(y is None),name
    if x is not None:
     assert torch.isfinite(x).all() and torch.isfinite(y).all(),name
     torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6,msg=lambda msg:name+': '+msg)
     maximum=max(maximum,float((x-y).abs().max()));active+=1
   cases[case]=dict(loss=float(new),loss_exact_equal=True,gradient_max_abs_difference=maximum,gradient_tolerance=dict(rtol=1e-5,atol=1e-6),parameters_with_gradients=active,train_rows=[ds.data[i]['evaluation_id'] for i in ids])
   del ga,gb,output,new,previous
  timings={};metas,inputs,targets=mixed;ad._valid=inputs['src_vid_mask'].bool()
  # No optimizer updates: both versions see the same model and Train batch.
  for label,fn in [('legacy',old.Adapter.loss),('packed',Adapter.loss)]:
   elapsed=[]
   for step in range(10):
    restore(saved['rng']);opt.zero_grad(set_to_none=True);torch.cuda.synchronize();begun=time.perf_counter()
    value=fn(ad,model(**inputs),targets,metas,saved['epoch']);assert torch.isfinite(value);value.backward();torch.cuda.synchronize()
    if step>=2:elapsed.append(time.perf_counter()-begun)
   timings[label]=dict(median_seconds=statistics.median(elapsed),samples_seconds=elapsed,warmups=2)
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items()),'Diagnostic changed model state'
  assert sha(d/'last.ckpt')==checkpoint_sha,'Formal checkpoint changed during diagnostic'
  report=dict(passed=True,split='train',checkpoint_sha256=checkpoint_sha,epoch=saved['epoch'],cursor=saved['cursor'],cases=cases,timings=timings,
   fixed_batch_forward_loss_backward_speed_ratio=timings['legacy']['median_seconds']/timings['packed']['median_seconds'],
   timing_scope='Fixed Train mixed32 forward/loss/backward, no optimizer step; concurrent GPU jobs continue. This does not establish full-epoch speedup.',
   formal_model_and_checkpoint_unchanged=True)
  dump(OUT/'reports/eatr_matching_gpu_verification_20261101.json',report)
  print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
