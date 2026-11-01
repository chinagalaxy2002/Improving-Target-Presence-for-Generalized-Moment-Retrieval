"""CPU training-loop regression: skipped Val, epoch patience, ties and resume."""
import contextlib,io,json,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import runner
from paths import dump,rows,sha
from engine_utils import save
from validation_schedule import validation_due


def check():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);(root/'data').mkdir();(root/'data/train.jsonl').write_text('{}\n')
  ad=SimpleNamespace(name='qd_detr',collate=lambda items:([],{'input':torch.ones(1,1)},{}))
  def batches(ad,split,training,generator):
   torch.randperm(3,generator=generator)
   return [0,1,2]
  def loss(ad,model,out,targets,metas,epoch):return ((out-torch.rand_like(out))**2).mean()
  def evaluate(ad,model,split,output,parent):
   epoch=len(rows(output.parent/'history.jsonl'))+1 if (output.parent/'history.jsonl').exists() else 1
   calls.append(epoch)
   # Same mean at every epoch, but all three exceed GMR only from epoch2.
   values=[.6,.6,.6] if epoch==1 else [.7,.7,.4]
   metrics=dict(mAP=1.,selection_mean=sum(values)/3,selection_values=values)
   return metrics,[]
  patches=[patch.object(runner,'OUT',root),patch.object(runner,'loader',batches),
   patch.object(runner,'loss',loss),patch.object(runner,'evaluate',evaluate),
   patch.object(runner,'preflight',lambda *a:None),
   patch.object(torch.cuda,'reset_peak_memory_stats',lambda:None),
   patch.object(torch.cuda,'max_memory_allocated',lambda:0)]
  with contextlib.ExitStack() as stack:
   for p in patches:stack.enter_context(p)
   stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
   def run(name,interrupted=False,parent=None,existing=None):
    global calls
    calls=[];d=root/name;d.mkdir();dump(d/'config.json',{})
    torch.manual_seed(2023);model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters(),lr=3e-5)
    args=SimpleNamespace(method='gmr_geq_full' if parent else 'gmr',interrupt_after_steps=1 if interrupted else 0)
    if existing:
     history=[dict(epoch=e,selection=float(e),mAP=float(e),val_values=[0.,0.,0.],best=float(e),best_epoch=e,stale=0) for e in range(1,4)]
     (d/'history.jsonl').write_text(''.join(json.dumps(h)+'\n' for h in history))
     g=torch.Generator().manual_seed(2023)
     state=runner.checkpoint(model,opt,4,0,g.get_state(),3.,3,0,False,sha(root/'data/train.jsonl'))
     save(d/'last.ckpt',state);save(d/'best.ckpt',state)
    runner.STOP=False;runner.train(ad,model,opt,parent,d,args)
    if interrupted:
     state=torch.load(d/'last.ckpt',weights_only=False);assert state['cursor']==1 and state['epoch']==1
     assert not (d/'TRAINED.json').exists()
     args.interrupt_after_steps=0
     # Reconstruct process state, then let train strictly restore last.
     torch.manual_seed(999);model=torch.nn.Linear(1,1);opt=torch.optim.AdamW(model.parameters(),lr=3e-5)
     runner.train(ad,model,opt,parent,d,args)
    return d,calls[:],torch.load(d/'last.ckpt',weights_only=False)
   a,events,sa=run('continuous')
   assert events==[5,10,15,20,25],events
   h=rows(a/'history.jsonl');assert len(h)==25 and h[-1]['stale']==20
   assert all(r['selection'] is None and r['val_values'] is None for r in h if not r['validated'])
   b,events2,sb=run('resumed',interrupted=True)
   assert events2==events
   assert rows(b/'history.jsonl')[-1]['best_epoch']==5
   assert all(torch.equal(v,sb['model'][k]) for k,v in sa['model'].items())
   assert torch.equal(sa['rng']['torch'],sb['rng']['torch'])
   assert torch.equal(sa['sampler_epoch_start'],sb['sampler_epoch_start'])
   assert sa['optimizer']['state'].keys()==sb['optimizer']['state'].keys()
   for k,v in sa['optimizer']['state'].items():
    assert all(torch.equal(t,sb['optimizer']['state'][k][n]) for n,t in v.items())
   c,events3,sc=run('legacy',existing=True)
   assert events3==[5,10,15,20,23],events3
   assert sc['best_epoch']==3 and rows(c/'history.jsonl')[-1]['stale']==20
   assert rows(c/'history.jsonl')[3]['validated'] is False
   parent=root/'parent';parent.mkdir();dump(parent/'best_val_metrics.json',dict(selection_values=[.65,.65,.2]))
   gmr=root/'runs/qd_detr/gmr';gmr.mkdir(parents=True);dump(gmr/'best_val_metrics.json',dict(selection_values=[.65,.65,.2]))
   d,events4,sd=run('stage_b',parent=parent)
   assert events4==list(range(1,8)),events4
   assert sd['best_epoch']==2 and sd['stale']==6,'Equal-score tie must not reset patience'
   assert all(r['validated'] for r in rows(d/'history.jsonl'))
  return dict(passed=True,stage_a_validations=events,stage_a_stopped_epoch=25,
   legacy_best_retained=True,legacy_terminal_validation_epoch=23,
   stage_b_validations=events4,stage_b_tie_did_not_reset_patience=True,
   interrupted_resume_model_optimizer_rng_sampler_equal=True)


if __name__=='__main__':print(json.dumps(check(),indent=2))
