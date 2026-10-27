#!/usr/bin/env python3
"""Scratch FlashVTG + GEQ + FlashMMR-style auxiliary PV. Isolated E4 run."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,random,subprocess,sys,time,traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import ROOT,OLD,dump,rows,seed,sha
from scratch_adapter import ScratchAdapter,forbid_pretrained_restore
from model import JointGMR
from run_E3 import tensor_hashes,gradient_norm,objective as base_objective
from pv import losses as pv_losses

SETTING='E4_scratch_flashmmr'
CODE=('run_E4_flashmmr.py','config_E4_flashmmr.json','scratch_adapter.py','model.py','pv.py','run_E3.py','evaluation.py','independent_metrics.py')

def construct():
    with forbid_pretrained_restore() as attempts:
        adapter=ScratchAdapter('flashvtg','cuda:0')
        model=JointGMR(adapter.base,'flashvtg',pv=True).to('cuda:0')
    assert attempts==[] and all(p.requires_grad for p in model.parameters())
    assert not list(model.base.exist_head.parameters())
    return adapter,model

def objective(adapter,out,metas,targets,epoch=0):
    base,parts,details=base_objective(adapter,out,metas,targets,epoch)
    quality,representation=pv_losses(out,metas)
    parts=dict(localization=parts['localization'],geq=parts['geq'],quality=quality,representation=representation,total=base+quality+.2*representation)
    assert all(torch.isfinite(v) for v in parts.values())
    return parts['total'],parts,details

def optimizer_for(model):
    return torch.optim.AdamW([dict(params=model.base.parameters(),lr=1e-5,name='random_gmr'),dict(params=list(model.geq.parameters())+list(model.pv.parameters()),lr=3e-5,name='random_geq_pv')],weight_decay=1e-4,foreach=False)

@torch.no_grad()
def validation_loss(adapter,model,data,epoch):
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all());modes={m:m.training for m in model.modules()};sums={};count=0
    try:
        model.eval();model.training=True;model.base.training=True;adapter.criterion.eval()
        for start in range(0,len(data),adapter.bs):
            items=[data[i] for i in range(start,min(start+adapter.bs,len(data)))];metas,inputs,targets=adapter.collate(items)
            _,parts,_=objective(adapter,model(**inputs),metas,targets,epoch)
            for k,v in parts.items():sums[k]=sums.get(k,0)+float(v)*len(items)
            count+=len(items)
        return {k:v/count for k,v in sums.items()}
    finally:
        for m,mode in modes.items():m.training=mode
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])

def preflight(adapter,model,train,val,directory,initial):
    original=json.loads((ROOT/'runs/flashvtg/E3/CONFIG.json').read_text())['initial_tensor_sha256']
    shared={n:h for n,h in initial.items() if not n.startswith('pv.')}
    assert shared==original,'E3/E4 native and GEQ initial tensors differ'
    model.train()
    ids=[next(i for i,m in enumerate(train.data) if test(m)) for test in (lambda m:bool(m['relevant_windows']),lambda m:not m['relevant_windows'],lambda m:len(m['relevant_windows'])>=2)]
    metas,inputs,targets=adapter.collate([train[i] for i in ids]);out=model(**inputs);loss,parts,details=objective(adapter,out,metas,targets)
    checks={f'{key}_to_transformer':gradient_norm(parts[key],model.base.transformer.named_parameters()) for key in ('geq','quality','representation','localization')}
    checks['native_boundary']=gradient_norm(parts['localization'],model.base.coord_head.named_parameters())
    checks['quality_to_gru']=gradient_norm(parts['quality'],model.pv.gru.named_parameters())
    assert out['moment_sequence'].shape[:2]==out['candidate_spans'].shape[:2]
    assert torch.equal(out['candidate_spans'],out['candidate_seconds'].gather(1,out['candidate_indices'][...,None].expand(-1,-1,3)))
    del out,loss,parts,details
    optimizer=optimizer_for(model);cases={}
    for label,which in [('positive',[ids[0]]),('negative',[ids[1]]),('multiple_gt',[ids[2]]),('native_batch8',list(range(adapter.bs)))]:
        metas,inputs,targets=adapter.collate([train[i] for i in which]);model.train();optimizer.zero_grad(set_to_none=True)
        loss,parts,details=objective(adapter,model(**inputs),metas,targets);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);optimizer.step()
        cases[label]=dict(parts={k:float(v.detach()) for k,v in parts.items()},gradient_norm=float(norm))
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    checkpoint=directory/'smoke.ckpt';torch.save(dict(model=model.state_dict(),smoke_only=True),checkpoint)
    model.eval();metas,inputs,_=adapter.collate([val[0]])
    with torch.no_grad():
        before=model(**inputs);expected={k:before[k].clone() for k in ('pred_exist_logits','candidate_seconds','quality_score')}
    second_adapter,second=construct();second.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True);second.eval()
    with torch.no_grad():
        after=second(**inputs);assert all(torch.equal(v,after[k]) for k,v in expected.items())
        # Perturb only PV parameters: official inference outputs must be identical.
        for p in second.pv.parameters():p.add_(torch.randn_like(p)*.2)
        changed=second(**inputs)
        assert torch.equal(after['pred_exist_logits'],changed['pred_exist_logits']) and torch.equal(after['candidate_seconds'],changed['candidate_seconds'])
    vi=[next(i for i,m in enumerate(val.data) if len(m['relevant_windows'])>=2),next(i for i,m in enumerate(val.data) if not m['relevant_windows'])]
    class View:
        def __len__(self):return len(vi)
        def __getitem__(self,i):return val[vi[i]]
    from evaluation import predict,evaluate
    pred,diag=predict(adapter,model,View());evaluate(pred,[val.data[i] for i in vi],directory,'smoke_val',diag)
    val_loss=validation_loss(adapter,model,View(),0)
    final=tensor_hashes(model);changed=[n for n,h in initial.items() if final[n]!=h]
    assert all(any(n.startswith(group) for n in changed) for group in ('base.transformer.','base.coord_head.','geq.','pv.'))
    dump(directory/'PREFLIGHT.json',dict(passed=True,shared_initial_tensors_equal_to_e3=True,random_initialization_without_restore=True,checks=checks,cases=cases,validation_loss=val_loss,strict_reload_equal=True,pv_perturbation_does_not_change_official_predictions=True,changed_parameters=changed,code_sha256={n:sha(ROOT/n) for n in CODE}))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        assert not (ROOT/f'runs/flashvtg/{SETTING}').exists()
        with (ROOT/f'logs/flashvtg_{SETTING}.log').open('x') as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'run_E4_flashmmr.py')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        dump(ROOT/'FLASHVTG_E4_JOB.json',dict(pid=process.pid,gpu=1,epochs=76,status_file=str(ROOT/f'runs/flashvtg/{SETTING}/STATUS.json'),automatic_test=True));print('E4_PID',process.pid);return
    torch.set_num_threads(2);seed();protocol=json.loads((ROOT/'config_E4_flashmmr.json').read_text());epochs=protocol['epochs']
    directory=ROOT/('preflights' if args.preflight_only else 'runs')/'flashvtg'/SETTING;directory.mkdir(parents=True,exist_ok=False)
    dump(directory/'STATUS.json',dict(status='initializing',pid=os.getpid(),epochs=epochs))
    try:
        adapter,model=construct();initial=tensor_hashes(model)
        original=json.loads((ROOT/'runs/flashvtg/E3/CONFIG.json').read_text())['initial_tensor_sha256']
        assert {n:h for n,h in initial.items() if not n.startswith('pv.')}==original
        dump(directory/'CONFIG.json',dict(protocol=protocol,initialization=adapter.manifest,initial_tensor_sha256=initial,shared_initial_equal_e3=True,source_sha256={n:sha(ROOT/n) for n in CODE},reference=json.loads((ROOT/'reference/SOURCE.json').read_text()),data_sha256={s:sha(ROOT/f'data/{s}.jsonl') for s in ('train','val','test')}))
        train,val=adapter.dataset('train'),adapter.dataset('val')
        if args.preflight_only:
            preflight(adapter,model,train,val,directory,initial);dump(directory/'STATUS.json',dict(status='preflight_passed'));print('E4_PREFLIGHT_PASSED',flush=True);return
        gate=json.loads((ROOT/f'preflights/flashvtg/{SETTING}/PREFLIGHT.json').read_text());assert gate['passed']
        assert all(sha(ROOT/n)==h for n,h in gate['code_sha256'].items())
        optimizer=optimizer_for(model);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:1.)
        seed();loader=DataLoader(train,batch_size=adapter.bs,shuffle=True,num_workers=0,collate_fn=list,generator=torch.Generator().manual_seed(2026),drop_last=False)
        for epoch in range(1,epochs+1):
            started=time.monotonic();sums={};count=0
            for step,items in enumerate(loader,1):
                model.train();adapter.criterion.train();optimizer.zero_grad(set_to_none=True)
                metas,inputs,targets=adapter.collate(items);loss,parts,details=objective(adapter,model(**inputs),metas,targets,epoch-1)
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);optimizer.step()
                for k,v in parts.items():sums[k]=sums.get(k,0)+float(v.detach())*len(items)
                count+=len(items)
                if step==1 or step%25==0:
                    progress=dict(status='training',setting=SETTING,epoch=epoch,total_epochs=epochs,step=step,total_steps=len(loader),loss=float(loss.detach()),gradient_norm=float(norm));dump(directory/'STATUS.json',progress);print(json.dumps(progress),flush=True)
            scheduler.step();vl=validation_loss(adapter,model,val,epoch-1)
            record=dict(epoch=epoch,train_losses={k:v/count for k,v in sums.items()},val_losses=vl,seconds=time.monotonic()-started)
            with (directory/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            checkpoint=directory/f'epoch_{epoch:03d}.ckpt'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,setting=SETTING,initial_tensor_sha256=initial,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),dataloader=loader.generator.get_state())),checkpoint)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
        dump(directory/'CHECKPOINT.json',dict(epoch=epochs,sha256=sha(checkpoint),strict_own_reload=True))
        from evaluation import predict,evaluate
        dump(directory/'STATUS.json',dict(status='evaluating',epoch=epochs))
        for split,data in [('val',val),('test',adapter.dataset('test'))]:
            pred,diag=predict(adapter,model,data);evaluate(pred,data.data,directory,split,diag,rows(OLD/f'results/flashvtg/gmr/{split}_predictions.jsonl'))
        dump(directory/'COMPLETE.json',dict(status='complete',epoch=epochs,seed=2026,setting=SETTING));dump(directory/'STATUS.json',dict(status='reporting',epoch=epochs))
        subprocess.run([sys.executable,str(ROOT/'report_E4_flashmmr.py')],cwd=ROOT,check=True)
        dump(directory/'STATUS.json',dict(status='complete',epoch=epochs,finished=time.time()))
    except Exception:
        dump(directory/'STATUS.json',dict(status='failed',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
