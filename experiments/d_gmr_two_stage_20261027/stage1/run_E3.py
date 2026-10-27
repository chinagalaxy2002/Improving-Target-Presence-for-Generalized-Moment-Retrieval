#!/usr/bin/env python3
"""Single-stage random-initialization GEQ training. Never loads trained GMR weights."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, hashlib, json, random, time, traceback
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import ROOT, OLD, BACKBONES, dump, rows, seed, sha
from scratch_adapter import ScratchAdapter, forbid_pretrained_restore
from model import JointGMR
def gradient_norm(loss, named):
    named = [(n,p) for n,p in named if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for n,p in named], retain_graph=True, allow_unused=True)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
    result = {n:float(g.norm()) for (n,p),g in zip(named,grads) if g is not None and bool(g.abs().max()>0)}
    assert result, 'Missing expected independent gradient'
    return result

EPOCHS = json.loads((ROOT/'config_E3.json').read_text())['epochs']

CODE = ('run_E3.py','scratch_adapter.py','model.py','adapters.py','common.py','evaluation.py','independent_metrics.py')

def tensor_hashes(model):
    return {n:hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,t in model.state_dict().items()}

def construct(backbone,device):
    with forbid_pretrained_restore() as attempts:
        adapter = ScratchAdapter(backbone,device)
        model = JointGMR(adapter.base,backbone,pv=False).to(device)
    assert attempts == [] and not hasattr(model,'pv')
    assert not list(model.base.exist_head.parameters())
    assert all(p.requires_grad for p in model.parameters())
    return adapter,model

def objective(adapter,out,metas,targets,epoch=0):
    local,details = adapter.native_loss(out['native_output'],targets,metas,epoch)
    logits=out['pred_exist_logits'].reshape(-1)
    labels=logits.new_tensor([bool(m['relevant_windows']) for m in metas])
    geq=F.binary_cross_entropy_with_logits(logits,labels)
    assert torch.equal(out['pred_exist_score'],logits.sigmoid())
    total=local+geq
    assert torch.isfinite(total),details
    return total,dict(localization=local,geq=geq,total=total),details

def optimizer_for(model):
    return torch.optim.AdamW([dict(params=model.base.parameters(),lr=1e-5,name='random_gmr'),dict(params=model.geq.parameters(),lr=3e-5,name='random_geq')],weight_decay=1e-4,foreach=False)

def preflight(adapter,model,train,val,directory,initial):
    model.train()
    ids=[next(i for i,m in enumerate(train.data) if predicate(m)) for predicate in (lambda m:bool(m['relevant_windows']),lambda m:not m['relevant_windows'],lambda m:len(m['relevant_windows'])>=2)]
    metas,inputs,targets=adapter.collate([train[i] for i in ids])
    out=model(**inputs);total,parts,details=objective(adapter,out,metas,targets)
    checks=dict(geq_to_transformer=gradient_norm(parts['geq'],model.base.transformer.named_parameters()),
                geq_to_video_projection=gradient_norm(parts['geq'],model.base.input_vid_proj.named_parameters()),
                geq_to_text_projection=gradient_norm(parts['geq'],model.base.input_txt_proj.named_parameters()))
    boundary='coord_head' if adapter.name=='flashvtg' else 'moment_span_embed' if adapter.name=='eatr' else 'span_embed'
    classifier='class_head' if adapter.name=='flashvtg' else 'class_embed'
    checks['localization_to_boundary']=gradient_norm(parts['localization'],[(n,p) for n,p in model.base.named_parameters() if boundary in n])
    checks['localization_to_classifier']=gradient_norm(parts['localization'],[(n,p) for n,p in model.base.named_parameters() if classifier in n])
    checks['localization_to_transformer']=gradient_norm(parts['localization'],model.base.transformer.named_parameters())
    assert torch.equal(out['candidate_spans'],out['candidate_seconds'].gather(1,out['candidate_indices'][...,None].expand(-1,-1,3)))
    del out,total,parts,details
    optimizer=optimizer_for(model);cases={}
    for case,indices in [('positive',[ids[0]]),('negative',[ids[1]]),('multi',[ids[2]]),('native_full_batch',list(range(adapter.bs)))]:
        metas,inputs,targets=adapter.collate([train[i] for i in indices]);model.train();optimizer.zero_grad(set_to_none=True)
        out=model(**inputs);total,parts,details=objective(adapter,out,metas,targets)
        total.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if case=='native_full_batch':
            missing=[n for n,p in model.base.named_parameters() if p.grad is None]
            allowed=['txt_position_embed.','txt_proj_linear.']
            if adapter.name in ('moment_detr','eatr'):allowed+=['saliency_proj.']
            if adapter.name=='flashvtg':allowed+=['saliency_proj1.','saliency_proj2.','pooling.att.']
            assert all(any(n.startswith(prefix) for prefix in allowed) for n in missing),missing
        optimizer.step();cases[case]=dict(losses={k:float(v.detach()) for k,v in parts.items()},gradient_norm=float(norm),native_loss_names=list(details))
        del out,total,parts,details
    final=tensor_hashes(model);changed=[n for n,h in initial.items() if final[n]!=h]
    assert any(n.startswith('base.transformer.') for n in changed) and any(boundary in n for n in changed) and any(n.startswith('geq.') for n in changed)
    checkpoint=directory/'smoke.ckpt';torch.save(dict(model=model.state_dict(),smoke_only=True,setting='E3'),checkpoint)
    metas,inputs,_=adapter.collate([val[0]]);model.eval()
    with torch.no_grad():out=model(**inputs);before={k:out[k].clone() for k in ('pred_exist_logits','candidate_seconds')}
    second_adapter,second=construct(adapter.name,adapter.device)
    # Only this experiment's own smoke checkpoint is reloaded, after guarded random construction.
    second.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True);second.eval()
    with torch.no_grad():
        after=second(**inputs)
        assert all(torch.equal(v,after[k]) for k,v in before.items())
    from evaluation import predict,evaluate
    vi=[next(i for i,m in enumerate(val.data) if len(m['relevant_windows'])>=2),next(i for i,m in enumerate(val.data) if not m['relevant_windows'])]
    class View:
        def __len__(self):return len(vi)
        def __getitem__(self,i):return val[vi[i]]
    pred,diag=predict(second_adapter,second,View());evaluate(pred,[val.data[i] for i in vi],directory,'smoke_val',diag)
    val_losses=validation_loss(adapter,model,View(),0)
    assert all(np.isfinite(x) for x in val_losses.values())
    dump(directory/'PREFLIGHT.json',dict(passed=True,setting='E3',backbone=adapter.name,random_initialization_guard=True,native_presence_parameters=0,native_presence_bce=False,geq_only_bce=True,pv=False,distillation=False,gradient_checks=checks,cases=cases,no_gradient_native_parameters=missing,changed_parameters=changed,strict_own_smoke_reload_equal=True,native_inference_and_official_eval=True,validation_loss_check=val_losses,code_sha256={n:sha(ROOT/n) for n in CODE},peak_vram_mib=torch.cuda.max_memory_allocated()/2**20))

@torch.no_grad()
def validation_loss(adapter,model,val,epoch):
    # Dense native localization losses require train-output branches (notably FlashVTG/CG).
    # Disable dropout/BN and preserve every RNG; no optimizer/gradient updates.
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    modes={m:m.training for m in model.modules()};sums={};count=0
    try:
        model.eval()
        # Only parent flags select the native dense output/API branch. Child
        # modules, including attention with functional dropout, remain in eval.
        model.training=True; model.base.training=True
        adapter.criterion.eval()
        for start in range(0,len(val),adapter.bs):
            items=[val[i] for i in range(start,min(len(val),start+adapter.bs))]
            metas,inputs,targets=adapter.collate(items);out=model(**inputs)
            _,parts,_=objective(adapter,out,metas,targets,epoch)
            for k,v in parts.items():sums[k]=sums.get(k,0)+float(v)*len(items)
            count+=len(items)
        return {k:v/count for k,v in sums.items()}
    finally:
        for m,mode in modes.items():m.training=mode
        random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state_all(state[3])


def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',choices=BACKBONES,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);seed()
    directory=ROOT/('preflights' if args.preflight_only else 'runs')/args.backbone/'E3';directory.mkdir(parents=True,exist_ok=False)
    dump(directory/'STATUS.json',dict(status='initializing',pid=os.getpid(),started=time.time()))
    try:
        adapter,model=construct(args.backbone,args.device);initial=tensor_hashes(model)
        dump(directory/'CONFIG.json',dict(setting='E3',protocol=json.loads((ROOT/'config_E3.json').read_text()),initialization=adapter.manifest,initial_tensor_sha256=initial,source_sha256={n:sha(ROOT/n) for n in CODE},factory_config_sha256=sha(ROOT/f'configs/E3/{args.backbone}.json'),data_sha256={s:sha(ROOT/f'data/{s}.jsonl') for s in ('train','val','test')}))
        train,val=adapter.dataset('train'),adapter.dataset('val')
        if args.preflight_only:
            # Same-seed construction must reproduce every tensor without checkpoint loading.
            seed();_,second=construct(args.backbone,args.device)
            assert tensor_hashes(second)==initial
            del second
            preflight(adapter,model,train,val,directory,initial)
            dump(directory/'STATUS.json',dict(status='preflight_passed'));print('E3_PREFLIGHT_PASSED',args.backbone,flush=True);return
        gate=json.loads((ROOT/f'preflights/{args.backbone}/E3/PREFLIGHT.json').read_text());assert gate['passed']
        assert all(sha(ROOT/n)==h for n,h in gate['code_sha256'].items()),'Source changed after E3 preflight'
        before=json.loads((ROOT/f'preflights/{args.backbone}/E3/CONFIG.json').read_text());assert before['initial_tensor_sha256']==initial
        dump(directory/'INITIALIZATION_VERIFIED.json',dict(passed=True,random_initial_hash_matches_preflight=True,trained_checkpoint_loaded=False,restore_api_attempts=0))
        optimizer=optimizer_for(model);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:1.)
        seed();loader=DataLoader(train,batch_size=adapter.bs,shuffle=True,num_workers=0,collate_fn=list,generator=torch.Generator().manual_seed(2026),drop_last=False)
        for epoch in range(1,EPOCHS+1):
            start=time.monotonic();sums={};count=0
            for step,items in enumerate(loader,1):
                model.train();adapter.criterion.train();optimizer.zero_grad(set_to_none=True)
                metas,inputs,targets=adapter.collate(items);out=model(**inputs)
                loss,parts,ld=objective(adapter,out,metas,targets,epoch-1);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);optimizer.step()
                log_values=dict(parts,**{f'native/{k}':v for k,v in ld.items() if torch.is_tensor(v) and v.ndim==0})
                for k,v in log_values.items():sums[k]=sums.get(k,0)+float(v.detach())*len(items)
                count+=len(items)
                if step==1 or step%25==0:
                    progress=dict(status='training',setting='E3',backbone=args.backbone,epoch=epoch,step=step,total_steps=len(loader),loss=float(loss.detach()),gradient_norm=float(norm))
                    dump(directory/'STATUS.json',progress);print(json.dumps(progress),flush=True)
            scheduler.step();vl=validation_loss(adapter,model,val,epoch-1)
            record=dict(epoch=epoch,train_losses={k:v/count for k,v in sums.items()},val_losses=vl,seconds=time.monotonic()-start)
            with (directory/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            checkpoint=directory/f'epoch_{epoch:03d}.ckpt'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,setting='E3',initialization=adapter.manifest,initial_tensor_sha256=initial,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),dataloader=loader.generator.get_state())),checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True);del saved
        final=tensor_hashes(model);changed=[n for n,h in initial.items() if final[n]!=h]
        assert any(n.startswith('base.transformer.') for n in changed) and any(n.startswith('geq.') for n in changed)
        dump(directory/'CHECKPOINT.json',dict(epoch=EPOCHS,setting='E3',sha256=sha(checkpoint),changed_parameters=changed,strict_own_checkpoint_reload=True))
        from evaluation import predict,evaluate
        dump(directory/'STATUS.json',dict(status='evaluating',epoch=EPOCHS))
        for split,data in [('val',val),('test',adapter.dataset('test'))]:
            pred,diag=predict(adapter,model,data)
            baseline=rows(OLD/f'results/{args.backbone}/gmr/{split}_predictions.jsonl')
            evaluate(pred,data.data,directory,split,diag,baseline)
            labels={m['qid']:bool(m['relevant_windows']) for m in data.data}
            dump(directory/f'{split}_geq_logits.json',dict(positive=[dict(qid=p['qid'],logit=p['pred_exist_logit_raw'],score=p['pred_exist_score']) for p in pred if labels[p['qid']]],negative=[dict(qid=p['qid'],logit=p['pred_exist_logit_raw'],score=p['pred_exist_score']) for p in pred if not labels[p['qid']]]))
        dump(directory/'COMPLETE.json',dict(status='complete',setting='E3',backbone=args.backbone,epoch=EPOCHS,seed=2026,random_initialization=True))
        dump(directory/'STATUS.json',dict(status='complete',finished=time.time()))
    except Exception:
        dump(directory/'STATUS.json',dict(status='failed',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
