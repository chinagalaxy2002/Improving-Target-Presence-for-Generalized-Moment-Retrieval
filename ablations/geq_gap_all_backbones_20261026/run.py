#!/usr/bin/env python3
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,random,sys,time,traceback
from pathlib import Path
EXP=Path(__file__).resolve().parent;PARENT=EXP.parent/'geq_mean_separation_20261026';SRC=EXP.parent/'end_to_end_geq_pv_20261026'
sys.path.insert(0,str(PARENT));sys.path.insert(0,str(SRC))
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import dump,rows,sha,seed,module
from adapters import Adapter
from model import JointGMR
BACKBONE=None

def construct():
    seed();adapter=Adapter(BACKBONE,'cuda:0');model=JointGMR(adapter.base,BACKBONE,False).to('cuda:0')
    initial=PREVIOUS.hashes(model);ref=json.loads((SRC/f'runs/{BACKBONE}/E1/CONFIG.json').read_text())
    assert initial==ref['shared_initial_tensor_sha256'] and adapter.bs==json.loads((EXP/f'configs/{BACKBONE}.json').read_text())['batch_size']
    assert all(p.requires_grad for p in model.parameters())
    return adapter,model,initial
# Load explicitly; never confuse the original run module with native backend files.
PREVIOUS=module('fixed_gap_training',PARENT/'train.py');BASE=PREVIOUS.BASE

def constraint(z,y,cfg):
    z=z.reshape(-1);y=y.reshape(-1).bool();s=z.sigmoid()
    if not y.any() or not (~y).any():return z.sum()*0.
    return cfg['weight']*(cfg['margin']-(s[y].mean()-s[~y].mean())).clamp_min(0).square()

def objective(adapter,out,metas,targets,cfg,epoch=0):
    base,parts,ld=BASE.objectives(adapter,out,metas,targets,epoch)
    z=out['pred_exist_logits'];y=z.new_tensor([bool(m['relevant_windows']) for m in metas]);added=constraint(z,y,cfg)
    parts=dict(parts,added=added,total=base+added)
    assert all(torch.isfinite(v) for v in parts.values())
    return parts['total'],parts,y

@torch.no_grad()
def validation(adapter,model,data,cfg,epoch):
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all());modes={m:m.training for m in model.modules()};cm=adapter.criterion.training;sums={};count=0;ss=[];yy=[]
    try:
        model.eval();model.training=True;model.base.training=True;adapter.criterion.eval()
        for start in range(0,len(data),adapter.bs):
            metas,inputs,targets=adapter.collate([data[i] for i in range(start,min(start+adapter.bs,len(data)))]);out=model(**inputs);_,parts,y=objective(adapter,out,metas,targets,cfg,epoch)
            for k,v in parts.items():sums[k]=sums.get(k,0)+float(v)*len(metas)
            count+=len(metas);ss.extend(out['pred_exist_score'].reshape(-1).tolist());yy.extend(y.bool().tolist())
        ss=np.array(ss);yy=np.array(yy)
        return dict(losses={k:v/count for k,v in sums.items()},positive_mean=float(ss[yy].mean()),negative_mean=float(ss[~yy].mean()),mean_gap=float(ss[yy].mean()-ss[~yy].mean()))
    finally:
        for m,mode in modes.items():m.training=mode
        adapter.criterion.train(cm);random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])

def code_hashes():
    files=[EXP/'run.py',EXP/'launch.py',EXP/'report.py',EXP/'PROTOCOL.json',PARENT/'train.py',PARENT/'separation_losses.py',PARENT/'evaluate.py',SRC/'run.py',SRC/'adapters.py',SRC/'model.py',SRC/'evaluation.py',SRC/'independent_metrics.py']+list((EXP/'configs').glob('*.json'))
    return {str(p):sha(p) for p in files}

def preflight(adapter,model,train,val,cfg,directory,initial,code):
    analytical=[]
    for probs,labels in [([.5,.5],[1,0]),([.9,.1],[1,0]),([.2,.3],[1,1]),([.7,.8],[0,0])]:
        z=torch.logit(torch.tensor(probs,dtype=torch.float64)).requires_grad_();y=torch.tensor(labels);loss=constraint(z,y,cfg);g=torch.autograd.grad(loss,z)[0]
        expected=cfg['weight']*cfg['margin']**2 if probs==[.5,.5] else 0.
        assert abs(float(loss)-expected)<1e-12
        if expected:assert g[0]<0 and g[1]>0
        else:assert not g.count_nonzero()
        analytical.append(dict(scores=probs,labels=labels,loss=float(loss),gradient=g.tolist()))
    ids=[next(i for i,m in enumerate(train.data) if f(m)) for f in [lambda m:bool(m['relevant_windows']),lambda m:not m['relevant_windows'],lambda m:len(m['relevant_windows'])>1]]
    model.train();metas,inputs,targets=adapter.collate([train[i] for i in ids]);out=model(**inputs);loss,parts,_=objective(adapter,out,metas,targets,cfg)
    checks=dict(added_to_transformer=BASE.gradient_norm(parts['added'],model.base.transformer.named_parameters()),added_to_geq=BASE.gradient_norm(parts['added'],model.geq.named_parameters()))
    del out,loss,parts
    native=BASE.preflight(adapter,model,train,val,directory);optimizer=BASE.optimizer_for(model);cases={}
    for case,indices in [('positive',[ids[0]]),('negative',[ids[1]]),('multi_gt_mixed',ids),('native_full_batch',list(range(adapter.bs)))]:
        optimizer.zero_grad(set_to_none=True);model.train();metas,inputs,targets=adapter.collate([train[i] for i in indices]);loss,parts,_=objective(adapter,model(**inputs),metas,targets,cfg);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);optimizer.step();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        cases[case]=dict(losses={k:float(v.detach()) for k,v in parts.items()},gradient_norm=float(norm))
    # Verify strict reconstruction after new-loss updates.
    torch.save(dict(model=model.state_dict()),directory/'search_smoke.ckpt');second_adapter,second,_=construct();second.load_state_dict(torch.load(directory/'search_smoke.ckpt',map_location='cpu',weights_only=False)['model'],strict=True)
    from evaluation import predict
    p,_=predict(adapter,model,val,limit=2);q,_=predict(second_adapter,second,val,limit=2);assert p==q
    changed=[n for n,h in PREVIOUS.hashes(model).items() if initial[n]!=h]
    boundary='base.moment_span_embed.' if BACKBONE=='eatr' else 'base.span_embed.'
    assert all(any(n.startswith(group) for n in changed) for group in ['base.transformer.',boundary,'base.exist_head.','geq.'])
    dump(directory/'PREFLIGHT.json',dict(passed=True,analytical=analytical,gradient_checks=checks,cases=cases,native=native,shared_initial_equal_e1=True,strict_prediction_reload=True,changed_parameters=changed,smoke_discarded=True,code_sha256=code))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backbone',choices=['moment_detr','qd_detr','cg_detr','eatr'],required=True);ap.add_argument('--preflight-only',action='store_true');args=ap.parse_args();torch.set_num_threads(2)
    global BACKBONE;BACKBONE=args.backbone
    cfg=json.loads((EXP/f'configs/{args.backbone}.json').read_text());directory=EXP/('preflights' if args.preflight_only else 'runs')/args.backbone;directory.mkdir(parents=True,exist_ok=False)
    dump(directory/'STATUS.json',dict(status='initializing',pid=os.getpid()))
    try:
        code=code_hashes();adapter,model,initial=construct();assert adapter.manifest['sha256']==cfg['initialization']['sha256']
        dump(directory/'CONFIG.json',dict(protocol=cfg,initialization=adapter.manifest,shared_initial_tensor_sha256=initial,trainable_parameters={n:p.numel() for n,p in model.named_parameters()},code_sha256=code))
        train,val=adapter.dataset('train'),adapter.dataset('val');added_cfg=cfg['added_loss']
        if args.preflight_only:
            preflight(adapter,model,train,val,added_cfg,directory,initial,code);dump(directory/'STATUS.json',dict(status='preflight_passed'));return
        gate=json.loads((EXP/f'preflights/{args.backbone}/PREFLIGHT.json').read_text());assert gate['passed'] and gate['code_sha256']==code
        optimizer=BASE.optimizer_for(model);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:1.)
        seed();loader=DataLoader(train,batch_size=adapter.bs,shuffle=True,num_workers=0,collate_fn=list,generator=torch.Generator().manual_seed(2026),drop_last=False);reference_loader=DataLoader(range(len(train)),batch_size=adapter.bs,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(2026),drop_last=False)
        for epoch in range(1,5):
            start=time.monotonic();sums={};count=0;mixed=0;active=0;sequence=[]
            for step,items in enumerate(loader,1):
                model.train();adapter.criterion.train();optimizer.zero_grad(set_to_none=True);metas,inputs,targets=adapter.collate(items);loss,parts,y=objective(adapter,model(**inputs),metas,targets,added_cfg,epoch-1)
                extra=None
                if step==1 or step%25==0:
                    gs=torch.autograd.grad(parts['added'],[p for p in model.parameters() if p.requires_grad],retain_graph=True,allow_unused=True);extra=float(sum((g.detach().square().sum() for g in gs if g is not None),loss.new_zeros(())).sqrt());del gs
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);optimizer.step()
                for k,v in parts.items():sums[k]=sums.get(k,0)+float(v.detach())*len(metas)
                count+=len(metas);mixed+=bool(y.any() and (~y.bool()).any());active+=bool(parts['added'].detach()>0);sequence.extend(m['qid'] for m in metas)
                if step==1 or step%25==0:
                    record=dict(status='training',trial=args.backbone,epoch=epoch,step=step,total_steps=len(loader),loss=float(loss.detach()),added_loss=float(parts['added'].detach()),positive_count=int(y.sum()),gradient_norm=float(norm),added_gradient_norm=extra);dump(directory/'STATUS.json',record)
                    with (directory/'batches.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    print(json.dumps(record),flush=True)
            scheduler.step();orderhash=hashlib.sha256(json.dumps(sequence).encode()).hexdigest();expected=[train.data[i]['qid'] for ids in reference_loader for i in ids.tolist()];assert sequence==expected,'Loader sequence differs from E1 seed protocol'
            vl=validation(adapter,model,val,added_cfg,epoch-1)
            with (directory/'history.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,train_losses={k:v/count for k,v in sums.items()},validation=vl,mixed_batch_fraction=mixed/len(loader),active_batch_fraction=active/len(loader),loader_qid_sha256=orderhash,loader_matches_e1_protocol=True,seconds=time.monotonic()-start))+'\n')
            checkpoint=directory/f'epoch_{epoch:03d}.ckpt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,trial=args.backbone),checkpoint)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True);dump(directory/'CHECKPOINT.json',dict(epoch=4,path=str(checkpoint),sha256=sha(checkpoint),strict_reload=True))
        dump(directory/'STATUS.json',dict(status='evaluating_val',epoch=4))
        from evaluation import predict,evaluate
        from common import OLD
        for split,data in [('val',val),('test',adapter.dataset('test'))]:
            pred,diag=predict(adapter,model,data);evaluate(pred,data.data,directory,split,diag,rows(OLD/f'results/{BACKBONE}/gmr/{split}_predictions.jsonl'))
        dump(directory/'COMPLETE.json',dict(status='complete',epoch=4,backbone=BACKBONE));dump(directory/'STATUS.json',dict(status='complete',epoch=4))
    except Exception:
        dump(directory/'STATUS.json',dict(status='failed',traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
