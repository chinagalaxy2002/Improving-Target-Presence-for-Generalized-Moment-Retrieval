#!/usr/bin/env python3
"""Two fixed pretrained FlashVTG + GEQ mean-supervision experiments."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,random,subprocess,sys,time,traceback
from pathlib import Path
EXP=Path(__file__).resolve().parent;SRC=EXP.parent/'end_to_end_geq_pv_20261026'
from separation_losses import mean_constraint,analytical_checks
sys.path.insert(0,str(SRC))
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import OLD,dump,rows,seed,sha,module
from adapters import Adapter
from model import JointGMR
BASE=module('verified_e1_training',SRC/'run.py')

def hashes(model):return {n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,p in model.state_dict().items()}

def provenance():
    planned=json.loads((EXP/'SOURCE_MANIFEST.json').read_text());actual={n:sha(SRC/n) for n in planned['sha256']}
    assert actual==planned['sha256'],'Source changed since fixed plan'
    return {str(p):sha(p) for p in [EXP/'train.py',EXP/'separation_losses.py',EXP/'evaluate.py',EXP/'summarize.py',SRC/'run.py',SRC/'model.py',SRC/'adapters.py',SRC/'evaluation.py',SRC/'independent_metrics.py',EXP/'configs/GAP.json',EXP/'configs/ENDPOINT.json']}

def objective(adapter,out,meta,targets,setting,epoch=0):
    base,parts,ld=BASE.objectives(adapter,out,meta,targets,epoch)
    z=out['pred_exist_logits'];labels=z.new_tensor([bool(m['relevant_windows']) for m in meta])
    added,pos,neg,mixed=mean_constraint(z,labels,setting)
    parts=dict(parts,added=added,total=base+added)
    assert all(torch.isfinite(v) for v in parts.values())
    return parts['total'],parts,ld,dict(mixed=mixed,positive_count=int(labels.sum()),negative_count=len(meta)-int(labels.sum()),positive_mean=None if pos is None else float(pos.detach()),negative_mean=None if neg is None else float(neg.detach()),constraint_active=bool(added.detach()>0))

def construct():
    seed();adapter=Adapter('flashvtg','cuda:0');model=JointGMR(adapter.base,'flashvtg',False).to('cuda:0')
    reference=json.loads((SRC/'runs/flashvtg/E1/CONFIG.json').read_text())
    initial=hashes(model);assert initial==reference['shared_initial_tensor_sha256']
    assert adapter.bs==8 and all(p.requires_grad for p in model.parameters())
    return adapter,model,initial

@torch.no_grad()
def validation(adapter,model,data,setting,epoch):
    rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all());modes={m:m.training for m in model.modules()};criterion_mode=adapter.criterion.training;sums={};count=0;positive=[];negative=[]
    try:
        model.eval();model.training=True;model.base.training=True;adapter.criterion.eval()
        for start in range(0,len(data),8):
            meta,inputs,targets=adapter.collate([data[i] for i in range(start,min(start+8,len(data)))]);out=model(**inputs)
            _,parts,_,_=objective(adapter,out,meta,targets,setting,epoch)
            for k,v in parts.items():sums[k]=sums.get(k,0)+float(v)*len(meta)
            scores=out['pred_exist_score'].reshape(-1).tolist()
            for m,s in zip(meta,scores):(positive if m['relevant_windows'] else negative).append(s)
            count+=len(meta)
        return dict(losses={k:v/count for k,v in sums.items()},positive_mean=float(np.mean(positive)),negative_mean=float(np.mean(negative)),mean_gap=float(np.mean(positive)-np.mean(negative)),loss_aggregation='batch-size weighted; class means aggregated across whole split')
    finally:
        for m,mode in modes.items():m.training=mode
        adapter.criterion.train(criterion_mode)
        random.setstate(rng[0]);np.random.set_state(rng[1]);torch.set_rng_state(rng[2]);torch.cuda.set_rng_state_all(rng[3])

def preflight(adapter,model,train,val,setting,directory,initial,code):
    formulas=analytical_checks(setting)
    ids=[next(i for i,m in enumerate(train.data) if f(m)) for f in [lambda m:bool(m['relevant_windows']),lambda m:not m['relevant_windows'],lambda m:len(m['relevant_windows'])>1]]
    # Native common preflight verifies head gradients, conditional inactive branches,
    # optimizer, strict rebuild, temporal coordinates and official export.
    model.train();meta,inputs,targets=adapter.collate([train[i] for i in ids]);out=model(**inputs)
    loss,parts,_,info=objective(adapter,out,meta,targets,setting)
    assert info['mixed'] and info['constraint_active']
    checks={'added_to_transformer':BASE.gradient_norm(parts['added'],model.base.transformer.named_parameters()),'added_to_geq':BASE.gradient_norm(parts['added'],model.geq.named_parameters())}
    del out,loss,parts
    native=BASE.preflight(adapter,model,train,val,directory)
    optimizer=BASE.optimizer_for(model);cases={}
    for case,which in [('positive',[ids[0]]),('negative',[ids[1]]),('mixed_multiple_gt',ids),('native_batch8',list(range(8)))]:
        model.train();optimizer.zero_grad(set_to_none=True);meta,inputs,targets=adapter.collate([train[i] for i in which])
        loss,parts,_,info=objective(adapter,model(**inputs),meta,targets,setting);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);optimizer.step()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        cases[case]=dict(losses={k:float(v.detach()) for k,v in parts.items()},gradient_norm=float(norm),**info)
    # Rebuild/reload after the actual new-loss updates as well.
    torch.save(dict(model=model.state_dict(),smoke_only=True),directory/'new_loss_smoke.ckpt')
    meta,inputs,_=adapter.collate([val[0]]);model.eval()
    with torch.no_grad():before=model(**inputs)
    other_adapter,other,_=construct();other.load_state_dict(torch.load(directory/'new_loss_smoke.ckpt',map_location='cpu',weights_only=False)['model'],strict=True);other.eval()
    with torch.no_grad():after=other(**inputs)
    assert all(torch.equal(before[k],after[k]) for k in ['pred_exist_logits','candidate_seconds'])
    from evaluation import predict
    pred1,_=predict(adapter,model,val,limit=2);pred2,_=predict(other_adapter,other,val,limit=2);assert pred1==pred2
    del other,other_adapter
    vl=validation(adapter,model,val,setting,0)
    final=hashes(model);changed=[n for n in initial if initial[n]!=final[n]]
    assert all(any(n.startswith(group) for n in changed) for group in ['base.transformer.','base.coord_head.','base.exist_head.','geq.'])
    dump(directory/'PREFLIGHT.json',dict(passed=True,setting=setting,analytical_checks=formulas,gradient_checks=checks,cases=cases,native_preflight=native,strict_new_loss_reload=True,official_prediction_reload_equal=True,validation=vl,initial_equal_e1=True,smoke_discarded=True,changed_parameters=changed,code_sha256=code))

def launch(setting,gpu):
    # GPU is selected explicitly; never terminate other experiments.
    uuid=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
    assert sum(uuid in l for l in lines)<2,'GPU already has two tasks'
    assert not (EXP/f'runs/{setting}').exists()
    (EXP/'logs').mkdir(exist_ok=True)
    with (EXP/f'logs/{setting}.log').open('x') as f:
        proc=subprocess.Popen([sys.executable,str(EXP/'train.py'),'--setting',setting],cwd=EXP,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1'),stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    dump(EXP/f'JOB_{setting}.json',dict(pid=proc.pid,gpu=gpu,setting=setting,epochs=4,automatic_val_test=True));print('LAUNCHED',setting,proc.pid,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--setting',choices=['GAP','ENDPOINT'],required=True);p.add_argument('--preflight-only',action='store_true');p.add_argument('--launch',action='store_true');p.add_argument('--gpu',type=int,default=0);args=p.parse_args()
    if args.launch:return launch(args.setting,args.gpu)
    torch.set_num_threads(2);code=provenance();setting=args.setting;cfg=json.loads((EXP/f'configs/{setting}.json').read_text())
    directory=EXP/('preflights' if args.preflight_only else 'runs')/setting;directory.mkdir(parents=True,exist_ok=False)
    dump(directory/'STATUS.json',dict(status='initializing',pid=os.getpid(),setting=setting))
    try:
        assert sha(Path(cfg['initialization']['checkpoint']))==cfg['initialization']['sha256']
        adapter,model,initial=construct();assert adapter.manifest['sha256']==cfg['initialization']['sha256']
        dump(directory/'CONFIG.json',dict(protocol=cfg,initialization=adapter.manifest,shared_initial_tensor_sha256=initial,trainable_parameters={n:p.numel() for n,p in model.named_parameters()},code_sha256=code,source_manifest_verified=True,checkpoint_bytes_reverified=True))
        train,val=adapter.dataset('train'),adapter.dataset('val')
        if args.preflight_only:
            preflight(adapter,model,train,val,setting,directory,initial,code);dump(directory/'STATUS.json',dict(status='preflight_passed'));print('PREFLIGHT_PASSED',setting,flush=True);return
        gate=json.loads((EXP/f'preflights/{setting}/PREFLIGHT.json').read_text());assert gate['passed'] and gate['code_sha256']==code
        optimizer=BASE.optimizer_for(model);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:1.)
        seed();loader=DataLoader(train,batch_size=8,shuffle=True,num_workers=0,collate_fn=list,generator=torch.Generator().manual_seed(2026),drop_last=False)
        reference_loader=DataLoader(range(len(train)),batch_size=8,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(2026),drop_last=False)
        for epoch in range(1,5):
            start=time.monotonic();sums={};count=0;nb=0;mixed=0;active=0;sequence=[]
            for step,items in enumerate(loader,1):
                model.train();adapter.criterion.train();optimizer.zero_grad(set_to_none=True);meta,inputs,targets=adapter.collate(items);out=model(**inputs)
                loss,parts,ld,info=objective(adapter,out,meta,targets,setting,epoch-1)
                extra_norm=None
                if step==1 or step%25==0:
                    gg=torch.autograd.grad(parts['added'],[p for p in model.parameters() if p.requires_grad],retain_graph=True,allow_unused=True)
                    extra_norm=float(torch.sqrt(sum((g.detach().square().sum() for g in gg if g is not None),loss.detach().new_zeros(()))))
                    del gg
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);optimizer.step()
                values=dict(parts,**{f'native/{k}':v for k,v in ld.items() if torch.is_tensor(v) and v.ndim==0})
                for k,v in values.items():sums[k]=sums.get(k,0)+float(v.detach())*len(meta)
                count+=len(meta);nb+=1;mixed+=info['mixed'];active+=info['constraint_active'];sequence.extend(m['qid'] for m in meta)
                batch_record=dict(epoch=epoch,step=step,**info,added_loss=float(parts['added'].detach()),gradient_norm=float(norm),added_gradient_norm=extra_norm)
                with (directory/'batches.jsonl').open('a') as f:f.write(json.dumps(batch_record)+'\n')
                if step==1 or step%25==0:
                    record=dict(status='training',setting=setting,total_steps=len(loader),loss=float(loss.detach()),**batch_record);dump(directory/'STATUS.json',record);print(json.dumps(record),flush=True)
            scheduler.step()
            expected=[train.data[i]['qid'] for ids in reference_loader for i in ids.tolist()];assert expected==sequence,'Loader order differs from E1 seed protocol'
            vl=validation(adapter,model,val,setting,epoch-1)
            with (directory/'history.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,train_losses={k:v/count for k,v in sums.items()},validation=vl,mixed_batch_fraction=mixed/nb,active_batch_fraction=active/nb,loader_qid_sha256=hashlib.sha256(json.dumps(sequence).encode()).hexdigest(),loader_matches_e1_protocol=True,seconds=time.monotonic()-start))+'\n')
            checkpoint=directory/f'epoch_{epoch:03d}.ckpt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,setting=setting,initialization=adapter.manifest,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),dataloader=loader.generator.get_state())),checkpoint)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
        dump(directory/'CHECKPOINT.json',dict(epoch=4,path=str(checkpoint),sha256=sha(checkpoint),strict_reload=True))
        dump(directory/'STATUS.json',dict(status='evaluating',epoch=4))
        from evaluation import predict,evaluate
        for split,data in [('val',val),('test',adapter.dataset('test'))]:
            pred,diag=predict(adapter,model,data);evaluate(pred,data.data,directory,split,diag,rows(OLD/f'results/flashvtg/gmr/{split}_predictions.jsonl'))
        subprocess.run([sys.executable,str(EXP/'evaluate.py'),'--setting',setting],check=True,cwd=EXP)
        dump(directory/'COMPLETE.json',dict(status='complete',epoch=4,setting=setting,seed=2026));dump(directory/'STATUS.json',dict(status='complete',epoch=4,finished=time.time()))
        subprocess.run([sys.executable,str(EXP/'summarize.py')],check=True,cwd=EXP)
    except Exception:
        dump(directory/'STATUS.json',dict(status='failed',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
