#!/usr/bin/env python3
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse,hashlib,importlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from variants import NativeVariant,ROWS,reference
from official_eval.eval_main import evaluate_gmr
from postprocessing import export

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n');tmp.replace(path)
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def seed():random.seed(2026);np.random.seed(2026);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def loss_for(out,metas,job):
    logits=out['pred_exist_logits'].reshape(-1)
    y=logits.new_tensor([bool(x['relevant_windows']) for x in metas])
    loss=F.binary_cross_entropy_with_logits(logits,y)
    if job['row']>=4:
        loss=loss+job['refine_weight']*reference.compute_boundary_refine_loss(out['refined_spans'],metas,logits.device,job)
    return loss

@torch.no_grad()
def predict(backend,model,data,bs,device,flash=False):
    model.eval()
    if flash:
        from prediction import predictions
        return predictions(model,data,backend.opt)
    result=[]
    for start in range(0,len(data),bs):
        meta,inputs=backend.collate([data[i] for i in range(start,min(start+bs,len(data)))],device)
        result.extend(export(model(**inputs),meta))
    return result,result

def evaluate(pred,gt,output,split):
    assert len(pred)==len(gt) and {p['qid'] for p in pred}=={g['qid'] for g in gt}
    (output/f'{split}_predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pred))
    reports={}
    for tau in [.4,.5,.6]:
        report=evaluate_gmr(pred,gt,cls_thresholds=(tau,),gmiou_cls_threshold=tau,map_num_workers=0,verbose=False)
        negatives={g['qid'] for g in gt if not g['relevant_windows']}
        report['brief']['N-acc']=100*sum(p['pred_exist_score']<=tau for p in pred if p['qid'] in negatives)/len(negatives)
        reports[str(tau)]=report
    dump(output/f'{split}_metrics.json',reports)

def main():
    p=argparse.ArgumentParser();p.add_argument('--backbone',required=True);p.add_argument('--row',type=int,required=True,choices=list(ROWS));p.add_argument('--device',default='cuda:0');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);seed()
    output=ROOT/'runs'/a.backbone/f'row_{a.row:02d}'
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    job=json.loads((ROOT/'config.json').read_text());job['row']=a.row
    flash=a.backbone=='flashvtg'
    if flash:
        sys.path.insert(0,str(ROOT/'code'))
        import flash_backend as backend
        model,init=backend.initialize(a.device,job,a.row)
    else:
        sys.path.insert(0,str(ROOT/'backends'/a.backbone))
        backend=importlib.import_module('backend');inner,init=backend.initialize(a.device,job)
        model=NativeVariant(inner,a.row).to(a.device)
    bs=int(init['baseline_config']['bsz']);eval_bs=1 if flash else int(init['baseline_config'].get('eval_bsz',bs))
    frozen={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}
    dump(output/'CONFIG.json',dict(job=job,label=ROWS[a.row],initialization=init,trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),batch_size=bs,selection='fixed epoch 4; baseline row frozen',test_selection=False,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*.py') if '__pycache__' not in str(p)}))
    train=backend.get_dataset('train');val=backend.get_dataset('val')
    assert len(train)==4138 and len(val)==465
    # Forward, gradient, masking, and exact checkpoint reload checks on mixed samples.
    idx=[next(i for i,m in enumerate(train.data) if len(m['relevant_windows'])==n) for n in [0,1,2]]
    metas,inputs=backend.collate([train[i] for i in idx],a.device)
    model.train();out=model(**inputs)
    assert torch.isfinite(out['pred_exist_logits']).all()
    if a.row>1:
        loss=loss_for(out,metas,job);loss.backward()
        assert any(p.grad is not None and bool(p.grad.abs().max()>0) for p in model.parameters() if p.requires_grad)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        model.zero_grad(set_to_none=True)
    model.eval()
    if flash:metas,inputs=backend.collate([train[idx[1]]],a.device)
    with torch.no_grad():expected=model(**inputs)['pred_exist_logits'].clone()
    state={n:t.detach().cpu().clone() for n,t in model.state_dict().items()};model.load_state_dict(state,strict=True)
    with torch.no_grad():assert torch.equal(expected,model(**inputs)['pred_exist_logits'])
    dump(output/'PREFLIGHT.json',dict(passed=True,mixed_positive_negative=True,strict_reload_equal=True,frozen_no_grad=True))
    print('PREFLIGHT_PASSED',a.backbone,a.row,flush=True)
    if a.preflight_only:return
    if a.row>1:
        groups=[{'params':[p for n,p in model.named_parameters() if p.requires_grad and '.lora_' not in n],'lr':job['lr']},{'params':[p for n,p in model.named_parameters() if p.requires_grad and '.lora_' in n],'lr':job['lora_lr']}]
        opt=torch.optim.AdamW(groups,weight_decay=job['wd'],foreach=False)
        scheduler=torch.optim.lr_scheduler.StepLR(opt,job['lr_drop'],gamma=.5)
        seed();loader=DataLoader(train,batch_size=bs,shuffle=True,collate_fn=list,num_workers=0,generator=torch.Generator().manual_seed(2026))
        for epoch in range(1,job['epochs']+1):
            begun=time.monotonic();total=0
            for step,items in enumerate(loader,1):
                model.train();metas,inputs=backend.collate(items,a.device);opt.zero_grad(set_to_none=True)
                out=model(**inputs);loss=loss_for(out,metas,job)
                assert torch.isfinite(loss);loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],job['grad_clip'],error_if_nonfinite=True);opt.step();total+=float(loss.detach())
                if step==1 or step%50==0:
                    record=dict(backbone=a.backbone,row=a.row,epoch=epoch,step=step,total_steps=len(loader),loss=float(loss.detach()))
                    dump(output/'progress.json',record);print(json.dumps(record),flush=True)
            scheduler.step()
            for n,p in model.named_parameters():
                if n in frozen:assert p.grad is None and torch.equal(p.detach().cpu(),frozen[n]),n
            with (output/'history.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,mean_loss=total/len(loader),seconds=time.monotonic()-begun))+'\n')
        checkpoint=output/'epoch_004.ckpt'
        torch.save(dict(model=model.state_dict(),job=job,initialization=init,epoch=4),checkpoint)
        dump(output/'CHECKPOINT.json',dict(path=str(checkpoint),sha256=sha(checkpoint),epoch=4,selection='fixed in config before training'))
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
    for split,data in [('val',val),('test',backend.get_dataset('test'))]:
        _,pred=predict(backend,model,data,eval_bs,a.device,flash)
        evaluate(pred,data.data,output,split)
    dump(output/'COMPLETE.json',dict(status='complete',backbone=a.backbone,row=a.row,epochs=0 if a.row==1 else 4,frozen_unchanged=True))
    print('COMPLETE',a.backbone,a.row,flush=True)
if __name__=='__main__':main()
