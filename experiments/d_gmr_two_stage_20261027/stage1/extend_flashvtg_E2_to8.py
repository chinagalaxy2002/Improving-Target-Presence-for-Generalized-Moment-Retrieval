#!/usr/bin/env python3
"""One E2 continuation, fixed total-epoch endpoints 6 and 8; no model selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,random,time,traceback,fcntl
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import ROOT,OLD,dump,rows,seed,sha,module
# Explicit path prevents a native backend's run.py from shadowing this trainer.
joint=module('e2_long_original_trainer',ROOT/'run.py')
from adapters import Adapter
from model import JointGMR
BASE=ROOT/'runs/flashvtg/E2'
OUT=ROOT/'runs/flashvtg/E2_long_6_8'

def main():
    lock=(ROOT/'flashvtg_E2_long_6_8.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        dump(OUT/'STATUS.json',dict(status='initializing',pid=os.getpid()))
        torch.set_num_threads(2);seed()
        cfg=json.loads((BASE/'CONFIG.json').read_text())
        gate=json.loads((ROOT/'preflights/flashvtg/E2/PREFLIGHT.json').read_text())
        assert gate['passed']
        for name in ('run.py','model.py','pv.py','adapters.py','evaluation.py'):
            assert sha(ROOT/name)==cfg['source_sha256'][name],name+' changed since original E2'
        checkpoint=BASE/'epoch_004.ckpt'
        original_meta=json.loads((BASE/'CHECKPOINT.json').read_text())
        assert sha(checkpoint)==original_meta['sha256']
        adapter=Adapter('flashvtg','cuda:0')
        model=JointGMR(adapter.base,'flashvtg',True).to('cuda:0')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['epoch']==4 and saved['setting']=='E2'
        assert adapter.manifest['sha256']==cfg['initialization']['sha256']
        model.load_state_dict(saved['model'],strict=True)
        assert all(torch.equal(t.cpu(),saved['model'][n]) for n,t in model.state_dict().items())
        assert all(p.requires_grad for p in model.parameters())
        optimizer=joint.optimizer_for(model)
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:1.)
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        assert [g['lr'] for g in optimizer.param_groups]==[1e-5,3e-5]
        train=adapter.dataset('train')
        loader=DataLoader(train,batch_size=adapter.bs,shuffle=True,num_workers=0,collate_fn=list,generator=torch.Generator().manual_seed(2026),drop_last=False)
        rng=saved['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']]);loader.generator.set_state(rng['dataloader'].cpu())
        dump(OUT/'CONFIG.json',dict(setting='E2_long_6_8',backbone='flashvtg',seed=2026,resume_epoch=4,total_epochs=8,evaluation_epochs=[6,8],resume_checkpoint=str(checkpoint),resume_sha256=sha(checkpoint),original_initialization=cfg['initialization']['checkpoint'],original_initialization_sha256=cfg['initialization']['sha256'],protocol=cfg['protocol'],protocol_override=dict(epochs=8,selection='fixed epochs 6 and 8; no Test selection'),batch_size=adapter.bs,source_sha256={n:sha(ROOT/n) for n in ('extend_flashvtg_E2_to8.py','run.py','model.py','pv.py','adapters.py','evaluation.py')},objective='unchanged native GMR + GEQ BCE + quality MSE + 0.2 representation CE',resumed_states=['model','optimizer','scheduler','python','numpy','torch','cuda','dataloader']))
        dump(OUT/'RESUME_VERIFIED.json',dict(passed=True,strict_model_equal=True,optimizer_scheduler_restored=True,all_rng_restored=True,all_requires_grad=True,source_preflight_passed=True))
        del saved
        for epoch in range(5,9):
            started=time.monotonic();sums={}
            for step,items in enumerate(loader,1):
                model.train();adapter.criterion.train();optimizer.zero_grad(set_to_none=True)
                meta,inputs,targets=adapter.collate(items);out=model(**inputs)
                loss,parts,ld=joint.objectives(adapter,out,meta,targets,epoch-1);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True)
                optimizer.step()
                values=dict(parts,total=loss)
                for k,v in values.items():sums[k]=sums.get(k,0)+float(v.detach())
                if step==1 or step%25==0:
                    status=dict(status='training',epoch=epoch,total_epochs=8,step=step,total_steps=len(loader),loss=float(loss.detach()),gradient_norm=float(norm))
                    dump(OUT/'STATUS.json',status);print(json.dumps(status),flush=True)
            scheduler.step()
            with (OUT/'history.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,mean_losses={k:v/len(loader) for k,v in sums.items()},seconds=time.monotonic()-started))+'\n')
            if epoch in (6,8):
                path=OUT/f'epoch_{epoch:03d}.ckpt'
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,setting='E2_long_6_8',initialization=adapter.manifest,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),dataloader=loader.generator.get_state())),path)
        # Both endpoints evaluated only after training to preserve its exact RNG trajectory.
        from evaluation import predict,evaluate
        for epoch in (6,8):
            dest=OUT/f'epoch_{epoch:03d}';dest.mkdir()
            path=OUT/f'epoch_{epoch:03d}.ckpt'
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'],strict=True)
            dump(dest/'CHECKPOINT.json',dict(epoch=epoch,path=str(path),sha256=sha(path),strict_reload=True))
            dump(OUT/'STATUS.json',dict(status='evaluating',epoch=epoch))
            for split in ('val','test'):
                data=adapter.dataset(split);pred,diag=predict(adapter,model,data)
                evaluate(pred,data.data,dest,split,diag,rows(OLD/f'results/flashvtg/gmr/{split}_predictions.jsonl'))
            dump(dest/'COMPLETE.json',dict(status='complete',epoch=epoch))
        lines=['# FlashVTG E2：固定总轮次 4 / 6 / 8','', '同一 E2 轨迹从第4轮恢复完整训练状态，只延长训练预算。第6、8轮预先固定，不按 Test 选择。','', '| Epoch | AUROC | AUPRC | Pos-Recall | mAP | mR+@5 | G-mIoU@1 |','|---|---|---|---|---|---|---|']
        for epoch,p in [(4,BASE),(6,OUT/'epoch_006'),(8,OUT/'epoch_008')]:
            b=json.loads((p/'test_metrics.json').read_text())['0.4']['brief']
            lines.append('| '+str(epoch)+' | '+' | '.join(f'{b[k]:.2f}' for k in ['AUROC','AUPRC','Pos-Recall','mAP','mR+@5','G-mIoU@1'])+' |')
        (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        dump(OUT/'COMPLETE.json',dict(status='complete',evaluation_epochs=[6,8],total_epochs=8))
        dump(OUT/'STATUS.json',dict(status='complete',total_epochs=8))
    except Exception:
        dump(OUT/'STATUS.json',dict(status='failed',traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
