#!/usr/bin/env python3
"""User-requested continuation: E3 FlashVTG epochs 27..76, exact optimizer/RNG resume."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,fcntl,json,random,subprocess,sys,time,traceback
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import ROOT,OLD,dump,rows,seed,sha
from run_E3 import construct,objective,optimizer_for,validation_loss,tensor_hashes

BASE=ROOT/'runs/flashvtg/E3'
OUT=ROOT/'runs/flashvtg/E3_extra50'
RESUME_EPOCH=26
EXTRA_EPOCHS=50
TARGET_EPOCH=76

def comparison():
    fields=['AUROC','AUPRC','Pos-Recall','TNR','mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3']
    records=[]
    for split in ('val','test'):
        for label,path in [('E3 epoch 26',BASE),('E3 epoch 76',OUT),('E1 epoch 4',ROOT/'runs/flashvtg/E1'),('E2 epoch 4',ROOT/'runs/flashvtg/E2')]:
            metrics=json.loads((path/f'{split}_metrics.json').read_text())
            for tau in ('0.4','0.5','0.6'):
                m=metrics[tau]['brief'];records.append(dict(Method=label,Split=split,Threshold=tau,**{k:m[k] for k in fields},**{'Rej-F1':m[f'Rej-F1@{tau}']}))
    report=ROOT/'reports/FlashVTG_E3_extra50';report.mkdir(exist_ok=True)
    with (report/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    columns=['AUROC','AUPRC','Rej-F1','Pos-Recall','TNR','mAP','mR+@5','G-mIoU@1']
    lines=['# FlashVTG E3：追加 50 轮\n','按用户要求在原 E3 第 26 轮后连续训练第 27–76 轮，使用相同目标、学习率、optimizer/scheduler 和恢复的 RNG；不重新随机初始化，不引入新模块。两个评估 checkpoint 为预先固定的第 26、76 轮，不按 Test 选择。\n','| Method | '+' | '.join(columns)+' |','|'+'---|'*(len(columns)+1)]
    for r in records:
        if r['Split']=='test' and r['Threshold']=='0.4':lines.append('| '+r['Method']+' | '+' | '.join(f'{r[k]:.2f}' for k in columns)+' |')
    lines.append('\n单位 %，Test τ=0.4。原五 backbone 的 E3 主表仍保留统一 26 轮预算；本扩展单独报告，训练预算与其他 backbone 不同。')
    (report/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Preserve 26-epoch curve and append continuation only for this diagnostic.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=rows(BASE/'history.jsonl')+rows(OUT/'history.jsonl')
    assert [x['epoch'] for x in history]==list(range(1,TARGET_EPOCH+1))
    plt.rcParams.update({'pdf.fonttype':42,'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,key in zip(axes,('localization','geq','total')):
        for split,color,style in [('train','#0072B2','-'),('val','#D55E00','--')]:
            ax.plot([x['epoch'] for x in history],[x[split+'_losses'][key] for x in history],color=color,ls=style,label=split.title())
        ax.axvline(26,color='#475569',ls=':',lw=1);ax.set_title(key);ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.set_ylim(bottom=0);ax.legend(frameon=False)
    fig.suptitle('FlashVTG E3: 26 epochs + user-requested 50-epoch continuation');fig.tight_layout()
    dest=ROOT/'figures'
    for ext in ('pdf','png','svg'):fig.savefig(dest/f'FlashVTG_E3_76epoch_loss_curves.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)

def run():
    lock=(ROOT/'flashvtg_E3_extra50.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    OUT.mkdir(parents=True,exist_ok=False)
    dump(OUT/'STATUS.json',dict(status='waiting_for_epoch_26',pid=os.getpid(),additional_epochs=EXTRA_EPOCHS,target_epoch=TARGET_EPOCH))
    try:
        while not (BASE/'COMPLETE.json').exists():
            if (BASE/'STATUS.json').exists() and json.loads((BASE/'STATUS.json').read_text()).get('status')=='failed':
                raise RuntimeError('Source 26-epoch FlashVTG E3 failed; continuation will not start')
            time.sleep(10)
        marker=json.loads((BASE/'COMPLETE.json').read_text());assert marker['epoch']==RESUME_EPOCH
        checkpoint=BASE/'epoch_026.ckpt';meta=json.loads((BASE/'CHECKPOINT.json').read_text());assert sha(checkpoint)==meta['sha256']
        torch.set_num_threads(2);seed();adapter,model=construct('flashvtg','cuda:0')
        initial=tensor_hashes(model)
        original_cfg=json.loads((BASE/'CONFIG.json').read_text());assert initial==original_cfg['initial_tensor_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['epoch']==RESUME_EPOCH and saved['setting']=='E3'
        model.load_state_dict(saved['model'],strict=True)
        optimizer=optimizer_for(model);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:1.)
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        assert [g['lr'] for g in optimizer.param_groups]==[1e-5,3e-5]
        train,val=adapter.dataset('train'),adapter.dataset('val')
        loader=DataLoader(train,batch_size=adapter.bs,shuffle=True,num_workers=0,collate_fn=list,generator=torch.Generator().manual_seed(2026),drop_last=False)
        rng=saved['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']]);loader.generator.set_state(rng['dataloader'].cpu())
        expected={n:t.clone() for n,t in saved['model'].items()}
        assert all(torch.equal(t.cpu(),expected[n]) for n,t in model.state_dict().items());del expected,saved
        dump(OUT/'CONFIG.json',dict(setting='E3_extra50',backbone='flashvtg',seed=2026,resume_checkpoint=str(checkpoint),resume_sha256=meta['sha256'],resume_epoch=RESUME_EPOCH,additional_epochs=EXTRA_EPOCHS,target_epoch=TARGET_EPOCH,resumed_states=['model','optimizer','scheduler','python_rng','numpy_rng','torch_rng','cuda_rng','dataloader_rng'],objective='unchanged native localization + GEQ BCE',learning_rates=[1e-5,3e-5],weight_decay=1e-4,gradient_clip=.1,original_config_sha256=sha(BASE/'CONFIG.json'),source_sha256={n:sha(ROOT/n) for n in ('extend_flashvtg_E3.py','run_E3.py','scratch_adapter.py','model.py','evaluation.py')},evaluation='fixed global epoch 76; automatic Val/Test official and independent verification; epoch26 outputs preserved'))
        dump(OUT/'RESUME_VERIFIED.json',dict(passed=True,strict_model_equal=True,optimizer_scheduler_restored=True,all_rng_restored=True,source_epoch=26))
        for epoch in range(RESUME_EPOCH+1,TARGET_EPOCH+1):
            started=time.monotonic();sums={};count=0
            for step,items in enumerate(loader,1):
                model.train();adapter.criterion.train();optimizer.zero_grad(set_to_none=True)
                metas,inputs,targets=adapter.collate(items);out=model(**inputs)
                loss,parts,ld=objective(adapter,out,metas,targets,epoch-1);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True);optimizer.step()
                values=dict(parts,**{f'native/{k}':v for k,v in ld.items() if torch.is_tensor(v) and v.ndim==0})
                for k,v in values.items():sums[k]=sums.get(k,0)+float(v.detach())*len(items)
                count+=len(items)
                if step==1 or step%25==0:
                    progress=dict(status='training',backbone='flashvtg',setting='E3_extra50',epoch=epoch,target_epoch=TARGET_EPOCH,step=step,total_steps=len(loader),loss=float(loss.detach()),gradient_norm=float(norm))
                    dump(OUT/'STATUS.json',progress);print(json.dumps(progress),flush=True)
            scheduler.step();vl=validation_loss(adapter,model,val,epoch-1)
            history=dict(epoch=epoch,train_losses={k:v/count for k,v in sums.items()},val_losses=vl,seconds=time.monotonic()-started)
            with (OUT/'history.jsonl').open('a') as f:f.write(json.dumps(history)+'\n')
            final=OUT/f'epoch_{epoch:03d}.ckpt'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,setting='E3_extra50',initialization=adapter.manifest,initial_tensor_sha256=initial,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),dataloader=loader.generator.get_state())),final)
        model.load_state_dict(torch.load(final,map_location='cpu',weights_only=False)['model'],strict=True)
        dump(OUT/'CHECKPOINT.json',dict(epoch=TARGET_EPOCH,sha256=sha(final),strict_reload=True))
        dump(OUT/'STATUS.json',dict(status='evaluating',epoch=TARGET_EPOCH))
        from evaluation import predict,evaluate
        for split,data in [('val',val),('test',adapter.dataset('test'))]:
            pred,diag=predict(adapter,model,data);evaluate(pred,data.data,OUT,split,diag,rows(OLD/f'results/flashvtg/gmr/{split}_predictions.jsonl'))
            labels={m['qid']:bool(m['relevant_windows']) for m in data.data}
            dump(OUT/f'{split}_geq_logits.json',{group:[dict(qid=p['qid'],logit=p['pred_exist_logit_raw'],score=p['pred_exist_score']) for p in pred if labels[p['qid']]==positive] for group,positive in [('positive',True),('negative',False)]})
        assert len(rows(OUT/'history.jsonl'))==EXTRA_EPOCHS
        comparison();dump(OUT/'COMPLETE.json',dict(status='complete',epoch=TARGET_EPOCH,additional_epochs=EXTRA_EPOCHS,seed=2026))
        dump(OUT/'STATUS.json',dict(status='complete',epoch=TARGET_EPOCH,finished=time.time()))
    except Exception:
        dump(OUT/'STATUS.json',dict(status='failed',traceback=traceback.format_exc()));raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--launch',action='store_true');args=parser.parse_args()
    if args.launch:
        assert not OUT.exists(),'Continuation already exists; never overwrite it'
        logpath=ROOT/'logs/flashvtg_E3_extra50.log'
        with logpath.open('x') as log:
            process=subprocess.Popen([sys.executable,str(ROOT/'extend_flashvtg_E3.py')],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        dump(ROOT/'FLASHVTG_E3_EXTENSION.json',dict(pid=process.pid,status='waiting_for_epoch_26',source_epochs=26,additional_epochs=50,total_epochs=76,gpu=0,log=str(logpath),status_file=str(OUT/'STATUS.json'),automatic_val_test=True))
        print('CONTINUATION_PID',process.pid)
    else:run()

if __name__=='__main__':main()
