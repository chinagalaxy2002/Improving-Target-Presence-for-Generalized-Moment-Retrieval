"""Read-only snapshot selection by validation loss; isolated Test comparison artifacts."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import json,csv,time
from pathlib import Path
import numpy as np
import torch
from common import ROOT,OLD,dump,rows,sha,seed

OUT=ROOT/'reports/FlashVTG_E4_valbest_snapshot'

def main():
    OUT.mkdir(exist_ok=False)
    run=ROOT/'runs/flashvtg/E4_scratch_flashmmr'
    history=rows(run/'history.jsonl')
    eligible=[h for h in history if (run/f"epoch_{h['epoch']:03d}.ckpt").exists() and np.isfinite(h['val_losses']['total'])]
    best=min(eligible,key=lambda h:(h['val_losses']['total'],h['epoch']))
    ep=best['epoch'];ckpt=run/f'epoch_{ep:03d}.ckpt'
    dump(OUT/'SELECTION.json',dict(rule='minimum validation total loss; ties earliest epoch; no Test selection',snapshot_time=time.time(),eligible_epochs=[h['epoch'] for h in eligible],best=best,checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),provisional=True))
    dump(OUT/'history_snapshot.json',history)
    print('SELECTED',ep,best['val_losses']['total'],flush=True)
    from run_E4_flashmmr import construct
    from evaluation import predict,evaluate
    seed();torch.set_num_threads(2)
    adapter,model=construct()
    state=torch.load(ckpt,map_location='cpu',weights_only=False)
    assert state['epoch']==ep
    model.load_state_dict(state['model'],strict=True);del state
    for split in ['val','test']:
        data=adapter.dataset(split);pred,diag=predict(adapter,model,data)
        evaluate(pred,data.data,OUT,split,diag,rows(OLD/f'results/flashvtg/gmr/{split}_predictions.jsonl'))
    make_plots(ep)
    dump(OUT/'COMPLETE.json',dict(epoch=ep,status='complete',checkpoint_sha256=sha(ckpt)))

def make_plots(ep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    methods=[('E1: pretrained + GEQ (4 ep)',ROOT/'runs/flashvtg/E1','#0072B2','-'),('E3: scratch + GEQ (70 ep)',ROOT/'temp_eval/E3_epoch_070','#E69F00','--'),('E3: scratch + GEQ (76 ep)',ROOT/'runs/flashvtg/E3_extra50','#009E73',':'),(f'E4: scratch + GEQ + PV ({ep} ep)',OUT,'#CC79A7','-.')]
    keys=['AUROC','AUPRC','Pos-Recall','TNR','Rej-F1@0.4','mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3','Positive-only Localization Recall@5/IoU0.5']
    metrics=[json.loads((p/'test_metrics.json').read_text())['0.4']['brief'] for _,p,_,_ in methods]
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['method']+keys);w.writeheader()
        for (label,_,_,_),m in zip(methods,metrics):w.writerow(dict(method=label,**{k:m[k] for k in keys}))
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(14,7))
    for ax,key in zip(axes.flat,['AUROC','AUPRC','Pos-Recall','TNR','mAP','mR+@5','G-mIoU@1','G-mIoU@3']):
        vals=[m[key] for m in metrics]
        ax.bar(np.arange(4),vals,color=[m[2] for m in methods],width=.65)
        for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.2f}',ha='center',fontsize=9)
        ax.set(xticks=range(4),xticklabels=['E1 / 4','E3 / 70','E3 / 76',f'E4 / {ep}'],ylim=(0,100),title=key,ylabel='Score (%)');ax.grid(axis='y',alpha=.18)
    fig.suptitle('FlashVTG Test: E4 selected by minimum Val total loss (snapshot)\nClassification threshold = 0.4; training budgets differ',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));save(fig,'metrics_comparison')
    gt=rows(ROOT/'data/test.jsonl');labels={g['qid']:bool(g['relevant_windows']) for g in gt}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5));dist,da=plt.subplots(1,2,figsize=(11,4.5))
    for label,p,color,style in methods:
        preds=rows(p/'test_predictions.jsonl');assert len(preds)==len(labels) and {p['qid'] for p in preds}==set(labels)
        y=np.array([labels[p['qid']] for p in preds]);s=np.array([p['pred_exist_score'] for p in preds])
        fpr,tpr,_=roc_curve(y,s);pr,re,_=precision_recall_curve(y,s)
        axes[0].plot(fpr,tpr,color=color,ls=style,label=f'{label}: {100*roc_auc_score(y,s):.2f}')
        axes[1].plot(re,pr,color=color,ls=style,label=f'{label}: {100*average_precision_score(y,s):.2f}')
        for ax,mask in zip(da,[y,~y]):
            density,bins=np.histogram(s[mask],bins=np.linspace(0,1,41),density=True)
            ax.stairs(density,bins,color=color,ls=style,label=label,linewidth=1.7)
    axes[0].plot([0,1],[0,1],color='gray',lw=.7,ls=':')
    axes[1].axhline(np.mean(y),color='gray',lw=.7,ls=':')
    for ax,xl,yl,title in zip(axes,['False positive rate','Recall'],['True positive rate','Precision'],['ROC (legend: AUROC %)','Precision–Recall (legend: AP %)']):
        ax.set(xlim=(0,1),ylim=(0,1.01),xlabel=xl,ylabel=yl,title=title);ax.legend(fontsize=7,loc='lower left');ax.grid(alpha=.15)
    fig.tight_layout();save(fig,'ranking_curves')
    for ax,title in zip(da,['Positive queries (n=544)','Negative queries (n=492)']):
        ax.axvline(.4,color='gray',ls=':',lw=1);ax.set(xlim=(0,1),xlabel='GEQ existence score',ylabel='Density',title=title);ax.legend(fontsize=7);ax.grid(alpha=.15)
    dist.tight_layout();save(dist,'existence_score_distributions')
    lines=['# FlashVTG E4：Val-best 临时评估','',f'E4 在快照已完成轮次中按最低 Val 总损失选择第 {ep} 轮；未使用 Test 选择。原定第76轮正式协议保持不变。E1 是预训练 GMR+GEQ 微调4轮，不是纯 E0。E3/E4 为随机初始化；训练预算不同，E1 另保留 native presence BCE。','', '| Method | AUROC | AUPRC | Pos-Recall | TNR | mAP | mR+@5 | G-mIoU@1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for (label,_,_,_),m in zip(methods,metrics):lines.append('| '+label+' | '+' | '.join(f'{m[k]:.2f}' for k in ['AUROC','AUPRC','Pos-Recall','TNR','mAP','mR+@5','G-mIoU@1'])+' |')
    lines+=['','单位 %，Test 阈值0.4；定位 mAP/mR 指标独立于存在性 hard gating。','', '图：metrics_comparison.pdf、ranking_curves.pdf、existence_score_distributions.pdf；均同时保存 SVG/PNG。','选择、checkpoint 哈希与候选轮次见 SELECTION.json；官方/独立复算见 test_diagnostics.json。']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
