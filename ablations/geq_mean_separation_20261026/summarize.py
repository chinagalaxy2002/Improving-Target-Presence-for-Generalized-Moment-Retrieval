#!/usr/bin/env python3
"""Locked partial/final E0/E1/GAP/ENDPOINT reporting; fixed epoch four only."""
import csv,fcntl,json,time
from pathlib import Path
import numpy as np
from evaluate import EXP,SRC,OLD,rows,aligned,measure,dump,digest
COLORS={'E0':'#666666','E1':'#0072B2','GAP':'#D55E00','ENDPOINT':'#009E73'}
STYLES={'E0':':','E1':'--','GAP':'-','ENDPOINT':'-.'}
KEYS=['AUROC','AUPRC','Rej-F1','Pos-Recall','TNR','mAP','mR@1','mR@5','mR+@5','Positive-only Localization Recall@5/IoU0.5','G-mIoU@1','G-mIoU@3','mu_pos','mu_neg','D_pos','D_neg','D_balanced','S','mean_gap']

def collect():
    found=[];predictions={};complete=[]
    paths=[('E0',OLD/'results/flashvtg/gmr'),('E1',SRC/'runs/flashvtg/E1')]
    for name in ['GAP','ENDPOINT']:
        path=EXP/'runs'/name
        if (path/'COMPLETE.json').exists():
            assert json.loads((path/'COMPLETE.json').read_text())['epoch']==4
            assert json.loads((path/'ENDPOINT_AUDIT.json').read_text())['passed'];paths.append((name,path));complete.append(name)
    for name,path in paths:
        for split in ['val','test']:
            gt=rows(SRC/f'data/{split}.jsonl');pred=aligned(path/f'{split}_predictions.jsonl',gt);endpoint=measure(pred,gt);official=json.loads((path/f'{split}_metrics.json').read_text())
            predictions[name,split]=(pred,gt)
            for tau in ['0.4','0.5','0.6']:
                b=official[tau]['brief'];record=dict(method=name,split=split,threshold=float(tau),epoch='pretrained' if name=='E0' else 4)
                record.update({k:v for k,v in endpoint.items() if k!='distribution'})
                record['Rej-F1']=b[f'Rej-F1@{tau}']
                for k in ['Pos-Recall','TNR','mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3']:
                    if k in b:record[k]=b[k]
                y=np.array([bool(g['relevant_windows']) for g in gt]);s=np.array([p['pred_exist_score'] for p in pred]);gate=s>float(tau)
                record['Pos-Recall']=100*float(gate[y].mean());record['TNR']=100*float((~gate[~y]).mean())
                loc=b.get('Positive-only Localization Recall@5/IoU0.5')
                if loc is None:
                    from evaluate import hit
                    loc=100*np.mean([hit(p,g,5) for p,g in zip(pred,gt) if g['relevant_windows']])
                record['Positive-only Localization Recall@5/IoU0.5']=float(loc);found.append(record)
    for r in found:
        for ref in ['E0','E1']:
            baseline=next(x for x in found if x['method']==ref and x['split']==r['split'] and x['threshold']==r['threshold'])
            for k in ['S','AUROC','AUPRC','Pos-Recall','TNR','mAP','mR+@5']:r[f'delta_{k}_vs_{ref}']=r[k]-baseline[k]
    return found,predictions,complete

def figures(data,predictions,complete):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,precision_recall_curve
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    directory=EXP/'figures';directory.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(directory/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    for split in ['val','test']:
        selected=[r for r in data if r['split']==split and r['threshold']==.4];names=[r['method'] for r in selected]
        fig,axes=plt.subplots(2,4,figsize=(13,7))
        for ax,key in zip(axes.flat,['S','AUROC','AUPRC','Pos-Recall','TNR','mAP','mR+@5','G-mIoU@1']):
            vals=[r[key] for r in selected];ax.bar(range(len(names)),vals,color=[COLORS[n] for n in names])
            for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.2f}',ha='center',fontsize=9)
            ax.set(xticks=range(len(names)),xticklabels=names,ylim=(0,100),title=key,ylabel='Percent');ax.grid(axis='y',alpha=.15)
        fig.suptitle(f'FlashVTG {split.title()}: fixed 4-epoch fine tuning; threshold = 0.4',fontsize=12);fig.tight_layout();save(fig,f'{split}_metrics')
        fig,axes=plt.subplots(1,2,figsize=(11,4.5));dist,da=plt.subplots(1,2,figsize=(11,4.5));logfig,la=plt.subplots(1,2,figsize=(11,4.5))
        for name in names:
            pred,gt=predictions[name,split];y=np.array([bool(g['relevant_windows']) for g in gt]);s=np.array([p['pred_exist_score'] for p in pred]);fpr,tpr,_=roc_curve(y,s);pr,re,_=precision_recall_curve(y,s)
            axes[0].plot(fpr,tpr,color=COLORS[name],ls=STYLES[name],label=name);axes[1].plot(re,pr,color=COLORS[name],ls=STYLES[name],label=name)
            for ax,mask in zip(da,[y,~y]):
                dens,bins=np.histogram(s[mask],bins=np.linspace(0,1,41),density=True);ax.stairs(dens,bins,color=COLORS[name],ls=STYLES[name],label=name,linewidth=1.7)
            if all('pred_exist_logit_raw' in p for p in pred):
                z=np.array([p['pred_exist_logit_raw'] for p in pred]);low=min(-5,float(z.min()));high=max(5,float(z.max()))
                for ax,mask in zip(la,[y,~y]):
                    dens,bins=np.histogram(z[mask],bins=np.linspace(low,high,51),density=True);ax.stairs(dens,bins,color=COLORS[name],ls=STYLES[name],label=name)
        axes[0].plot([0,1],[0,1],color='gray',ls=':',lw=.8);axes[1].axhline(y.mean(),color='gray',ls=':',lw=.8)
        for ax,xl,yl,title in zip(axes,['False positive rate','Recall'],['True positive rate','Precision'],['ROC','Precision–Recall']):ax.set(xlim=(0,1),ylim=(0,1.01),xlabel=xl,ylabel=yl,title=title);ax.legend();ax.grid(alpha=.15)
        fig.tight_layout();save(fig,f'{split}_ranking_curves')
        for ax,title in zip(da,[f'Positive queries (n={y.sum()})',f'Negative queries (n={(~y).sum()})']):ax.axvline(.4,color='gray',ls=':',lw=.8);ax.set(xlim=(0,1),xlabel='Existence score',ylabel='Density',title=title);ax.legend();ax.grid(alpha=.15)
        dist.tight_layout();save(dist,f'{split}_score_distributions')
        for ax,title in zip(la,['Positive queries','Negative queries']):ax.set(xlabel='Raw existence logit',ylabel='Density',title=title);ax.legend();ax.grid(alpha=.15)
        logfig.tight_layout();save(logfig,f'{split}_logit_distributions')
    histories={name:rows(EXP/f'runs/{name}/history.jsonl') for name in complete};fig,axes=plt.subplots(2,3,figsize=(12,7))
    for name,history in histories.items():
        x=[h['epoch'] for h in history]
        for ax,key in zip(axes.flat,['native','geq','added','total']):
            ax.plot(x,[h['train_losses'][key] for h in history],color=COLORS[name],ls='-',marker='o',label=name+' train');ax.plot(x,[h['validation']['losses'][key] for h in history],color=COLORS[name],ls='--',marker='s',label=name+' Val');ax.set(title=key,xlabel='Epoch',ylabel='Loss',xticks=x)
        axes[1,1].plot(x,[h['mixed_batch_fraction'] for h in history],color=COLORS[name],ls='-',marker='o',label=name+' mixed');axes[1,1].plot(x,[h['active_batch_fraction'] for h in history],color=COLORS[name],ls='--',marker='s',label=name+' active');axes[1,1].set(title='Batch fractions',xlabel='Epoch',ylabel='Fraction',ylim=(0,1.05))
        axes[1,2].plot(x,[h['validation']['positive_mean'] for h in history],color=COLORS[name],ls='-',marker='o',label=name+' positive');axes[1,2].plot(x,[h['validation']['negative_mean'] for h in history],color=COLORS[name],ls='--',marker='s',label=name+' negative');axes[1,2].set(title='Val class means',xlabel='Epoch',ylabel='Mean score',ylim=(0,1))
    for ax in axes.flat:
        if ax.lines:ax.legend(fontsize=7)
        ax.grid(alpha=.15)
    fig.tight_layout();save(fig,'training_curves')

def main():
    reports=EXP/'reports';reports.mkdir(exist_ok=True)
    with (reports/'.summary.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data,predictions,complete=collect()
        with (reports/'all_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
        state='complete' if len(complete)==2 else 'partial'
        lines=['# FlashVTG 正负均分监督实验','',f'状态：{state}；已完成：'+(', '.join(complete) or '无新训练完成')+'。未完成实验不填充结果。','', 'E0复用原始GMR；E1复用GEQ微调4轮；GAP/ENDPOINT独立原始checkpoint初始化、固定4轮，无PV。S=100*(1−(D_pos+D_neg)/2)，正负类别等权；S/mean_gap是同一信息，不替代AUROC/AUPRC。','', '| Method | S ↑ | D_pos ↓ | D_neg ↓ | AUROC | AUPRC | Pos-Recall | TNR | mAP | mR+@5 | G-mIoU@1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        selected=[r for r in data if r['split']=='test' and r['threshold']==.4]
        for r in selected:lines.append('| '+r['method']+' | '+' | '.join(f'{r[k]:.4f}' if k.startswith('D_') else f'{r[k]:.2f}' for k in ['S','D_pos','D_neg','AUROC','AUPRC','Pos-Recall','TNR','mAP','mR+@5','G-mIoU@1'])+' |')
        lines+=['','Test阈值0.4；完整Val/Test与0.5/0.6见all_metrics.csv；定位mAP/mR不受存在性hard gate影响。','', '## 实测变化（相对E1）','']
        conclusions=[]
        for name in complete:
            r=next(r for r in selected if r['method']==name);h=rows(EXP/f'runs/{name}/history.jsonl');first=h[0];last=h[-1]
            sentence=f"{name}：ΔS={r['delta_S_vs_E1']:+.2f}，ΔAUROC={r['delta_AUROC_vs_E1']:+.2f}，ΔAUPRC={r['delta_AUPRC_vs_E1']:+.2f}，Δ正例存在性召回={r['delta_Pos-Recall_vs_E1']:+.2f}，ΔTNR={r['delta_TNR_vs_E1']:+.2f}，ΔmAP={r['delta_mAP_vs_E1']:+.2f}，ΔmR+@5={r['delta_mR+@5_vs_E1']:+.2f} 个百分点。"
            lines.append(sentence);conclusions.append(sentence)
            lines.append(f"第1/4轮混合batch比例={first['mixed_batch_fraction']:.3f}/{last['mixed_batch_fraction']:.3f}，约束激活比例={first['active_batch_fraction']:.3f}/{last['active_batch_fraction']:.3f}。正负类误差分别见表；不能把分数尺度变化直接称为排序提升。")
        lines+=['','preflight验证新增损失梯度路径；逐batch新增损失、梯度与激活记录位于runs/*/batches.jsonl。相对E1的正例存在性/定位lost/restored qid位于endpoint_metrics.json。没有Test选epoch、调阈值、调权重或追加训练。单seed，不报告统计显著性。']
        (reports/'RESULTS.md').write_text('\n'.join(lines)+'\n');(EXP/'CONCLUSIONS.md').write_text('# 实测结论\n\n'+ ('\n\n'.join(conclusions) if conclusions else '训练尚未完成，不作性能结论。')+'\n\n单seed固定预算探索；正负均分更分离不等价于排序或定位必然改善。\n')
        figures(data,predictions,complete)
        audit={name:json.loads((EXP/f'runs/{name}/ENDPOINT_AUDIT.json').read_text()) for name in complete};dump(reports/'AUDIT.json',dict(status=state,completed=complete,all_completed_audits_passed=all(x['passed'] for x in audit.values()),audits=audit))
        dump(EXP/'STATUS.json',dict(status=state,completed=complete,pending=[n for n in ['GAP','ENDPOINT'] if n not in complete],updated=time.time(),automatic_evaluation=True))
if __name__=='__main__':main()
