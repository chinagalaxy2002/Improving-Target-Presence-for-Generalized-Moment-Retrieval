#!/usr/bin/env python3
"""Whole-split five-backbone E0/E1/GAP means, official metrics and vector figures."""
import csv,json,sys,hashlib
from pathlib import Path
EXP=Path(__file__).resolve().parent;SRC=EXP.parent/'end_to_end_geq_pv_20261026';PARENT=EXP.parent/'geq_mean_separation_20261026';OLD=EXP.parent/'global_hard_roi_20261023'
sys.path.insert(0,str(PARENT))
from evaluate import aligned,rows,measure,dump,compare,digest
BACKBONES=['moment_detr','qd_detr','cg_detr','eatr','flashvtg'];NAMES=dict(zip(BACKBONES,['Moment-DETR','QD-DETR','CG-DETR','EaTR','FlashVTG']))
COLORS={'E0':'#64748b','E1':'#0072B2','GAP':'#D55E00'};STYLES={'E0':'--','E1':'-','GAP':'-.'}

def main():
    report=EXP/'reports';report.mkdir(exist_ok=True);data=[];preds={};manifest=[];lost={}
    for bb in BACKBONES:
        gap=EXP/'runs'/bb if bb!='flashvtg' else EXP.parent/'geq_gap_search_20261026/runs/GAP_m60_w200'
        assert (gap/('COMPLETE.json' if bb!='flashvtg' else 'TEST_COMPLETE.json')).exists()
        for method,path in [('E0',OLD/f'results/{bb}/gmr'),('E1',SRC/f'runs/{bb}/E1'),('GAP',gap)]:
            for split in ['val','test']:
                gtpath=SRC/f'data/{split}.jsonl';gt=rows(gtpath);p=aligned(path/f'{split}_predictions.jsonl',gt);m=measure(p,gt);official=json.loads((path/f'{split}_metrics.json').read_text())
                if method=='GAP':assert json.loads((path/f'{split}_diagnostics.json').read_text())['independent_metric_verification']['passed']
                preds[bb,method,split]=(p,gt)
                for tau in ['0.4','0.5','0.6']:
                    b=official[tau]['brief'];record=dict(backbone=bb,method=method,split=split,threshold=float(tau),**{k:v for k,v in m.items() if k!='distribution'})
                    for k in ['mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3']:record[k]=b[k]
                    record['Rej-F1']=b[f'Rej-F1@{tau}'];positive=[bool(g['relevant_windows']) for g in gt];gate=[x['pred_exist_score']>float(tau) for x in p]
                    record['Pos-Recall']=100*sum(a and y for a,y in zip(gate,positive))/sum(positive);record['TNR']=100*sum(not a and not y for a,y in zip(gate,positive))/(len(positive)-sum(positive))
                    data.append(record)
                manifest.append(dict(backbone=bb,method=method,split=split,predictions=str(path/f'{split}_predictions.jsonl'),sha256=digest(path/f'{split}_predictions.jsonl'),ground_truth_sha256=digest(gtpath)))
                if method=='GAP':lost[bb+'_'+split]=compare(p,preds[bb,'E1',split][0],gt)
    for r in data:
        for ref in ['E0','E1']:
            baseline=next(x for x in data if x['backbone']==r['backbone'] and x['method']==ref and x['split']==r['split'] and x['threshold']==r['threshold'])
            for k in ['mean_gap','S','AUROC','AUPRC','mAP','mR+@5','Pos-Recall','TNR']:r[f'delta_{k}_vs_{ref}']=r[k]-baseline[k]
    with (report/'all_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    with (report/'existence_score_means.csv').open('w') as f:
        ks=['backbone','method','split','mu_pos','mu_neg','mean_gap','D_pos','D_neg','D_balanced','S','delta_mean_gap_vs_E0','delta_mean_gap_vs_E1'];w=csv.DictWriter(f,fieldnames=ks);w.writeheader();w.writerows({k:r[k] for k in ks} for r in data if r['threshold']==.4)
    text=['# GAP 迁移至五个 backbone','', '固定迁移 FlashVTG Val 选定的 margin=0.6、λ=2。其他四个 backbone 各独立原始GMR checkpoint初始化、全量微调4轮；GEQ初始张量与对应E1相同。无PV、无新超参数搜索、无Test选模型。FlashVTG复用已完成结果。','']
    for split in ['test','val']:
        text += [f'## {split.title()} 正负均分及均分差距','','| Backbone | Method | 正样本均分 μ₊ | 负样本均分 μ₋ | 均分差距 ↑ | S ↑ | AUROC | AUPRC | mAP | mR+@5 | G-mIoU@1 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in data:
            if r['split']==split and r['threshold']==.4:text.append('| '+NAMES[r['backbone']]+' | '+r['method']+' | '+' | '.join(f'{r[k]:.4f}' if k in ['mu_pos','mu_neg','mean_gap'] else f'{r[k]:.2f}' for k in ['mu_pos','mu_neg','mean_gap','S','AUROC','AUPRC','mAP','mR+@5','G-mIoU@1'])+' |')
    text+=['','## GAP 相对 E1 的 Test 变化','','| Backbone | Δ均分差距 | ΔS | ΔAUROC | ΔmAP | ΔmR+@5 |','|---|---:|---:|---:|---:|---:|']
    for r in data:
        if r['split']=='test' and r['method']=='GAP' and r['threshold']==.4:text.append('| '+NAMES[r['backbone']]+' | '+' | '.join(f'{r[k]:+.4f}' if k=='delta_mean_gap_vs_E1' else f'{r[k]:+.2f}' for k in ['delta_mean_gap_vs_E1','delta_S_vs_E1','delta_AUROC_vs_E1','delta_mAP_vs_E1','delta_mR+@5_vs_E1'])+' |')
    text+=['','均分差距与S分别为概率差和百分制；S=50*(1+均分差距)，不是两项独立证据。各类均值在整个split计算，不平均batch均值。存在性分类固定阈值0.4；0.5/0.6另存CSV。定位不经过存在性hard gating，G-mIoU与纯定位分开解释。单seed，不报告统计显著性。','', '图：../figures/test_mean_gap_comparison.pdf；../figures/test_existence_score_distributions.pdf；../figures/test_performance_comparison.pdf；Val有对应图。']
    (report/'RESULTS.md').write_text('\n'.join(text)+'\n');dump(report/'LOST_RESTORED_VS_E1.json',lost);dump(report/'SOURCE_MANIFEST.json',manifest)
    figures(data,preds)
    assert all(digest(Path(m['predictions']))==m['sha256'] for m in manifest)
    dump(report/'AUDIT.json',dict(passed=True,new_training_completed=4,flashvtg_reused=True,models=15,split_evaluations=30,all_gap_independent_verified=True,source_integrity=True,whole_split_means=True))
    (EXP/'CONCLUSIONS.md').write_text('# GAP 五backbone迁移实测\n\n'+'\n'.join(text[text.index('## GAP 相对 E1 的 Test 变化'):])+'\n')
    dump(EXP/'STATUS.json',dict(status='complete',new_training_completed=4,all_five_backbones_reported=True))

def figures(data,preds):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False});out=EXP/'figures';out.mkdir(exist_ok=True)
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    for split in ['val','test']:
        selected={(r['backbone'],r['method']):r for r in data if r['split']==split and r['threshold']==.4}
        fig,axes=plt.subplots(1,3,figsize=(16,4.8))
        for ax,key,title in zip(axes,['mu_pos','mu_neg','mean_gap'],['Positive mean score ↑','Negative mean score ↓','Mean gap = positive − negative ↑']):
            for j,method in enumerate(['E0','E1','GAP']):
                xx=np.arange(5)+(j-1)*.24;vv=[selected[bb,method][key] for bb in BACKBONES];ax.bar(xx,vv,width=.23,color=COLORS[method],label=method)
                for x,v in zip(xx,vv):ax.text(x,v+.012,f'{v:.3f}',ha='center',rotation=90,fontsize=7)
            ax.set(xticks=range(5),xticklabels=[NAMES[b] for b in BACKBONES],ylim=(0,1),ylabel='Probability / difference',title=title);ax.tick_params(axis='x',rotation=20);ax.grid(axis='y',alpha=.15);ax.legend(ncol=3,fontsize=8)
        fig.suptitle(f'{split.title()}: GMR vs GEQ vs GEQ + GAP (margin 0.6, weight 2; 4 epochs)',fontsize=12);fig.tight_layout();save(fig,split+'_mean_gap_comparison')
        fig,axes=plt.subplots(5,2,figsize=(12.2,14),sharex=True)
        for row,bb in enumerate(BACKBONES):
            for col in [0,1]:
                ax=axes[row,col]
                for method in ['E0','E1','GAP']:
                    pp,gt=preds[bb,method,split];y=np.array([bool(g['relevant_windows']) for g in gt]);ss=np.array([p['pred_exist_score'] for p in pp]);mask=y if col==0 else ~y;vals=ss[mask];den,edges=np.histogram(vals,bins=np.linspace(0,1,41),density=True)
                    ax.stairs(den,edges,color=COLORS[method],ls=STYLES[method],lw=1.6,label=f'{method}: mean {vals.mean():.3f}');ax.axvline(np.median(vals),color=COLORS[method],ls=STYLES[method],alpha=.5,lw=.8)
                ax.axvline(.4,color='#334155',ls=':',lw=.9);ax.set(xlim=(0,1),ylabel='Density',title=NAMES[bb]+' — '+('Positive' if col==0 else 'Negative')+f' (n={mask.sum()})');ax.legend(fontsize=8,frameon=False);ax.grid(axis='y',alpha=.12)
                if row==4:ax.set_xlabel('Existence score')
        fig.suptitle(f'{split.title()} existence-score distributions: E0 / E1 / GAP',fontsize=14);fig.tight_layout(rect=(0,0,1,.975));save(fig,split+'_existence_score_distributions')
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        for ax,key in zip(axes,['AUROC','mAP','mR+@5']):
            for j,method in enumerate(['E0','E1','GAP']):
                xx=np.arange(5)+(j-1)*.24;vv=[selected[bb,method][key] for bb in BACKBONES];ax.bar(xx,vv,width=.23,color=COLORS[method],label=method)
                for x,v in zip(xx,vv):ax.text(x,v+1,f'{v:.2f}',ha='center',rotation=90,fontsize=7)
            ax.set(xticks=range(5),xticklabels=[NAMES[b] for b in BACKBONES],ylim=(0,100),ylabel='Percent',title=key);ax.tick_params(axis='x',rotation=20);ax.legend(ncol=3,fontsize=8);ax.grid(axis='y',alpha=.15)
        fig.tight_layout();save(fig,split+'_performance_comparison')
if __name__=='__main__':main()
