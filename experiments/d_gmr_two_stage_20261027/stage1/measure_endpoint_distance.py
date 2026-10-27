#!/usr/bin/env python3
"""Class-balanced endpoint-distance evaluation of immutable saved predictions."""
import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from common import ROOT,OLD,BACKBONES,rows,dump,sha

OUT=ROOT/'reports/endpoint_distance'
NAMES=dict(moment_detr='Moment-DETR',qd_detr='QD-DETR',cg_detr='CG-DETR',eatr='EaTR',flashvtg='FlashVTG')

def measure(pred,gt):
    ids=[p['qid'] for p in pred];assert len(ids)==len(set(ids))==len(gt)
    mapping={p['qid']:p for p in pred};assert set(mapping)=={g['qid'] for g in gt}
    y=np.array([bool(g['relevant_windows']) for g in gt]);s=np.array([mapping[g['qid']]['pred_exist_score'] for g in gt],dtype=float)
    assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all() and y.any() and (~y).any()
    positive=float(s[y].mean());negative=float(s[~y].mean());dp=1-positive;dn=negative;d=.5*(dp+dn);score=100*(1-d)
    # Independently sum query-level errors with class-balanced weights.
    weights=np.where(y,.5/y.sum(),.5/(~y).sum());error=np.where(y,1-s,s)
    assert abs(float(weights@error)-d)<1e-12 and abs(score-50*(1+positive-negative))<1e-12
    hist_d=0
    # Density integration includes bin width; within-bin means make this exact.
    for mask in [y,~y]:
        counts,edges=np.histogram(s[mask],bins=np.linspace(0,1,41));idx=np.minimum(np.searchsorted(edges,s[mask],side='right')-1,39)
        for j,n in enumerate(counts):
            if n:hist_d+=.5*(n/mask.sum())*float(error[mask][idx==j].mean())
    assert abs(hist_d-d)<1e-12
    return dict(n=len(y),n_positive=int(y.sum()),n_negative=int((~y).sum()),positive_mean=positive,negative_mean=negative,D_positive=dp,D_negative=dn,D_balanced=d,mean_gap=positive-negative,S_percent=score,AUROC=100*roc_auc_score(y,s),AUPRC=100*average_precision_score(y,s)),[(g['qid'],int(yy),float(ss),float(ee),float(ww)) for g,yy,ss,ee,ww in zip(gt,y,s,error,weights)]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    entries=[]
    for b in BACKBONES:
        entries.append((b,'E0','original',0,OLD/f'results/{b}/gmr'))
        for m,ep in [('E1',4),('E2',4),('E3',26)]:entries.append((b,m,'fixed_budget',ep,ROOT/f'runs/{b}/{m}'))
    entries += [('flashvtg','E3_70','additional_snapshot',70,ROOT/'temp_eval/E3_epoch_070'),('flashvtg','E3_76','fixed_extension',76,ROOT/'runs/flashvtg/E3_extra50'),('flashvtg','E4_ValBest42','val_loss_selected_snapshot',42,ROOT/'reports/FlashVTG_E4_valbest_snapshot'),('flashvtg','E4_76','fixed_extension',76,ROOT/'runs/flashvtg/E4_scratch_flashmmr')]
    for ep in [6,8]:entries.append(('flashvtg',f'E2_{ep}','fixed_extension',ep,ROOT/f'runs/flashvtg/E2_long_6_8/epoch_{ep:03d}'))
    result=[];manifest=[];skipped=[]
    with (OUT/'per_query_errors.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['backbone','method','split','qid','exist_label','exist_score','endpoint_error','class_balanced_weight'])
        for b,method,group,ep,path in entries:
            for split in ['val','test']:
                p=path/f'{split}_predictions.jsonl';gtpath=ROOT/f'data/{split}.jsonl'
                if not p.exists():skipped.append(dict(backbone=b,method=method,split=split,reason='saved predictions unavailable'));continue
                met,queries=measure(rows(p),rows(gtpath));result.append(dict(backbone=b,method=method,protocol=group,epoch=ep,split=split,**met))
                writer.writerows((b,method,split,*q) for q in queries)
                manifest.append(dict(backbone=b,method=method,split=split,predictions=str(p),sha256=sha(p),ground_truth=str(gtpath),ground_truth_sha256=sha(gtpath)))
    for record in result:
        base=next(r for r in result if r['backbone']==record['backbone'] and r['split']==record['split'] and r['method']=='E0')
        record['delta_S_vs_E0_pp']=record['S_percent']-base['S_percent'];record['delta_D_vs_E0']=record['D_balanced']-base['D_balanced'];record['delta_AUROC_vs_E0_pp']=record['AUROC']-base['AUROC']
    with (OUT/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
    dump(OUT/'MANIFEST.json',dict(definition='D=0.5*(mean_negative_score+mean_positive(1-score)); S=100*(1-D)',threshold_free=True,class_balanced=True,test_selection=False,files=manifest,skipped=skipped))
    plot(result)
    report(result)
    # Recheck every source after plotting; existing experiment evidence stays intact.
    assert all(sha(x['predictions'])==x['sha256'] and sha(x['ground_truth'])==x['ground_truth_sha256'] for x in manifest)
    dump(OUT/'AUDIT.json',dict(passed=True,evaluated=len(result),source_integrity=True,exact_qid_alignment=True,finite_probability_scores=True,class_balanced_query_sum_equals_formula=True,density_integral_equals_formula=True,skipped=skipped))
    print((OUT/'RESULTS.md').read_text())

def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#666666','#0072B2','#D55E00','#009E73']
    fig,ax=plt.subplots(figsize=(11,5))
    for j,m in enumerate(['E0','E1','E2','E3']):
        vals=[next(r for r in result if r['backbone']==b and r['method']==m and r['split']=='test')['S_percent'] for b in BACKBONES]
        xx=np.arange(5)+(j-1.5)*.19;ax.bar(xx,vals,width=.18,label=f'{m}'+(' (4 ep)' if m in ['E1','E2'] else ' (26 ep)' if m=='E3' else ''),color=colors[j])
        for x,v in zip(xx,vals):ax.text(x,v+.7,f'{v:.1f}',ha='center',fontsize=8)
    ax.set(xticks=range(5),xticklabels=[NAMES[b] for b in BACKBONES],ylim=(0,100),ylabel='Balanced endpoint score S (%)',title='Test: class-balanced endpoint score (higher is better)');ax.axhline(50,color='gray',ls=':',lw=1);ax.legend(ncol=4);ax.grid(axis='y',alpha=.15);fig.tight_layout();save(fig,'all_backbones_test')
    flash=[r for r in result if r['backbone']=='flashvtg' and r['split']=='test'];fig,axes=plt.subplots(1,3,figsize=(16,5))
    labels=[('E0\npretrained' if r['method']=='E0' else ('E4 Val-best' if r['method']=='E4_ValBest42' else r['method'].split('_')[0])+'\n'+str(r['epoch'])+' ep') for r in flash];cs=['#666666' if r['method']=='E0' else '#0072B2' if r['method']=='E1' else '#CC79A7' if r['method'].startswith('E4') else '#009E73' if r['method'].startswith('E3') else '#D55E00' for r in flash]
    for ax,key,title,limit in zip(axes,['S_percent','D_positive','D_negative'],['Balanced endpoint score S ↑','Positive error: mean(1 − score) ↓','Negative error: mean(score) ↓'],[(0,100),(0,1),(0,1)]):
        vv=[r[key] for r in flash];ax.bar(range(len(flash)),vv,color=cs)
        for i,v in enumerate(vv):ax.text(i,v+(1 if key=='S_percent' else .01),f'{v:.2f}',ha='center',fontsize=8)
        ax.set(xticks=range(len(flash)),xticklabels=labels,ylim=limit,title=title,ylabel='Percent' if key=='S_percent' else 'Error');ax.tick_params(axis='x',labelsize=7,rotation=40);ax.grid(axis='y',alpha=.15)
    fig.suptitle('FlashVTG Test: E0 reference and saved checkpoints; budgets and selection rules differ',fontsize=12);fig.tight_layout();save(fig,'flashvtg_test_endpoint_comparison')

def save(fig,name):
    import matplotlib.pyplot as plt
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def report(result):
    text=['# 平衡端点距离评估：与原始 GMR（E0）比较','','这是复用保存预测的评估实验，不新增训练、不用 Test 选模型。每个 split 中正负类别各占 1/2 权重；原始分数直接计算，无分箱近似。','','`D+ = mean_positive(1-score)`；`D- = mean_negative(score)`；`D = (D+ + D-)/2`；`S = 100*(1-D)`。D 越低越好，S 越高越好。S 等价于 50*(1+正类平均分−负类平均分)，不完整描述分布重叠，也不替代 AUROC/AUPRC。恒定分数为50，完全正确为100、完全反向为0。不将该公式声称为全新指标。','','## 五 backbone 固定预算 Test','','E1/E2 微调4轮，E3 随机初始化26轮，E0复用原预训练模型。','','| Backbone | Method | D+ ↓ | D− ↓ | D ↓ | S ↑ | ΔS vs E0 | AUROC |','|---|---|---:|---:|---:|---:|---:|---:|']
    def line(r,backbone=False):
        return '| '+(''+NAMES[r['backbone']]+' | ' if backbone else '')+r['method']+' | '+' | '.join(f"{r[k]:.4f}" for k in ['D_positive','D_negative','D_balanced'])+f" | {r['S_percent']:.2f} | {r['delta_S_vs_E0_pp']:+.2f} | {r['AUROC']:.2f} |"
    text += [line(r,True) for r in result if r['split']=='test' and r['method'] in ['E0','E1','E2','E3']]
    text+=['','## FlashVTG 扩展 Test','','E3_70是既有过程checkpoint，不按其Test选最终模型；E4_ValBest42按前次快照第1–67轮最低Val总损失选择；E4_76为预定终点。E2_6/8为已有续训记录。E1与E3/E4的初始化、native presence监督和训练预算不同，不能据此作纯PV因果归因。','','| Method | D+ ↓ | D− ↓ | D ↓ | S ↑ | ΔS vs E0 | AUROC |','|---|---:|---:|---:|---:|---:|---:|---:|']
    text += [line(r) for r in result if r['split']=='test' and r['backbone']=='flashvtg']
    text+=['','Val/Test完整指标及AUPRC见 metrics.csv；逐查询分数、误差和类别权重见 per_query_errors.csv；来源hash与缺失预测见 MANIFEST.json；复核见 AUDIT.json。','', '图：all_backbones_test.pdf 与 flashvtg_test_endpoint_comparison.pdf；同时输出SVG/PNG。','', '解读：降低负类误差如果伴随正类误差增加，需要同时报告两个分量。较高S不保证更高AUROC或更准确的概率校准。此处不报告统计显著性或跨种子稳定性。']
    (OUT/'RESULTS.md').write_text('\n'.join(text)+'\n')

if __name__=='__main__':main()
