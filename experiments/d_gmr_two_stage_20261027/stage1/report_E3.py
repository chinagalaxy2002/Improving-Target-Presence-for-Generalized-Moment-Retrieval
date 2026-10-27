#!/usr/bin/env python3
"""Combine E0/E1/E2/E3 saved metrics; audit scratch training and draw diagnostics."""
import csv,json,hashlib,time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, BACKBONES, OLD, dump, rows, sha
from independent_metrics import verify
NAMES=dict(moment_detr='Moment-DETR',qd_detr='QD-DETR',cg_detr='CG-DETR',eatr='EaTR',flashvtg='FlashVTG')
EPOCHS=json.loads((ROOT/'config_E3.json').read_text())['epochs']
PRIMARY=['AUROC','AUPRC','Rej-F1','Pos-Recall','mAP','mR+@5','G-mIoU@1']
FULL=PRIMARY+['N-acc','TNR','mR@1','mR@5','G-mIoU@3','Positive-only Localization Recall@5/IoU0.5']

def flat(r,tau='0.4'):
    b=r[tau]['brief'].copy();b['Rej-F1']=b[f'Rej-F1@{tau}'];return b

def csv_write(path,records):
    fields=list(dict.fromkeys(k for row in records for k in row))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)

def plots(histories,logits):
    dest=ROOT/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(5,3,figsize=(13,13),squeeze=False)
    for i,b in enumerate(BACKBONES):
        for j,key in enumerate(('localization','geq','total')):
            ax=axes[i,j]
            for split,color,style in [('train','#0072B2','-o'),('val','#D55E00','--s')]:
                vals=[h[split+'_losses'][key] for h in histories[b]]
                ax.plot(range(1,EPOCHS+1),vals,style,color=color,lw=1.7,ms=4,label=split.title())
            ax.set_title(f'{NAMES[b]}: {key}');ax.set_ylabel('Loss');ax.set_xticks([1]+list(range(5,EPOCHS+1,5))+([EPOCHS] if EPOCHS%5 else []));ax.set_ylim(bottom=0);ax.grid(alpha=.2)
            if i==4:ax.set_xlabel('Epoch')
            if i==0:ax.legend(frameon=False)
    fig.suptitle(f'E3: random initialization, {EPOCHS} joint-training epochs',fontsize=15,y=.995)
    fig.tight_layout(rect=(0,0,1,.98))
    for ext in ('pdf','png','svg'):fig.savefig(dest/f'E3_train_val_loss_curves.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(5,2,figsize=(12,13.5),squeeze=False)
    for i,b in enumerate(BACKBONES):
        combined=np.array([r['logit'] for split in ('val','test') for group in ('positive','negative') for r in logits[b,split][group]])
        lo,hi=float(combined.min()),float(combined.max());margin=max(.05,(hi-lo)*.03);bins=np.linspace(lo-margin,hi+margin,41)
        for j,split in enumerate(('val','test')):
            ax=axes[i,j]
            for group,color,style in [('positive','#D55E00','-'),('negative','#0072B2','--')]:
                vals=[r['logit'] for r in logits[b,split][group]]
                density,_=np.histogram(vals,bins=bins,density=True)
                ax.stairs(density,bins,lw=1.8,color=color,linestyle=style,label=f'{group.title()} (n={len(vals)})')
                ax.stairs(density,bins,color=color,fill=True,lw=0,alpha=.09)
            ax.axvline(np.log(.4/.6),color='#475569',ls=':',lw=1,label='Score threshold 0.4')
            ax.set_title(f'{NAMES[b]} — {split.title()}');ax.set_ylabel('Density');ax.set_ylim(bottom=0);ax.legend(fontsize=8,frameon=False)
            if i==4:ax.set_xlabel('GEQ existence logit')
    fig.suptitle(f'E3 GEQ logit distributions at fixed epoch {EPOCHS}',fontsize=15,y=.995);fig.tight_layout(rect=(0,0,1,.98))
    for ext in ('pdf','png','svg'):fig.savefig(dest/f'E3_geq_logit_distributions.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)

def main():
    report=ROOT/'reports';audit={};metrics={};histories={};logits={};allrows=[];deltas=[];curve_rows=[];logit_stats=[]
    for b in BACKBONES:
        for method in ('E0','E1','E2','E3'):
            directory=ROOT/f'reused_baselines/{b}' if method=='E0' else ROOT/f'runs/{b}/{method}'
            if method=='E3':
                assert json.loads((directory/'COMPLETE.json').read_text())['epoch']==EPOCHS
                cfg=json.loads((directory/'CONFIG.json').read_text());pre=json.loads((ROOT/f'preflights/{b}/E3/CONFIG.json').read_text())
                assert cfg['initial_tensor_sha256']==pre['initial_tensor_sha256']
                assert cfg['initialization']['checkpoint'] is None and cfg['initialization']['pretrained_restore_attempts']==[]
                h=rows(directory/'history.jsonl');assert [x['epoch'] for x in h]==list(range(1,EPOCHS+1));histories[b]=h
                for record in h:
                    for split in ('train','val'):
                        curve_rows.append(dict(Backbone=NAMES[b],Epoch=record['epoch'],Split=split,**record[split+'_losses']))
                meta=json.loads((directory/'CHECKPOINT.json').read_text());assert sha(directory/f'epoch_{EPOCHS:03d}.ckpt')==meta['sha256']
                ck=torch.load(directory/f'epoch_{EPOCHS:03d}.ckpt',map_location='cpu',weights_only=False)
                assert ck['epoch']==EPOCHS and [g['lr'] for g in ck['optimizer']['param_groups']]==[1e-5,3e-5]
                assert not any(n.startswith('base.exist_head.') or n.startswith('pv.') for n in ck['model'])
                initial=cfg['initial_tensor_sha256'];changes=[]
                for n,t in ck['model'].items():
                    if hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()!=initial[n]:changes.append(n)
                assert any(n.startswith('base.transformer.') for n in changes) and any(n.startswith('geq.') for n in changes)
                e1initial=json.loads((ROOT/f'runs/{b}/E1/CONFIG.json').read_text())['shared_initial_tensor_sha256']
                shared_matrices=[n for n,t in ck['model'].items() if n.startswith('base.') and n in e1initial and t.ndim>=2]
                diff=[n for n in shared_matrices if initial[n]!=e1initial[n]];assert len(diff)>0
                audit[b]=dict(passed=True,random_init_without_restore=True,same_seed_reconstruction_equal=True,initial_hash_matches_preflight=True,checkpoint_sha256=meta['sha256'],updated_parameter_names=changes,shared_matrices_different_from_e1_pretrained_initialization=len(diff),shared_matrices_total=len(shared_matrices),video_presence_supervision='GEQ BCE only',native_head_parameters=0,distillation=False,strict_pure_initialization_comparison=False)
                del ck
            for split in ('val','test'):
                r=json.loads((directory/f'{split}_metrics.json').read_text());metrics[b,method,split]=r
                if method=='E3':
                    audit[b][split+'_independent_metrics']=verify(rows(directory/f'{split}_predictions.jsonl'),rows(ROOT/f'data/{split}.jsonl'),r)
                    raw=json.loads((directory/f'{split}_geq_logits.json').read_text());logits[b,split]=raw
                    for group in ('positive','negative'):
                        vals=np.array([x['logit'] for x in raw[group]]);scores=np.array([x['score'] for x in raw[group]])
                        logit_stats.append(dict(Backbone=NAMES[b],Split=split,Group=group,N=len(vals),MeanLogit=float(vals.mean()),StdLogit=float(vals.std()),MinLogit=float(vals.min()),MaxLogit=float(vals.max()),MeanScore=float(scores.mean()),StdScore=float(scores.std())))
                for tau in ('0.4','0.5','0.6'):
                    f=flat(r,tau);allrows.append(dict(Backbone=NAMES[b],Method=method,Split=split,Threshold=tau,**{k:f[k] for k in FULL}))
        for split in ('val','test'):
            for tau in ('0.4','0.5','0.6'):
                for other in ('E0','E1','E2'):
                    a,z=flat(metrics[b,'E3',split],tau),flat(metrics[b,other,split],tau)
                    deltas.append(dict(Backbone=NAMES[b],Comparison='E3-'+other,Split=split,Threshold=tau,**{k:a[k]-z[k] for k in FULL}))
    before=json.loads((report/'E3_PRESERVATION_BEFORE.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in before.items()),'An existing E0/E1/E2 artifact changed'
    dump(report/'E3_FINAL_AUDIT.json',dict(passed=True,formal_e3_runs=5,seed=2026,epochs=EPOCHS,prior_artifacts_unchanged=len(before),finished=time.time(),runs=audit))
    csv_write(report/'all_metrics_with_E3.csv',allrows);csv_write(report/'E3_comparisons_pp.csv',deltas);csv_write(report/'E3_loss_curves.csv',curve_rows);csv_write(report/'E3_logit_statistics.csv',logit_stats)
    table=['| Backbone | Method | '+' | '.join(PRIMARY)+' |','|'+'---|'*(len(PRIMARY)+2)]
    for b in BACKBONES:
        for method in ('E0','E1','E2','E3'):
            f=flat(metrics[b,method,'test']);table.append('| '+NAMES[b]+' | '+method+' | '+' | '.join(f'{f[k]:.2f}' for k in PRIMARY)+' |')
    (report/'RESULTS_WITH_E3.md').write_text('# E0/E1/E2/E3：固定预算比较\n\nStandard Test，τ=0.4；单位 %。E0=原始预训练 GMR；E1=预训练 GMR+端到端 GEQ；E2=预训练 GMR+GEQ+PV；E3=随机初始化 GMR+GEQ。E1/E2 固定 4 个微调 epoch；E3 按用户追加要求固定 26 个从零训练 epoch；seed=2026。\n\n'+'\n'.join(table)+'\n\n**E1/E3 不是纯初始化对照：E3 只使用 GEQ 视频级 BCE，E1 还监督 native presence BCE；CG 的 E3 另关闭原生内部蒸馏。** E3 延用 backbone lr=1e-5、GEQ lr=3e-5，两者训练预算也不同；不能用当前预算断言从零训练的最终上限。完整 Val/Test 与 0.4/0.5/0.6 见 [CSV](all_metrics_with_E3.csv)。\n\n[结论](../CONCLUSIONS_E3.md) · [E3 审计](E3_FINAL_AUDIT.json)\n')
    conclusions=['# E3 随机初始化 GEQ：结果与边界\n','新增五次训练已全部完成，全部使用原生随机工厂初始化，没有加载任何已训练 GMR/GEQ 权重。GEQ 是唯一训练及导出的存在性头；原生候选分类、回归、Hungarian/point 匹配及必要的辅助定位监督继续启用。\n','## 可比性\n','E1 与 E3 的存在性监督路径不完全一致：E1 保留 native presence BCE + GEQ BCE；E3 按请求只有 GEQ BCE，原 native head 被替换为无参数 API 占位。CG E3 同时禁用原生 loss_distill；其候选/GT 匹配、saliency、moment-sentence alignment 与 dummy orthogonality 原生项保留，没有增加新 saliency 或 teacher。因此结果不能把所有差异归因于初始化。\n','E3 按用户追加要求固定 26 epochs（E1/E2 为 4 epochs），backbone lr=1e-5、GEQ lr=3e-5、AdamW/1e-4、clip=0.1，与 E1 的优化协议保持一致。此低学习率来自先前微调协议，未针对 scratch 调整；随机模型可能严重欠训练，不做超参数搜索。验证损失每轮记录但不选 epoch：使用原生训练输出分支构造定位 loss，同时保持所有子模块（含 attention/dropout/BN）eval，以计算确定性的无更新损失，并恢复 RNG；它与训练时随机 dropout 下的损失定义不同，不应仅按二者差值判断过拟合。\n','## 六项研究问题\n']
    for b in BACKBONES:
        a=flat(metrics[b,'E3','test']);e=flat(metrics[b,'E1','test']);v=flat(metrics[b,'E3','val']);h=histories[b]
        conclusions += [f'### {NAMES[b]}\n',f"1. 正负区分：Test AUROC {a['AUROC']:.2f}%，AUPRC {a['AUPRC']:.2f}%（正例比例基线为 {544/1036*100:.2f}%）；Val AUROC {v['AUROC']:.2f}%。这些是当前预算下的观测，不代表显著性检验。\n",f"2. 与 E1 排序比较：AUROC {a['AUROC']-e['AUROC']:+.2f} pp，AUPRC {a['AUPRC']-e['AUPRC']:+.2f} pp。\n",f"3. 固定 τ=0.4：正例存在性 TPR {a['Pos-Recall']:.2f}%（E3−E1 {a['Pos-Recall']-e['Pos-Recall']:+.2f} pp），负例 TNR {a['TNR']:.2f}%。高 TPR 必须结合 TNR 看，不能把全接受当成良好判别。\n",f"4. 定位：Test mAP {a['mAP']:.2f}%，mR@1 {a['mR@1']:.2f}%，mR@5 {a['mR@5']:.2f}%，mR+@5 {a['mR+@5']:.2f}%；E3−E1 mAP {a['mAP']-e['mAP']:+.2f} pp，mR+@5 {a['mR+@5']-e['mR+@5']:+.2f} pp。这些数值按无存在性 hard gate 的官方定位接口计算，非零指标本身不等于充分收敛。\n",f"5. 预训练作用：上述不同预算及监督路径下的差异可以描述当前训练方案的实践表现，但原生存在性监督不同（CG 另有蒸馏差异），不足以孤立证明纯初始化的因果效应。\n",f"6. 收敛诊断：训练定位 loss 从 {h[0]['train_losses']['localization']:.4f} 到 {h[-1]['train_losses']['localization']:.4f}，GEQ BCE 从 {h[0]['train_losses']['geq']:.4f} 到 {h[-1]['train_losses']['geq']:.4f}；Val 定位 loss 从 {h[0]['val_losses']['localization']:.4f} 到 {h[-1]['val_losses']['localization']:.4f}，GEQ BCE 从 {h[0]['val_losses']['geq']:.4f} 到 {h[-1]['val_losses']['geq']:.4f}。训练预算不足、低 backbone 学习率、共享表示未充分形成或多目标梯度竞争都属于可能解释；本实验未隔离这些机制，不能据此认定 GEQ 设计失败。\n"]
    conclusions += ['## 证据文件\n','全部初始参数 SHA256 与实际 factory 配置保存在各 runs/*/E3/CONFIG.json；PREFLIGHT 检查了恢复 API 禁止、同 seed 重建哈希一致、GEQ/定位梯度、全负/正例/多 GT/原生 full batch、新建模型重载以及原生推理/官方指标。E3_FINAL_AUDIT 确认旧 E0/E1/E2 结果文件逐字节不变，并独立复算新增 Val/Test 定位指标。\n','[四方法主表](reports/RESULTS_WITH_E3.md) · [完整 CSV](reports/all_metrics_with_E3.csv) · [训练/验证曲线](figures/E3_train_val_loss_curves.pdf) · [GEQ logit 分布](figures/E3_geq_logit_distributions.pdf) · [审计](reports/E3_FINAL_AUDIT.json)\n','本实验只有 seed=2026、E3 固定 26 轮预算（E1/E2 为 4 轮）。未按 Test 选择模型、epoch、阈值或超参数，不报告统计显著性或跨种子稳定性。\n']
    (ROOT/'CONCLUSIONS_E3.md').write_text('\n'.join(conclusions));plots(histories,logits)
    print('E3_RESULTS_AUDITED_AND_REPORTED',flush=True)

if __name__=='__main__':main()
