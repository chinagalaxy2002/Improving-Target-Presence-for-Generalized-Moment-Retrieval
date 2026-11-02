#!/usr/bin/env python3
"""Publish the SoccerGMR-only Sim-DETR extension into the existing E1 reports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

KIT = Path(__file__).resolve().parent
REPO = KIT.parents[1]
sys.path.insert(0,str(KIT.parent/'geq_protocol'))
import plotting
import matplotlib.pyplot as plt


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(p,d):
    p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def replace_section(path,marker,body,before=None):
    text = path.read_text()
    begin,end = '<!-- '+marker+'_BEGIN -->','<!-- '+marker+'_END -->'
    block = begin+'\n'+body.rstrip()+'\n'+end
    if begin in text:
        prefix,tail = text.split(begin,1)
        _,suffix = tail.split(end,1)
        text = prefix+block+suffix
    elif before and before in text:
        text = text.replace(before,block+'\n\n'+before,1)
    else:
        text = text.rstrip()+'\n\n'+block+'\n'
    path.write_text(text)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=Path('/home/guoxiangyu/simdetr_gmr_multiroute_geq'))
    args=parser.parse_args()
    out=REPO/'results/geq_epoch_study/sim_detr'
    snapshot=read(out/'full_frozen_comparison_snapshot.json')
    baseline=snapshot['data']['series']['baseline']
    records=[]
    histories={}
    for mode in ('baseline','full','frozen'):
        d=snapshot['data']['series'][mode]
        marker=read(out/'run_metadata'/mode/'COMPLETE.json')
        metrics=d['metrics'];points=[d['grid'][i] for i in (40,50,60)]
        mean=sum(x['gmiou_1'] for x in points)/3
        base_mean=sum(baseline['grid'][i]['gmiou_1'] for i in (40,50,60))/3
        history=[]
        if mode!='baseline':
            p=args.source/f'runs/geq_baseline_{mode}/history.jsonl'
            history=[json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            assert max(h['selection']['mean'] for h in history)==next(h['selection']['mean'] for h in history if h['epoch']==marker['selected_epoch'])
            (out/'run_metadata'/mode/'history.jsonl').write_bytes(p.read_bytes())
            histories[mode]=history
        record=dict(backbone='sim_detr',dataset='SoccerGMR',split='Standard Test',method=mode,
            selected_epoch_raw=marker['selected_epoch'],mAP_percent=metrics['mAP'],
            delta_mAP_vs_GMR_pp=round(metrics['mAP']-baseline['metrics']['mAP'],8),AUROC_percent=metrics['AUROC'],
            G1_04=points[0]['gmiou_1'],G1_05=points[1]['gmiou_1'],G1_06=points[2]['gmiou_1'],G1_mean=mean,
            delta_G1_mean_vs_GMR=mean-base_mean,G3_mean=sum(p['gmiou_3'] for p in points)/3,
            checkpoint_sha256=marker['checkpoint_sha256'])
        records.append(record)
    with (out/'standard_test_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,list(records[0]),lineterminator='\n');writer.writeheader();writer.writerows(records)
    write_json(out/'standard_test_summary.json',records)
    plotting.configure_style()
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    fig.suptitle('Sim-DETR | SoccerGMR Standard Val (465 queries)',fontsize=12)
    for mode,color,marker in [('full','#0072B2','o'),('frozen','#D55E00','s')]:
        history=histories[mode]
        epochs=[h['epoch'] for h in history]
        means=[h['selection']['mean'] for h in history]
        maps=[h['val_mAP'] for h in history]
        pick=next(r['selected_epoch_raw'] for r in records if r['method']==mode)
        axes[0].plot(epochs,means,color=color,marker=marker,markersize=4,label=mode+' (picked epoch '+str(pick)+')')
        axes[1].plot(epochs,maps,color=color,marker=marker,markersize=4,label=mode)
        axes[0].scatter([pick],[means[epochs.index(pick)]],s=110,color=color,marker='*',zorder=5)
    axes[0].set(title='Val selects both GEQ bests at epoch 11',xlabel='Fine-tuning epoch',ylabel='Val mean G-mIoU@1 (0-1)',ylim=(0,1))
    axes[1].set(title='Localization at each Val epoch',xlabel='Fine-tuning epoch',ylabel='Val mAP (%)')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    plotting.save(fig,out/'validation_selection')
    write_json(out/'validation_selection_snapshot.json',dict(dataset='SoccerGMR Standard Val',history=histories,
        selection='mean G-mIoU@1 at 0.4/0.5/0.6; stars indicate the selected checkpoint',
        note='No baseline G1 line: the original training runner hardcoded it, so it is not asserted as measured evidence here'))
    table=['| 方法 | mAP (%) | ΔmAP vs GMR (pp) | AUROC (%) | G1@0.4 | G1@0.5 | G1@0.6 | G1均值 | ΔG1均值 vs GMR |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    labels={'baseline':'GMR baseline','full':'GMR + GEQ Full','frozen':'GMR + GEQ Frozen'}
    for r in records:
        table.append(f"| {labels[r['method']]} | {r['mAP_percent']:.2f} | {r['delta_mAP_vs_GMR_pp']:+.2f} | {r['AUROC_percent']:.2f} | {r['G1_04']:.4f} | {r['G1_05']:.4f} | {r['G1_06']:.4f} | {r['G1_mean']:.4f} | {r['delta_G1_mean_vs_GMR']:+.4f} |")
    details='\n'.join(table)
    text='''# Sim-DETR：SoccerGMR Standard GMR + GEQ 扩展

**数据集仅为 SoccerGMR Standard Test：1036条，544正例、492负例；本次没有评测或适配 Charades，也没有新增 Full 数据集结果。**

GEQ Full较相同GMR baseline提高1.53个百分点mAP、0.0697三点G-mIoU@1均值；Frozen保持mAP不变并提高0.0476三点均值。改进依赖存在性阈值，部分阈值上Frozen G-mIoU低于GMR，因此保留完整0–1扫描和负增益。

## 与 main E1 八组对照是否是同一方法

核心方法一致：GEQ依次读取文本及原生encoder视频证据；Full损失为原生GMR loss + GEQ BCE，原模型/GEQ学习率1e-5/3e-5，GEQ seed2023，按Standard Val G1在0.4/0.5/0.6均值选best、连续6轮无提升早停、上限20轮。Frozen冻结原模型、只训练GEQ BCE。

本次已将Sim适配器接入仓库，直接使用共享`GlobalEventQueryHead`；共享头与原Sim头严格加载且合成输入前向逐位相同，E1的`EvidenceRead.forward`与共享实现AST相同，见[METHOD_EQUIVALENCE.json](METHOD_EQUIVALENCE.json)。Sim有两层encoder且HD2MR会重复调用，当前适配保留首次encoder两层证据。初始化来自Sim自己的GMR best，不属于原四骨干main E1检查点；原生loss、batch与后处理随骨干保留。因此这是同一GEQ方法的Sim骨干扩展，不混入原八组的计数或同初始化消融。

## Standard Test 结果

G-mIoU和差值为0–1；mAP/AUROC为百分数，ΔmAP为百分点。所有增减均相对本骨干GMR。τ是存在性阈值，不是定位IoU阈值。

TABLE

## 与既有README相同的图

![SoccerGMR Sim-DETR Standard Test](full_frozen_comparison.png)

[六面板PDF](full_frozen_comparison.pdf) · [SVG](full_frozen_comparison.svg) · [101点CSV及相对GMR差值](full_frozen_comparison_sweep.csv) · [离线绘图快照](full_frozen_comparison_snapshot.json)

![SoccerGMR Sim-DETR G1 G3](full_frozen_comparison_gmiou_1_3.png)

[G1/G3 PDF](full_frozen_comparison_gmiou_1_3.pdf) · [SVG](full_frozen_comparison_gmiou_1_3.svg)

## Val选模

![SoccerGMR Sim-DETR validation selection](validation_selection.png)

两种GEQ在历史Val上的最高三点均值均选中epoch11。Frozen在所选epoch的Val mAP为22.59，Full为22.19。原迁移报告将前者写成21.10，本处已按真实历史纠正。图中没有复用原训练脚本硬编码的baseline Val G1线。

## 推理与来源验收

本次未重训或调参，严格加载已有GMR/Full/Frozen best，重新在完整1036条Standard Test导出预测。三个检查点前后SHA256不变；0.4/0.5/0.6固定工作点的mAP、mR、AUROC、G1/G3与原Sim迁移指标逐项复现一致。Frozen的300个原生张量与GMR完全相同，正式导出复用GMR候选边界、顺序、分数，仅替换存在性分数，mAP/mR完全相同。

[正式预测、官方指标与完成标记](run_metadata/) · [评测完成记录](EVALUATION_COMPLETE.json) · [审计](AUDIT.json) · [视觉与图表设计检查](FIGURE_REVIEW.md) · [汇总CSV](standard_test_summary.csv)

所有方法保留原Sim Standard输入/原生后处理，未根据Test表现修正长视频处理或选择阈值。本次结果只验证这个seed、这些Val选定模型上的方法扩展，不证明GEQ结构对定位增益的单因素因果，也不包含等预算无GEQ续训控制。

## 已接入代码与复现

[Sim适配器与原生源码](../../../backbones/sim_detr/README.md) · [共享GEQ头](../../../shared/geq_protocol/geq_model.py) · [评测入口](../../../shared/geq_epoch_study/sim_detr_extension.py) · [报告入口](../../../shared/geq_epoch_study/sim_detr_report.py) · [共享绘图器](../../../shared/geq_protocol/plotting.py)

在仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 /home/guoxiangyu/miniconda3/envs/sim_detr/bin/python \\
  shared/geq_epoch_study/sim_detr_extension.py --action evaluate
python shared/geq_epoch_study/sim_detr_extension.py --action plot
python shared/geq_epoch_study/sim_detr_report.py
```

推理与绘图分开使用已有环境：Sim环境提供Torch2.5.1/CUDA12.4，CPU绘图环境提供NumPy/Matplotlib。合法完成结果按哈希复用；离线快照重绘无需原环境、检查点或GPU。权重、视频、特征未包含在发布文件中。
'''.replace('TABLE',details)
    (out/'README.md').write_text(text)
    root_block='''## Sim-DETR：SoccerGMR Standard 的 GEQ 方法扩展

**本节是 SoccerGMR Standard Test（1036条），不是 Charades。** GEQ核心实现与上面的main E1对照一致，已复用共享GEQ头及六面板绘图器；Sim使用自身原生GMR初始化，作为骨干扩展单列，不改变原四骨干八组计数。本次只评测已有GMR/GEQ Full/Frozen三种模型。

| Backbone | GMR mAP (%) | GEQ Full mAP (%) | ΔFull vs GMR (pp) | GEQ Frozen mAP (%) | GMR G1均值 | Full G1均值 | Frozen G1均值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sim-DETR | 20.73 | 22.26 | **+1.53** | 20.73 | 0.4735 | 0.5433 | 0.5212 |

![Sim-DETR SoccerGMR Standard GEQ comparison](results/geq_epoch_study/sim_detr/full_frozen_comparison.png)

[完整Standard指标与来源审计](results/geq_epoch_study/sim_detr/README.md) · [PDF](results/geq_epoch_study/sim_detr/full_frozen_comparison.pdf) · [SVG](results/geq_epoch_study/sim_detr/full_frozen_comparison.svg) · [101点及相对GMR CSV](results/geq_epoch_study/sim_detr/full_frozen_comparison_sweep.csv) · [已接入代码](backbones/sim_detr/README.md)
'''
    replace_section(REPO/'README.md','SIMDETR_SOCCERGMR_GEQ',root_block,before='## 历史结果')
    replace_section(REPO/'results/geq_epoch_study/README.md','SIMDETR_SOCCERGMR_GEQ',
        '## Sim-DETR：同类方法扩展，SoccerGMR Standard\n\n原四骨干八组结果保持不变；新增Sim自己的GMR起点及GEQ Full/Frozen，单列扩展。\n\n'+details+
        '\n\n![Sim-DETR SoccerGMR Standard](sim_detr/full_frozen_comparison.png)\n\n[完整指标、代码接入与来源](sim_detr/README.md)')
    print('SOCCERGMR_SIMDETR_REPORT_READY',out)


if __name__=='__main__':
    main()
