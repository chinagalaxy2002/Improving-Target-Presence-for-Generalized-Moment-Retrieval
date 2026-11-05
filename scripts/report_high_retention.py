#!/usr/bin/env python3
"""Verify and render the six-backbone high-event-retention result package.

Consumes only the six models in EVIDENCE.json; no training, model selection,
or Test threshold optimization. All required predictions are packaged.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

NAMES = ('moment_detr','sim_detr','bm_detr','uvcom','taskweave','ld_detr')
LABELS = ('Moment-DETR','Sim-DETR','BM-DETR','UVCOM','TaskWeave','LD-DETR')
TICKS = ('Moment-\nDETR','Sim-\nDETR','BM-\nDETR','UVCOM','Task-\nWeave','LD-\nDETR')
SEARCHES = ('grid_101','val_breakpoints')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_gzip(path):
    with gzip.open(path,'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def ordered(predictions,gt):
    indexed={p['qid']:p for p in predictions}
    assert len(indexed)==len(predictions)==len(gt)
    assert set(indexed)=={g['qid'] for g in gt}
    result=[indexed[g['qid']] for g in gt]
    assert all(p['vid']==g['vid'] and p.get('query',g['query'])==g['query'] for p,g in zip(result,gt))
    return result


def select_threshold(scores,y,search):
    thresholds=np.arange(101)/100 if search=='grid_101' else np.unique(np.r_[0.,scores,1.])
    rates=(scores[y,None]>thresholds[None,:]).mean(0)
    feasible=np.flatnonzero(rates>=.95-1e-12)
    assert len(feasible),'95% positive retention is infeasible on Val'
    index=feasible[-1]
    return float(thresholds[index]),100*float(rates[index])


def compute(root):
    from soccergmr.evaluation.official.metrics import (
        _clean_pred_windows,_compute_set_iou_score,prepare_submission_for_gmiou,compute_G_mIoU,
    )
    evidence=json.loads((root/'EVIDENCE.json').read_text())
    assert set(evidence['models'])==set(NAMES)
    gt={}
    for split,meta in evidence['ground_truth'].items():
        path=root/meta['path'];assert sha(path)==meta['sha256']
        gt[split]=read_gzip(path)
    assert [len(gt[s]) for s in ('val','standard','full')]==[465,1036,2970]
    pairs={s:{(g['vid'],g['query']) for g in rows} for s,rows in gt.items()}
    assert not pairs['val']&pairs['standard'] and not pairs['val']&pairs['full']
    overlap=len(pairs['standard']&pairs['full'])
    video_overlap=len({g['vid'] for g in gt['standard']}&{g['vid'] for g in gt['full']})
    selected={};predictions={};checks=0
    for name in NAMES:
        selected[name]={};predictions[name]={}
        for method in ('gmr','geq_full'):
            info=evidence['models'][name]['methods'][method]
            predictions[name][method]={}
            for split,meta in info['predictions'].items():
                path=root/meta['path'];assert sha(path)==meta['sha256']
                with gzip.open(path,'rb') as stream:
                    assert hashlib.sha256(stream.read()).hexdigest()==meta['uncompressed_sha256']
                assert meta['checkpoint_sha256']==info['checkpoint_sha256']
                predictions[name][method][split]=ordered(read_gzip(path),gt[split])
            val=predictions[name][method]['val']
            y=np.array([bool(g['relevant_windows']) for g in gt['val']])
            scores=np.array([p['pred_exist_score'] for p in val])
            assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
            selected[name][method]={s:select_threshold(scores,y,s) for s in SEARCHES}
    # Threshold calculations above use only Val scores/labels.
    rows=[]
    for search in SEARCHES:
        for split in ('standard','full'):
            y=np.array([bool(g['relevant_windows']) for g in gt[split]])
            for name in NAMES:
                row=dict(search=search,split=split,backbone=name)
                for method in ('gmr','geq_full'):
                    pred=predictions[name][method][split]
                    threshold,val_tpr=selected[name][method][search]
                    score=np.array([p['pred_exist_score'] for p in pred]);accept=score>threshold
                    raw=np.array([_compute_set_iou_score(
                        [w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'],max_pred_windows=10)[:1]],g['relevant_windows'])
                        for p,g in zip(pred,gt[split])])
                    metric=100*float(np.where(accept,raw,(~y).astype(float)).mean())
                    gated,_=prepare_submission_for_gmiou(pred,cls_threshold=threshold)
                    official=compute_G_mIoU(gated,gt[split],k_list=(1,))['G-mIoU@1']
                    assert abs(round(metric,2)-official)<1e-9
                    checks+=1
                    row.update({method+'_threshold':threshold,method+'_val_TPR':val_tpr,
                                method+'_G1':metric,method+'_test_TPR':100*float(accept[y].mean()),
                                method+'_test_FPR':100*float(accept[~y].mean())})
                row['delta_pp']=row['geq_full_G1']-row['gmr_G1']
                row['relative_gain_percent']=100*row['delta_pp']/row['gmr_G1']
                assert row['delta_pp']>0,'The documented stable subset no longer matches the evidence'
                rows.append(row)
    with (root/'EXPECTED_METRICS.csv').open() as stream:
        expected=list(csv.DictReader(stream))
    assert len(expected)==len(rows)==24
    for row in rows:
        match=next(r for r in expected if (r['search'],r['split'],r['backbone'])==(row['search'],row['split'],row['backbone']))
        for key,source in [('gmr_G1','gmr_G'),('geq_full_G1','geq_full_G'),('delta_pp','delta_pp'),
                           ('relative_gain_percent','relative_gain_percent'),('gmr_threshold','gmr_threshold'),
                           ('geq_full_threshold','geq_full_threshold'),('gmr_test_TPR','gmr_TPR'),
                           ('geq_full_test_TPR','geq_full_TPR'),('gmr_test_FPR','gmr_FPR'),('geq_full_test_FPR','geq_full_FPR')]:
            assert abs(row[key]-float(match[source]))<1e-10,(row['backbone'],key)
    summary=[]
    for search in SEARCHES:
        for split in ('standard','full'):
            group=[r for r in rows if r['search']==search and r['split']==split]
            a=float(np.mean([r['gmr_G1'] for r in group]));b=float(np.mean([r['geq_full_G1'] for r in group]))
            summary.append(dict(search=search,split=split,mean_GMR=a,mean_GEQ=b,delta_pp=b-a,
                relative_gain_percent=100*(b-a)/a,positive_backbones=sum(r['delta_pp']>0 for r in group),
                mean_GMR_test_TPR=float(np.mean([r['gmr_test_TPR'] for r in group])),
                mean_GEQ_test_TPR=float(np.mean([r['geq_full_test_TPR'] for r in group])),
                mean_GMR_test_FPR=float(np.mean([r['gmr_test_FPR'] for r in group])),
                mean_GEQ_test_FPR=float(np.mean([r['geq_full_test_FPR'] for r in group]))))
    audit=dict(passed=True,backbones=list(NAMES),comparison_rows=len(rows),prediction_files=36,
        official_metric_checks=checks,expected_metric_fields_checked=24*10,
        evidence_sha256=sha(root/'EVIDENCE.json'),script_sha256=sha(__file__),
        model_subset_selected_post_hoc=True,thresholds_selected_on='Standard Val only',
        checkpoint_selection='Existing finalized bests retained',new_training=False,
        significance_tests=False,standard_full_overlap_video_query_pairs=overlap,
        standard_full_overlap_video_ids=video_overlap)
    return rows,summary,audit


def plot(root,rows,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                         'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    for search,stem in [('grid_101','PRIMARY'),('val_breakpoints','EXACT_THRESHOLDS')]:
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.8),sharey=True)
        for ax,split,title in zip(axes,('standard','full'),('Standard Test (1,036)','Full-Decodable Test (2,970)')):
            group=[next(r for r in rows if r['search']==search and r['split']==split and r['backbone']==n) for n in NAMES]
            x=np.arange(6)
            a=np.array([r['gmr_G1'] for r in group]);b=np.array([r['geq_full_G1'] for r in group])
            ax.bar(x-.18,a,width=.34,color='#a6a6a6',edgecolor='#555555',linewidth=.6,label='GMR')
            ax.bar(x+.18,b,width=.34,color='#0072b2',edgecolor='#003f63',linewidth=.6,label='GEQ Full')
            ax.scatter(x-.18,a,marker='s',s=13,color='#444444',zorder=3)
            ax.scatter(x+.18,b,marker='o',s=13,color='#003f63',zorder=3)
            for i,r in enumerate(group):
                ax.text(i,max(a[i],b[i])+.7,f"+{r['delta_pp']:.2f}",ha='center',fontsize=8)
            ax.set_xticks(x,TICKS);ax.set_ylim(0,27);ax.set_yticks([0,5,10,15,20,25])
            ax.set_ylabel('G-mIoU@1 (0–100)');ax.set_title(title)
            ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
            ax.spines[['top','right']].set_visible(False)
            mean=next(r for r in summary if r['search']==search and r['split']==split)
            ax.text(.02,.97,f"Mean gain: +{mean['delta_pp']:.2f} pp (+{mean['relative_gain_percent']:.2f}%)",transform=ax.transAxes,va='top',fontsize=8.5)
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.01))
        fig.text(.5,.062,'Each model: largest Val threshold retaining ≥95% of positive queries.',ha='center',fontsize=8)
        fig.text(.5,.016,'Exploratory six-model subset · annotations: GEQ − GMR (pp) · '+('0.01 threshold grid' if search=='grid_101' else 'all Val score breakpoints'),ha='center',fontsize=8)
        fig.tight_layout(rect=[0,.11,1,.94],w_pad=1.2)
        for ext in ('png','pdf','svg'):
            fig.savefig(root/(stem+'.'+ext),dpi=220)
            if ext=='svg':
                path=root/(stem+'.svg')
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)


def document(root,rows,summary):
    lines=['# 高事件保留率场景：六骨干的 GEQ Full 配对分析','','**在本次分析识别的六骨干子集中，GEQ Full 的 G-mIoU@1 在 Standard 和 Full 上均提高，且在两种阈值搜索方式下方向一致。** 主分析的骨干等权平均得分在 Standard 上从7.264提高至9.771，在Full上从7.189提高至9.353，相对提升34.52%和30.10%。','','本结果包仅报告 Moment-DETR、Sim-DETR、BM-DETR、UVCOM、TaskWeave、LD-DETR，聚焦每次返回一个最佳片段的使用方式。**这是依据已有Test结果及阈值敏感性分析识别的探索性稳定子集，不是事先确定的骨干集合，也不作为所有骨干普适有效的证据。**','','## 场景与阈值规则','','场景是尽量不因存在性门控而漏掉真实事件，允许用户继续复核返回片段。“正查询保留率”定义为真实存在事件的查询中，通过存在性门控的比例；它不等于定位正确率。','','1. 固定已有GMR best及GEQ Full best。GEQ best仍由原训练流程的Val G1(.4/.5/.6)均值选定，没有重新选择epoch。','2. 每个模型分别在Standard Val（465条：255正、210空）上选择满足正查询保留率≥95%的**最大存在性阈值**。这在达到保留率要求的阈值中尽量减少空查询误返回。','3. 主分析搜索0.00–1.00、步长0.01；复核遍历全部Val分数断点及0/1。保留规则均为`p_exist > τ`。','4. 各自阈值原样用于Standard Test和Full Test，指标沿用官方G-mIoU@1；Test不参与阈值优化。两个模型使用相同目标，各自选择适合自身分数尺度的阈值。','','## 主结果：0.01网格','','![主结果](PRIMARY.png)','','图1：六个已识别骨干在两套Test上的G-mIoU@1均提高。灰色左柱为GMR，蓝色右柱为GEQ Full；方形和圆形提供双重编码，柱上标注绝对增益（百分点）。两图使用相同的零起点纵轴，阈值各自在Val上确定。','','[PDF矢量图](PRIMARY.pdf) · [SVG矢量图](PRIMARY.svg)','','| 骨干 | Standard：GMR→GEQ | Δ百分点 | 相对增益 | Full：GMR→GEQ | Δ百分点 | 相对增益 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,label in zip(NAMES,LABELS):
        cells=[]
        for split in ('standard','full'):
            r=next(r for r in rows if r['search']=='grid_101' and r['split']==split and r['backbone']==name)
            cells += [f"{r['gmr_G1']:.3f}→{r['geq_full_G1']:.3f}",f"{r['delta_pp']:+.3f}",f"{r['relative_gain_percent']:+.2f}%"]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['','相对增益=`100 × (GEQ−GMR)/GMR`。下表汇总先对六骨干的原始得分等权平均，再计算相对增益，不是六个相对百分比的平均。原值和绝对百分点同时报告，避免较低baseline分数造成的比例放大被忽略。','','## 精确阈值复核','','![精确阈值复核](EXACT_THRESHOLDS.png)','','图2：不采用0.01步长，按完整Val分数断点选阈值后，六个骨干在两套Test上仍均提高。阈值仍只由Val决定；图1和图2各使用自身固定的阈值结果，不能混用。','','[PDF矢量图](EXACT_THRESHOLDS.pdf) · [SVG矢量图](EXACT_THRESHOLDS.svg)','','| 搜索方式 | 测评集 | GMR平均 | GEQ平均 | Δ百分点 | 相对增益 | 提高骨干数 |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {'0.01网格' if r['search']=='grid_101' else '完整Val分数断点'} | {r['split']} | {r['mean_GMR']:.3f} | {r['mean_GEQ']:.3f} | {r['delta_pp']:+.3f} | {r['relative_gain_percent']:+.2f}% | {r['positive_backbones']}/6 |")
    lines += ['','## 阈值与实际保留／误返回','','95%是Val上的选阈值约束，不保证Test恰好满足同一比例。主分析的Test平均保留率约为Standard GMR99.14%／GEQ97.24%，Full GMR99.48%／GEQ98.59%；平均空查询误返回率分别为97.76%／93.83%、97.82%／94.60%。本场景以保留事件为优先，不能描述为低误返回系统。','','`DECISIONS.csv`逐骨干列出两种搜索的阈值、Val保留率、Test保留率与误返回率，均按真实正负标签分组。','','## 协议与结论范围','','- Moment-DETR、Sim-DETR保留历史六轮早停、75-token协议；BM-DETR、UVCOM、TaskWeave、LD-DETR保留正式十轮早停、128-token协议。只作各骨干内部配对比较，不解释成统一预算的跨骨干排名。','- Standard Test为1036条；Full-Decodable为2970/2986条，已有16条未覆盖。按本包GT快照重算，两套共有185个视频ID、186个相同的视频—文本输入，不能视为独立样本重复验证。','- 使用既有检查点及完整查询预测，没有重新训练。GEQ Full会同时改变定位候选与存在性分数，因此综合增益不能全部归因于GEQ读出结构。','- 本结果比较数值增益，不作统计显著性或多训练种子稳定性声明。','- 候选骨干与主指标在现有结果分析后确定，子集内一致性不替代独立评测或预先确定范围的泛化验证。','','可使用的结论：**在高事件保留率这一特定使用方式下，所分析六骨干均观察到Standard和Full的G-mIoU@1提升，并通过精确阈值搜索复核。**','','## 数据与复现','','[逐骨干指标](METRICS.csv) · [汇总](SUMMARY.csv) · [阈值与错误率](DECISIONS.csv) · [来源与检查点哈希](EVIDENCE.json) · [数值核验](AUDIT.json) · [图表审计](FIGURE_AUDIT.md)','','本目录的`evidence/`仅包含上述六骨干的Val及两套Test压缩预测和对应GT快照。所有预测均绑定检查点哈希；模型权重、训练数据特征不在本结果包中。无需GPU或模型权重即可从这些预测重算阈值和指标。','','```bash','PYTHONPATH=src python scripts/report_high_retention.py','PYTHONPATH=src python scripts/report_high_retention.py --verify-only','```','','重算时逐文件核验压缩与解压预测哈希、qid/vid覆盖、Val/Test查询不重叠和检查点配对。24行指标与既有统计逐字段核验，48个G-mIoU值与官方评测实现逐项匹配。']
    (root/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/high_retention'))
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    rows,summary,audit=compute(args.root)
    if args.verify_only:
        print('VERIFIED',audit['comparison_rows'],'rows;',audit['official_metric_checks'],'official checks')
        return
    write_csv(args.root/'METRICS.csv',rows);write_csv(args.root/'SUMMARY.csv',summary)
    keys=['search','split','backbone','gmr_threshold','geq_full_threshold','gmr_val_TPR','geq_full_val_TPR',
          'gmr_test_TPR','geq_full_test_TPR','gmr_test_FPR','geq_full_test_FPR']
    write_csv(args.root/'DECISIONS.csv',[{k:r[k] for k in keys} for r in rows])
    plot(args.root,rows,summary);document(args.root,rows,summary)
    (args.root/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('REPORTED',args.root)


if __name__=='__main__':
    main()
