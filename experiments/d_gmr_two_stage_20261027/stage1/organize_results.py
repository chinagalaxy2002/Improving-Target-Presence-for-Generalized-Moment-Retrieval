#!/usr/bin/env python3
"""Audit completed fixed-budget experiments and organize evidence without training."""
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from common import ROOT, OLD, BACKBONES, dump, rows, sha
from independent_metrics import verify

NAMES=dict(moment_detr='Moment-DETR',qd_detr='QD-DETR',cg_detr='CG-DETR',eatr='EaTR',flashvtg='FlashVTG')
FIELDS=['AUROC','AUPRC','Rej-F1','Pos-Recall','N-acc','mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3','Positive-only Localization Recall@5/IoU0.5']

def flat(r,tau='0.4'):
    x=r[tau]['brief'].copy();x['Rej-F1']=x[f'Rej-F1@{tau}'];return x

def csv_write(path, records):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

def main():
    report=ROOT/'reports';audit={};metrics={};diags={};frozen={};deltas=[];diagnostics=[];history=[]
    queue=json.loads((ROOT/'QUEUE.json').read_text());assert queue['status']=='complete' and len(queue['jobs'])==10
    assert all(j['status']=='complete' and j['returncode']==0 for j in queue['jobs'])
    assert max(j['finished'] for j in queue['jobs'] if j['setting']=='E1')<=min(j['started'] for j in queue['jobs'] if j['setting']=='E2')
    for b in BACKBONES:
        configs={}
        for method in ('E1','E2'):
            p=ROOT/f'runs/{b}/{method}';cfg=json.loads((p/'CONFIG.json').read_text());configs[method]=cfg
            complete=json.loads((p/'COMPLETE.json').read_text());ckinfo=json.loads((p/'CHECKPOINT.json').read_text())
            assert complete['epoch']==4 and complete['seed']==2026
            h=rows(p/'history.jsonl');assert [x['epoch'] for x in h]==[1,2,3,4]
            ck=p/'epoch_004.ckpt';assert sha(ck)==ckinfo['sha256']
            saved=torch.load(ck,map_location='cpu',weights_only=False);assert saved['epoch']==4 and saved['setting']==method
            assert [g['lr'] for g in saved['optimizer']['param_groups']]==[1e-5,3e-5]
            changed={}
            for n,original in cfg['shared_initial_tensor_sha256'].items():
                t=saved['model'][n];actual=hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()
                if actual!=original:changed[n]=t.numel()
            assert any(n.startswith('base.transformer.') for n in changed) and any(n.startswith('geq.') for n in changed)
            del saved
            verified={}
            for split in ('val','test'):
                gt=rows(ROOT/f'data/{split}.jsonl');pred=rows(p/f'{split}_predictions.jsonl');r=json.loads((p/f'{split}_metrics.json').read_text())
                assert len(pred)==len(gt)==len({x['qid'] for x in pred})
                verified[split]=verify(pred,gt,r)
                metrics[b,method,split]=r;diags[b,method,split]=json.loads((p/f'{split}_diagnostics.json').read_text())
            for record in h:
                history.append(dict(Backbone=NAMES[b],Method=method,Epoch=record['epoch'],Seconds=record['seconds'],**record['mean_losses']))
            audit[f'{b}/{method}']=dict(passed=True,epochs=4,seed=2026,checkpoint_sha256=ckinfo['sha256'],changed_native_tensors=sum(n.startswith('base.') for n in changed),changed_geq_tensors=sum(n.startswith('geq.') for n in changed),split_verification=verified)
        assert configs['E1']['shared_initial_tensor_sha256']==configs['E2']['shared_initial_tensor_sha256']
        assert configs['E1']['initialization']['sha256']==configs['E2']['initialization']['sha256']
        for split in ('val','test'):
            metrics[b,'GMR',split]=json.loads((ROOT/f'reused_baselines/{b}/{split}_metrics.json').read_text())
            frozen[b,split]=json.loads((ROOT/f'reused_frozen_geq/{b}/{split}_metrics.json').read_text())
            for tau in ('0.4','0.5','0.6'):
                for label,a,z in [('E1-GMR',metrics[b,'E1',split],metrics[b,'GMR',split]),('E2-E1',metrics[b,'E2',split],metrics[b,'E1',split]),('E2-GMR',metrics[b,'E2',split],metrics[b,'GMR',split]),('E1-FrozenGEQ',metrics[b,'E1',split],frozen[b,split])]:
                    aa,zz=flat(a,tau),flat(z,tau)
                    deltas.append(dict(Backbone=NAMES[b],Split=split,Threshold=tau,Comparison=label,**{k:aa[k]-zz[k] for k in FIELDS}))
            for method in ('E1','E2'):
                d=diags[b,method,split];q=d['vs_gmr_localization_hits']['5'];pr=d['vs_gmr_presence']['0.4']
                diagnostics.append(dict(Backbone=NAMES[b],Method=method,Split=split,PositiveLocalizationRecall5=d['positive_only_localization']['5']['recall'],LostLocalization5=q['lost_count'],RestoredLocalization5=q['restored_count'],LostPresence=len(pr['lost_qids']),RestoredPresence=len(pr['restored_qids']),OracleQueryRecall8=d['top8_oracle']['query_recall_iou05'],OracleGTCoverage8=d['top8_oracle']['mean_gt_coverage_iou05'],QualityPearson=d['pv_quality_pearson'],RawMAP=d['raw_localization']['mAP'],FinalMAP=flat(metrics[b,method,split])['mAP'],PositiveMeanScore=d['score_distribution']['positive']['mean'],NegativeMeanScore=d['score_distribution']['negative']['mean']))
    csv_write(report/'comparisons_pp.csv',deltas);csv_write(report/'diagnostics_summary.csv',diagnostics)
    # Original histories have different native loss keys; preserve their union.
    fields=list(dict.fromkeys(k for x in history for k in x))
    with (report/'training_history.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(history)
    dump(report/'FINAL_AUDIT.json',dict(passed=True,formal_runs=10,epochs=4,seed=2026,e1_before_e2=True,finished=time.time(),runs=audit))
    primary=['AUROC','AUPRC','Rej-F1','Pos-Recall','mAP','mR+@5','G-mIoU@1']
    table=['| Backbone | Method | '+' | '.join(primary)+' |','|'+'---|'*(len(primary)+2)]
    for b in BACKBONES:
        for method in ('GMR','E1','E2'):
            m=flat(metrics[b,method,'test']);table.append('| '+NAMES[b]+' | '+method+' | '+' | '.join(f'{m[k]:.2f}' for k in primary)+' |')
    (report/'RESULTS.md').write_text('# 单阶段 GEQ / GEQ + PV：正式结果\n\n10/10 次训练已完成，全部使用 epoch 4、seed 2026。Standard Test，τ=0.4，数值为百分比。E1=端到端 GMR+GEQ；E2=端到端 GMR+GEQ+训练期 PV。\n\n'+'\n'.join(table)+'\n\n存在性 TPR（Pos-Recall）与无门控定位 recall 分别计算。完整 Val/Test、固定 τ=0.4/0.5/0.6 见 [all_metrics.csv](all_metrics.csv)；差值见 [comparisons_pp.csv](comparisons_pp.csv)；诊断见 [diagnostics_summary.csv](diagnostics_summary.csv)。\n\n[结果分析](../CONCLUSIONS.md) · [正式产物审计](FINAL_AUDIT.json)\n')
    c=['# 单阶段全量 GEQ 与 Post-Verification：结果结论\n',
       '10/10 次正式训练及 Val/Test 评估均完成。固定 seed=2026、epoch=4、原生 batch size、相同 GMR 初始化和损失；E2 从原始 GMR 独立初始化。Checkpoint 哈希、4 行训练历史、共享初始化、实际 Transformer/GEQ 权重更新，以及全部 20 份新模型预测的独立定位指标复算已通过最终审计。未追加训练、调参或按 Test 选择阈值/epoch。\n',
       '## 核心判断\n',
       '**本次固定预算没有支持 H1 所期待的普遍存在性排序改善，也没有支持 H2 所期待的正查询存在性召回恢复。** E1 的主要一致收益是固定阈值下拒识增强；PV 在部分 backbone 上改善个别定位指标，但没有同时改善排序、正例召回和定位的统一效果。\n',
       '1. **E1 vs 原始 GMR 的排序：** Test AUROC 在 2/5 个 backbone 提高（CG +0.12、Flash +0.39 pp），AUPRC 仅 Flash 提高（+0.70 pp）；其余 AUROC 下降 0.25–0.96 pp。Val 的 AUROC 仅 Moment-DETR 上升，QD 持平，CG/EaTR/Flash 下降。因此不能认为全量解冻普遍改善 Target Presence Ranking。\n',
       '2. **E1 vs 冻结 GEQ-only：** Test AUROC 在五个 backbone 全部下降，范围 0.13–0.57 pp；AUPRC 按未取整数值仅 CG 有不足 0.01 pp 的变化，其余下降。全量更新在本预算下没有提供明确排序收益。\n',
       '3. **E1 的定位与正例召回：** Test mAP 在 Moment/QD 下降，在 CG/EaTR/Flash 上升；EaTR 增幅最大 +1.79 pp，但其 Val mAP 为 -0.44 pp，不能外推为稳定收益。无存在性门控的 Top-5/IoU≥0.5 正查询命中率在 Moment/EaTR/Flash 提高，在 QD/CG 下降。固定 τ=0.4 的存在性 TPR 在五个 backbone 均下降 22.61–60.29 pp，同时 Rej-F1 均提高，说明工作点明显更偏向拒识。\n',
       '4. **E2 vs E1 的正例存在性召回：** Test TPR 无一提高；Moment/Flash 持平，QD/CG/EaTR 分别下降 0.74/1.47/0.55 pp。Val 仅 EaTR 有 +0.39 pp，其他持平或下降。PV 没有恢复本次最关心的存在性召回。\n',
       '5. **E2 的多片段/Top-K 定位：** Test mR@5 在 Moment/QD/Flash 上升，在 CG/EaTR 下降；mR+@5 仅 EaTR/Flash 上升（+1.05/+1.53 pp），Moment/QD/CG 下降。因此部分多片段收益不能解释为整体正查询召回改善。\n',
       '6. **AUROC 与 mAP/mR+ 的取舍：** E2 的 AUROC 相对 E1 在 Moment +0.32 pp，其余 -0.03 至 -0.22 pp；mAP 仅 Moment/QD 提高（+0.38/+0.51 pp），mR+@5 恰在这两者下降。EaTR/Flash 的 mR+ 提高则伴随 mAP 下降 0.56/0.54 pp。没有一个 backbone 同时提高 Test mAP 与 mR+@5。这里只描述差值，不把小幅下降称为统计等效。\n',
       '7. **瓶颈判断：** 存在性排序没有出现一致改善，是 H1 的直接限制；更保守的固定阈值工作点损失大量正例，是召回问题的直接表现。E2 Top-8 oracle query recall 只有 29.41%–75.74%，尤其 Moment 约 29%，候选覆盖限制了 PV 的可用输入。PV 全查询有效候选的质量 Pearson 为 0.17–0.44，只说明有限的总体质量相关性；该值混合了负查询零目标，不能当作正例内部排序能力或因果证据。多项辅助 loss 随 epoch 下降，也没有自动转化为评估收益。现有诊断不足以把全部失败归因于单一模块。\n',
       '## 逐 backbone 的实际变化\n']
    for b in BACKBONES:
        a=flat(metrics[b,'GMR','test']);e=flat(metrics[b,'E1','test']);p=flat(metrics[b,'E2','test']);f=flat(frozen[b,'test'])
        diff=lambda x,y,keys:'，'.join(f'{k} {x[k]-y[k]:+.2f} pp' for k in keys)
        c += [f'### {NAMES[b]}\n','E1−GMR：'+diff(e,a,['AUROC','AUPRC','Pos-Recall','mAP','mR+@5'])+'。\n','E1−冻结 GEQ：'+diff(e,f,['AUROC','AUPRC','mAP'])+'。\n','E2−E1：'+diff(p,e,['AUROC','AUPRC','Pos-Recall','mAP','mR@5','mR+@5'])+'。\n']
        for method in ('E1','E2'):
            d=diags[b,method,'test'];q=d['vs_gmr_localization_hits']['5'];oracle=d['top8_oracle'];corr=d['pv_quality_pearson']
            c.append(f"{method} 无门控 Top-5/IoU≥0.5 正查询命中率 {d['positive_only_localization']['5']['recall']:.2f}%；相对 E0 丢失 {q['lost_count']} 条、恢复 {q['restored_count']} 条；Top-8 Oracle query recall {oracle['query_recall_iou05']:.2f}%，GT coverage {oracle['mean_gt_coverage_iou05']:.2f}%；PV Pearson {corr if corr is not None else '不适用'}。\n")
    c += ['## 指标与证据边界\n','主表只报告原预设 τ=0.4；0.5/0.6 作为固定敏感性结果存档，不挑最佳工作点。mAP/mR/mR+ 与 positive-only localization recall 不做存在性 hard gating；FlashVTG 保留官方 soft calibration/NMS，raw mAP 和最终 mAP 分别存档。G-mIoU 的提升不能直接写成定位精度提升。\n','这是单种子、固定四轮预算、历史 Test 已观察的探索性结果。未检验统计显著性、跨种子稳定性或普遍负结论。当前证据支持“这份实现及固定预算未获得预期的综合改善”，不支持“GEQ/PV 在所有条件下无效”。\n','[唯一主表](reports/RESULTS.md) · [全部指标](reports/all_metrics.csv) · [完整差值](reports/comparisons_pp.csv) · [诊断汇总](reports/diagnostics_summary.csv) · [最终审计](reports/FINAL_AUDIT.json)\n']
    (ROOT/'CONCLUSIONS.md').write_text('\n'.join(c))
    print('FINAL_RESULTS_ORGANIZED',len(audit),flush=True)

if __name__=='__main__':main()
