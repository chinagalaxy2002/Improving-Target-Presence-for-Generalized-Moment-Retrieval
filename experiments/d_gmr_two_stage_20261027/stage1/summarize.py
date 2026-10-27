#!/usr/bin/env python3
"""Recompute E0, summarize real fixed-epoch outputs; missing runs stay pending."""
import csv, json
from common import ROOT, OLD, BACKBONES, rows, dump
from evaluation import evaluate

PRIMARY=['AUROC','AUPRC','Rej-F1','Pos-Recall','mAP','mR+@5','G-mIoU@1']
FULL=PRIMARY+['mR@1','mR@5','N-acc','TNR','G-mIoU@3','Positive-only Localization Recall@5/IoU0.5']

def flatten(report,tau):
    brief=report['brief'].copy();brief['Rej-F1']=brief[f'Rej-F1@{tau}'];return brief

def main():
    report_dir=ROOT/'reports';report_dir.mkdir(exist_ok=True)
    metrics={};frozen={};records=[];pending=[]
    for backbone in BACKBONES:
        for split in ('val','test'):
            gt=rows(ROOT/f'data/{split}.jsonl')
            baseline=rows(OLD/f'results/{backbone}/gmr/{split}_predictions.jsonl')
            cache=ROOT/f'reused_baselines/{backbone}'
            # Reuse predictions without re-training; official + independent recomputation.
            report=evaluate(baseline,gt,cache,split)
            frozen_pred=rows(OLD/f'results/{backbone}/global_only/{split}_predictions.jsonl') if (OLD/f'results/{backbone}/global_only/{split}_predictions.jsonl').exists() else None
            if frozen_pred:
                frozen[backbone,split]=evaluate(frozen_pred,gt,ROOT/f'reused_frozen_geq/{backbone}',split)
            for method in ('GMR','E1','E2'):
                if method!='GMR':
                    directory=ROOT/f'runs/{backbone}/{method}'
                    if not (directory/'COMPLETE.json').exists():
                        pending.append(f'{backbone}/{method}/{split}');continue
                    report=json.loads((directory/f'{split}_metrics.json').read_text())
                    # Independently recompute saved artifacts again at reporting time.
                    from independent_metrics import verify
                    verify(rows(directory/f'{split}_predictions.jsonl'),gt,report)
                metrics[backbone,method,split]=report
                for tau, r in report.items():
                    b=flatten(r,tau)
                    records.append(dict(Backbone=backbone,Method=method,Split=split,Threshold=tau,**{k:b[k] for k in FULL}))
    with (report_dir/'all_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['Backbone','Method','Split','Threshold']+FULL);writer.writeheader();writer.writerows(records)
    header='| Backbone | Method | '+' | '.join(PRIMARY)+' |'
    table=[header,'|'+'---|'*(len(PRIMARY)+2)]
    for b in BACKBONES:
        for method in ('GMR','E1','E2'):
            report=metrics.get((b,method,'test'))
            values=[f'{flatten(report["0.4"],"0.4")[k]:.2f}' for k in PRIMARY] if report else ['pending']*len(PRIMARY)
            table.append('| '+b+' | '+method+' | '+' | '.join(values)+' |')
    (report_dir/'RESULTS.md').write_text('# 固定 epoch 4：Standard Test，τ=0.4\n\n'+'\n'.join(table)+'\n\n数值为百分比。pending 表示尚无正式四轮结果，不能用 Preflight 数值替代。完整 Val/Test 与固定阈值 0.4/0.5/0.6 见 all_metrics.csv。\n')
    conclusions=['# CONCLUSIONS：单阶段 GEQ 与训练期 PV\n',f'已获得 {sum((b,m,"test") in metrics for b in BACKBONES for m in ("E1","E2"))}/10 个正式训练结果。Preflight 不属于实验结果。\n']
    for b in BACKBONES:
        conclusions.append(f'## {b}\n')
        if not all((b,m,'test') in metrics for m in ('GMR','E1','E2')):
            conclusions.append('正式 E1/E2 尚未全部完成，H1/H2 均不能判定。\n');continue
        base,e1,e2=[flatten(metrics[b,m,'test']['0.4'],'0.4') for m in ('GMR','E1','E2')]
        def differences(a,z,keys):return '，'.join(f'{k} {a[k]-z[k]:+.2f} pp' for k in keys)
        conclusions.append('E1 相对 GMR：'+differences(e1,base,['AUROC','AUPRC','Pos-Recall','mAP','mR+@5'])+'。\n')
        if (b,'test') in frozen:
            old=flatten(frozen[b,'test']['0.4'],'0.4')
            conclusions.append('E1 相对冻结 backbone 的 GEQ-only：'+differences(e1,old,['AUROC','AUPRC','Pos-Recall','mAP'])+'。\n')
        conclusions.append('E2 相对 E1：'+differences(e2,e1,['AUROC','AUPRC','Pos-Recall','mAP','mR@5','mR+@5'])+'。\n')
        for method in ('E1','E2'):
            diag=json.loads((ROOT/f'runs/{b}/{method}/test_diagnostics.json').read_text())
            conclusions.append(f'{method} Top-8 Oracle：{diag["top8_oracle"]}；PV 质量 Pearson：{diag["pv_quality_pearson"]}。\n')
        conclusions.append('候选覆盖不足限制 PV 的可恢复范围；质量相关性与存在性排序需要结合上面数值分别判断，不能单凭相关性作因果归因。\n')
    conclusions.append('这是 seed=2026、四轮预算的探索性比较。不报告统计显著性或跨种子稳定性。G-mIoU 包含拒识与定位贡献；mR+@5 衡量多片段召回，不代表正查询存在性召回。Test 不用于选轮次、阈值或权重。\n')
    (ROOT/'CONCLUSIONS.md').write_text('\n'.join(conclusions))
    dump(report_dir/'SUMMARY_STATUS.json',dict(available_new_test_results=sum((b,m,'test') in metrics for b in BACKBONES for m in ('E1','E2')),pending=pending))
    print('REPORTS_WRITTEN',flush=True)

if __name__=='__main__':main()
