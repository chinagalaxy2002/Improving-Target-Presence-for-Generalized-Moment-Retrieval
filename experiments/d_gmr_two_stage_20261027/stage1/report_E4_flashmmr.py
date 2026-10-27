#!/usr/bin/env python3
"""Audit and report scratch FlashVTG+GEQ+PV, with equal-budget E3 when available."""
import csv,json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,dump,rows,sha
from independent_metrics import verify

DIR=ROOT/'runs/flashvtg/E4_scratch_flashmmr'
DEST=ROOT/'reports/FlashVTG_E4_flashmmr'
KEYS=['AUROC','AUPRC','Rej-F1','Pos-Recall','TNR','mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3']

def main():
    DEST.mkdir(exist_ok=True)
    assert json.loads((DIR/'COMPLETE.json').read_text())['epoch']==76
    h=rows(DIR/'history.jsonl');assert [x['epoch'] for x in h]==list(range(1,77))
    info=json.loads((DIR/'CHECKPOINT.json').read_text());assert sha(DIR/'epoch_076.ckpt')==info['sha256']
    cfg=json.loads((DIR/'CONFIG.json').read_text());e3cfg=json.loads((ROOT/'runs/flashvtg/E3/CONFIG.json').read_text())
    initial=cfg['initial_tensor_sha256'];assert {n:v for n,v in initial.items() if not n.startswith('pv.')}==e3cfg['initial_tensor_sha256']
    ck=torch.load(DIR/'epoch_076.ckpt',map_location='cpu',weights_only=False)
    assert ck['epoch']==76 and [g['lr'] for g in ck['optimizer']['param_groups']]==[1e-5,3e-5]
    assert any(n.startswith('pv.gru.') for n in ck['model']) and not any(n.startswith('base.exist_head.') for n in ck['model'])
    del ck
    checks={split:verify(rows(DIR/f'{split}_predictions.jsonl'),rows(ROOT/f'data/{split}.jsonl'),json.loads((DIR/f'{split}_metrics.json').read_text())) for split in ('val','test')}
    candidates=[('E3 scratch GEQ, epoch 26',ROOT/'runs/flashvtg/E3'),('E3 scratch GEQ, epoch 76',ROOT/'runs/flashvtg/E3_extra50'),('E4 scratch GEQ + PV, epoch 76',DIR)]
    reports={};records=[];missing=[]
    for label,path in candidates:
        if not (path/'COMPLETE.json').exists():missing.append(label);continue
        for split in ('val','test'):
            m=json.loads((path/f'{split}_metrics.json').read_text());reports[label,split]=m
            for tau in ('0.4','0.5','0.6'):
                f=dict(m[tau]['brief']);f['Rej-F1']=f[f'Rej-F1@{tau}']
                records.append(dict(Method=label,Split=split,Threshold=tau,**{k:f[k] for k in KEYS}))
    with (DEST/'all_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    mainkeys=['AUROC','AUPRC','Rej-F1','Pos-Recall','mAP','mR+@5','G-mIoU@1']
    lines=['# FlashVTG 随机初始化：加入 FlashMMR 风格 PV\n','E4 从随机初始化独立训练 76 轮，与 E3 的原始 GMR/GEQ 初始张量相同。Native presence BCE 在两者均关闭。仅增加候选内 GRU 质量与 GT 语义一致性辅助监督，不用 PV 推理门控/重排。\n','## Test，固定阈值 0.4\n','| Method | '+' | '.join(mainkeys)+' |','|'+'---|'*(len(mainkeys)+1)]
    for r in records:
        if r['Split']=='test' and r['Threshold']=='0.4':lines.append('| '+r['Method']+' | '+' | '.join(f'{r[k]:.2f}' for k in mainkeys)+' |')
    base='E3 scratch GEQ, epoch 76';new='E4 scratch GEQ + PV, epoch 76'
    if (base,'test') in reports:
        a=reports[new,'test']['0.4']['brief'];z=reports[base,'test']['0.4']['brief']
        lines.append('\nE4−E3（均 epoch 76）：'+'，'.join(f'{k} {a[k]-z[k]:+.2f} pp' for k in ['AUROC','AUPRC','Pos-Recall','mAP','mR@5','mR+@5','G-mIoU@1'])+'。')
        lines.append('这些是单种子固定预算的实际差异，不代表统计显著性或跨种子稳定收益。')
    else:lines.append('\nE3 的 76 轮尚未完成，不能使用 26 轮结果冒充等预算比较；完成后可重新运行本脚本。')
    diag=json.loads((DIR/'test_diagnostics.json').read_text())
    lines.append(f"\nE4 Top-8 Oracle query recall / GT coverage：{diag['top8_oracle']}；PV quality Pearson：{diag['pv_quality_pearson']}。候选覆盖与存在性排序需要分别判断，不能把 G-mIoU 提升直接当成定位收益。")
    lines.append('\n这是论文机制的工程适配，公开训练入口缺失，不能称为原论文完整复现；GRU 时间轴在候选内部，不另造候选间 Transformer。')
    (DEST/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    dump(DEST/'AUDIT.json',dict(passed=True,epoch=76,seed=2026,checkpoint_sha256=info['sha256'],same_shared_initialization_as_e3=True,independent_metrics=checks,missing_comparisons=missing))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(13,7));keys=['localization','geq','quality','representation','total']
    for ax,key in zip(axes.flat,keys):
        for split,color,style in [('train','#0072B2','-'),('val','#D55E00','--')]:
            ax.plot([x['epoch'] for x in h],[x[split+'_losses'][key] for x in h],color=color,ls=style,label=split.title())
        ax.set_title(key);ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.set_ylim(bottom=0);ax.legend(frameon=False)
    axes.flat[-1].axis('off');fig.suptitle('Scratch FlashVTG + GEQ + FlashMMR-style PV: fixed 76 epochs');fig.tight_layout()
    for ext in ('pdf','svg','png'):fig.savefig(ROOT/f'figures/FlashVTG_E4_flashmmr_loss_curves.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig);print('E4_REPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
