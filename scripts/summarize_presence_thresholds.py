#!/usr/bin/env python3
"""Publish one paired table for the two shared Val-threshold protocols."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = {'moment_detr':'Moment-DETR', 'eatr':'EaTR', 'qd_detr':'QD-DETR',
          'cg_detr':'CG-DETR', 'sim_detr':'Sim-DETR', 'bm_detr':'BM-DETR',
          'bam_detr':'BAM-DETR', 'uvcom':'UVCOM', 'tr_detr':'TR-DETR',
          'taskweave':'TaskWeave', 'ld_detr':'LD-DETR'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root
    with (root/'FINAL_TABLE.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==22
    assert len({(r['backbone'],r['threshold_method']) for r in rows})==22
    audit=json.loads((root/'AUDIT.json').read_text())
    assert audit['passed'] and audit['official_test_cell_checks']==264
    display=[]
    lines=['# 11骨干：两种Val选阈值方法在两套Test上的G-mIoU', '',
           '**A：GMR-Val**——在既定GMR best上选择Val G-mIoU@1最高的阈值；**B：GEQ-Val**——在既定GEQ best上选择Val G-mIoU@1最高的阈值。GEQ best仍由原训练流程中的Val G1(.4/.5/.6)均值选定，没有重新选择epoch。', '',
           '**每行的GMR与GEQ均使用该行同一个阈值**。阈值只在Standard Val上搜索0.00–1.00（步长0.01），两套Test与@3均直接沿用。并列时优先取最接近0.5、再取较小阈值。没有在Test上选阈值或模型。', '',
           '表中`GMR→GEQ`为指标原值（0–100），Δ=GEQ−GMR，单位为百分点。原值显示两位小数，Δ根据未舍入值计算并显示三位小数。', '',
           '| 骨干 | 方法 | τ* | Standard @1：GMR→GEQ | Δ@1 | Standard @3：GMR→GEQ | Δ@3 | Full @1：GMR→GEQ | Δ@1 | Full @3：GMR→GEQ | Δ@3 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    summary={}
    for r in rows:
        row=[LABELS[r['backbone']], 'A' if r['threshold_method']=='GMR_Val' else 'B', f"{float(r['threshold']):.2f}"]
        for split in ('standard','full'):
            for k in (1,3):
                a=float(r[f'{split}_GMR_G{k}']);b=float(r[f'{split}_GEQ_G{k}']);d=float(r[f'{split}_delta_G{k}'])
                assert abs((b-a)-d)<1e-10
                row.extend([f'{a:.2f} → {b:.2f}',f'{d:+.3f}'])
        display.append(row)
        lines.append('| '+' | '.join(row)+' |')
    for method,title in [('GMR_Val','A'),('GEQ_Val','B')]:
        chosen=[r for r in rows if r['threshold_method']==method]
        summary[title]={}
        for split in ('standard','full'):
            summary[title][split]={}
            for k in (1,3):
                d=np.array([float(r[f'{split}_delta_G{k}']) for r in chosen])
                summary[title][split][str(k)]={'positive':int((d>0).sum()),'negative':int((d<0).sum()),
                                              'mean_delta_pp':float(d.mean())}
    lines += ['', '**汇总（骨干等权，@1／@3）**：', '',
              f"- A：Standard 提高 {summary['A']['standard']['1']['positive']}/11、{summary['A']['standard']['3']['positive']}/11，平均增益 {summary['A']['standard']['1']['mean_delta_pp']:+.3f}／{summary['A']['standard']['3']['mean_delta_pp']:+.3f}；Full 提高 {summary['A']['full']['1']['positive']}/11、{summary['A']['full']['3']['positive']}/11，平均增益 {summary['A']['full']['1']['mean_delta_pp']:+.3f}／{summary['A']['full']['3']['mean_delta_pp']:+.3f} 个百分点。",
              f"- B：Standard 提高 {summary['B']['standard']['1']['positive']}/11、{summary['B']['standard']['3']['positive']}/11，平均增益 {summary['B']['standard']['1']['mean_delta_pp']:+.3f}／{summary['B']['standard']['3']['mean_delta_pp']:+.3f}；Full 提高 {summary['B']['full']['1']['positive']}/11、{summary['B']['full']['3']['positive']}/11，平均增益 {summary['B']['full']['1']['mean_delta_pp']:+.3f}／{summary['B']['full']['3']['mean_delta_pp']:+.3f} 个百分点。", '',
              '**如何解释**：B的数值增益更大，但它把baseline也放在GEQ的Val最佳阈值上；例如TaskWeave的GMR最佳阈值为0.53，GEQ为0.50，baseline在0.50的明显损失会放大B的差值。因此，回应“baseline只需调阈值”的质疑应优先看A，再结合各自Val阈值及固定候选排序分析；不能把B的较大差值全部归因于新增判别能力。', '',
              '**数据范围**：Standard Test 1036条；Full-Decodable Test 2970/2986条，两套重叠1036条。历史五骨干为原六轮早停／75-token，新六骨干为十轮早停／128-token。保留已有最佳检查点与正式Test预测，不重新训练；不同协议不作受控跨骨干排名。微小正增益不等于统计显著。', '',
              '[CSV完整数值](FINAL_TABLE.csv) · [PDF表格](FINAL_TABLE.pdf) · [PNG表格](FINAL_TABLE.png) · [Val阈值选择记录](VAL_SELECTION.json) · [审计](AUDIT.json)', '']
    (root/'FINAL_TABLE.md').write_text('\n'.join(lines))
    headers=['Backbone','Val selector','Threshold', 'Standard @1\nGMR → GEQ','Δ @1\n(pp)',
             'Standard @3\nGMR → GEQ','Δ @3\n(pp)', 'Full @1\nGMR → GEQ','Δ @1\n(pp)',
             'Full @3\nGMR → GEQ','Δ @3\n(pp)']
    plt.rcParams.update({'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(19,11))
    ax.axis('off')
    table=ax.table(cellText=display,colLabels=headers,cellLoc='center',
                   colWidths=[.095,.065,.055,.115,.065,.115,.065,.115,.065,.115,.065],
                   bbox=[0,.10,1,.82])
    table.auto_set_font_size(False);table.set_fontsize(9)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#d5dce4');cell.set_linewidth(.4)
        if i==0:
            cell.set_facecolor('#17324d');cell.set_text_props(color='white',weight='bold');cell.set_height(cell.get_height()*1.5)
        else:
            cell.set_facecolor('#f0f5fa' if ((i-1)//2)%2==0 else 'white')
            if j in (4,6,8,10):
                value=float(display[i-1][j]);cell.set_text_props(color='#147a45' if value>0 else '#b73232' if value<0 else '#555555')
    ax.text(0, .975, '11 backbones · Val-selected thresholds · paired Test G-mIoU', transform=ax.transAxes,fontsize=18,weight='bold')
    ax.text(0,.938,'A = threshold selected on GMR best Val.  B = threshold selected on GEQ best Val.  Both models use the SAME threshold in every row.',transform=ax.transAxes,fontsize=11)
    ax.text(0,.06,'Scores: 0–100. Deltas: GEQ − GMR, percentage points, computed before rounding. @3 reuses the @1-selected threshold.',transform=ax.transAxes,fontsize=10)
    ax.text(0,.032,'Standard: 1,036 queries. Full-Decodable: 2,970/2,986; overlaps Standard. Historical 5: patience 6 / video 75; new 6: patience 10 / video 128.',transform=ax.transAxes,fontsize=10)
    ax.text(0,.004,'B is not a comparison against an independently optimized baseline; its larger gains alone do not rule out an operating-point explanation.',transform=ax.transAxes,fontsize=10,color='#8e3b20')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):
        fig.savefig(root/f'FINAL_TABLE.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*') if p.is_file() and p.name!='TABLE_MANIFEST.json'}
    (root/'TABLE_MANIFEST.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    print(root/'FINAL_TABLE.md')


if __name__=='__main__':
    main()
