#!/usr/bin/env python3
"""Offline tables and cross-artifact checks; values all come from new Full predictions."""
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from common import *

NAMES=dict(moment_detr='Moment-DETR',eatr='EaTR',qd_detr='QD-DETR',cg_detr='CG-DETR')
COMMON=['AUROC','mAP','mR@1','mR@5','mR+@5']
PER=['N-acc','Rej-F1','G-mIoU@1','G-mIoU@3']
EXTRA=['mR@3','mR+@1','mR+@3','mIoU@1','mIoU@3','mIoU@5','mIoU+@1','mIoU+@3','mIoU+@5']
TAUS=[.4,.5,.6]

def write_csv(path,data):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)

def fmt(v,key): return f'{v:.4f}' if key.startswith('G-mIoU') else f'{v:.2f}'
def delta(v,b,key):
    d=round(v-b,4 if key.startswith('G-mIoU') else 2)+0.
    return d,('+' if d>0 else '')+fmt(d,key)

def table(stem,title,records,columns,pdf):
    widths=[110,260]+[78]*len(columns);total=sum(widths);h=116+34*len(records)+105
    fig=plt.figure(figsize=((total+40)/72,h/72),facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(-20,total+20);ax.set_ylim(h,0);ax.axis('off')
    edges=[0]
    for w in widths:edges.append(edges[-1]+w)
    ax.text(0,24,title,fontsize=14,fontweight='bold')
    ax.text(0,45,'Standard-trained checkpoints | 2970/2986 queries (1633 positive, 1337 negative) | 16 historical exclusions',fontsize=9)
    ax.text(0,64,'Value above; (method - same-backbone GMR) below. G-mIoU: 0-1; others: percent / percentage points.',fontsize=9,color='#475569')
    headers=['Backbone','Method']+[k.replace('@tau=','\nτ=') for k in columns]
    for j,k in enumerate(headers):ax.text(edges[j]+3 if j<2 else (edges[j]+edges[j+1])/2,92,k,ha='left' if j<2 else 'center',va='center',fontsize=8.2,fontweight='bold')
    ax.plot([0,total],[116,116],color='#172b3a')
    bases={r['backbone']:r for r in records if r['method']=='gmr'}
    tex=[r'\noindent\textbf{'+title.replace('_',r'\_')+r'}\par',r'\resizebox{\textwidth}{!}{\begin{tabular}{ll'+'r'*len(columns)+r'}\toprule',
         'Backbone & Method & '+' & '.join(k.replace('+',r'$^{+}$').replace('@tau=',r' $\tau=$').replace('%',r'\%') for k in columns)+r' \\\midrule']
    last=None
    for i,r in enumerate(records):
        y=116+34*i;base=bases[r['backbone']];isbase=r['method']=='gmr'
        if isbase:ax.add_patch(plt.Rectangle((0,y),total,34,facecolor='#e2edf8',edgecolor='none'))
        elif list(NAMES).index(r['backbone'])%2:ax.add_patch(plt.Rectangle((0,y),total,34,facecolor='#f1f5f9',edgecolor='none'))
        if last and last!=r['backbone']:ax.plot([0,total],[y,y],color='#64748b',lw=.6);tex.append(r'\midrule')
        label=r['label']+(' (baseline)' if isbase else '')
        vals=[NAMES[r['backbone']] if last!=r['backbone'] else '',label]
        for j,s in enumerate(vals):ax.text(edges[j]+3,y+17,s,fontsize=8.7,va='center',fontweight='bold' if isbase else 'normal')
        texvals=vals[:]
        for j,k in enumerate(columns,2):
            v=r['values'][k];b=base['values'][k];value=fmt(v,k)
            ax.text(edges[j+1]-6,y+17 if isbase else y+10,value,ha='right',va='center',fontsize=8.5,fontweight='bold' if isbase else 'normal')
            if isbase:texvals.append(r'\textbf{'+value+'}')
            else:
                d,ds=delta(v,b,k);color='#15803d' if d>0 else '#b91c1c' if d<0 else '#64748b'
                ax.text(edges[j+1]-6,y+25,'('+ds+')',ha='right',va='center',fontsize=7.5,color=color)
                tc='Gain' if d>0 else 'Loss' if d<0 else 'Zero'
                texvals.append(r'\shortstack{'+value+r' \\ {\scriptsize\textcolor{'+tc+'}{('+ds+')}}}')
        tex.append(' & '.join(texvals)+r' \\');last=r['backbone']
    end=116+34*len(records);ax.plot([0,total],[end,end],color='#172b3a')
    ax.text(0,end+24,'Green: gain',color='#15803d',fontsize=9)
    ax.text(105,end+24,'Red: decrease',color='#b91c1c',fontsize=9)
    ax.text(225,end+24,'Gray: unchanged; all deltas are relative to GMR.',color='#64748b',fontsize=9)
    ax.text(0,end+45,'N-acc = TN / negatives. τ is the existence threshold. Localization uses positive ungated candidates.',fontsize=9)
    ax.text(0,end+65,'Full dataset and GEQ Full training mode are distinct. These are Full-Decodable results, not original Full coverage.',fontsize=9)
    ax.text(0,end+85,'Native 75-step video limit and postprocessing retained; temporal input limitations are detailed in README / INPUT_AUDIT.json.',fontsize=8.5)
    fig.savefig(HERE/(stem+'.png'),dpi=150)
    fig.savefig(HERE/(stem+'.pdf'),metadata={'CreationDate':None,'ModDate':None});pdf.savefig(fig);plt.close(fig)
    tex += [r'\bottomrule\end{tabular}}\par',r'G-mIoU: 0--1. Others: percent. Parentheses: method minus same-backbone GMR.\par']
    return '\n'.join(tex)

def main():
    audit=json.loads((HERE/'INPUT_AUDIT.json').read_text());records=[];checks=[];official={}
    assert audit['code_sha256']==code_files()
    for p,h in audit['input_sha256'].items():assert sha(p)==h,p
    feature_hashes=json.loads((HERE/'FEATURE_SHA256.json').read_text())
    assert sha(HERE/'FEATURE_SHA256.json')==audit['feature_sha256']
    for p,h in feature_hashes.items():assert sha(p)==h,p
    for e in CHECKPOINTS['checkpoints']:
        d=run_dir(e);complete=json.loads((d/'COMPLETE.json').read_text())
        assert complete['passed'] and complete['strict_load'] and complete['coverage']['rows']==2970
        assert complete['identity']['input_audit_sha256']==sha(HERE/'INPUT_AUDIT.json')
        assert complete['identity']['code_sha256']==code_files()
        assert complete['predictions_sha256']==sha(d/'predictions.jsonl')
        assert complete['metrics_sha256']==sha(d/'metrics.json')
        reports=json.loads((d/'metrics.json').read_text());official[e['backbone']+'/'+e['method']]=reports
        b=reports['0.4']['brief'];values={k:b[k] for k in COMMON+EXTRA}
        for t in TAUS:
            report=reports[str(t)];cls=report['GMR-CLS']['per_threshold'][str(t)]
            assert report['stats']['num_total']==2970 and report['stats']['num_positive']==1633 and report['stats']['num_negative']==1337
            assert all(report['brief'][k]==b[k] for k in COMMON+EXTRA)
            vals={'N-acc':100*cls['TN']/(cls['TN']+cls['FP']),'Rej-F1':cls['Rej-F1'],
                  **{k:report['brief'][k]/100 for k in ('G-mIoU@1','G-mIoU@3','G-mIoU@5')},
                  **{k:cls[k] for k in ('Acc','Rej-P','Rej-R')}}
            for k,v in vals.items():values[f'{k}@tau={t}']=v
        assert all(math.isfinite(v) and 0<=v<=(1 if k.startswith('G-mIoU') else 100) for k,v in values.items())
        if e['method'].endswith('_frozen'):
            assert complete['frozen']['native_tensors_equal'] and complete['frozen']['full_candidates_exact'] and complete['frozen']['all_localization_metrics_exact']
            parent='multi' if e['method'].startswith('multi') else 'gmr'
            pp=rows(HERE/'runs'/e['backbone']/parent/'predictions.jsonl');ps=rows(d/'predictions.jsonl')
            assert all({k:v for k,v in p.items() if k!='pred_exist_score'}=={k:v for k,v in q.items() if k!='pred_exist_score'} for p,q in zip(pp,ps))
            checks.append(dict(backbone=e['backbone'],method=e['method'],**complete['frozen']))
        records.append(dict(backbone=e['backbone'],method=e['method'],label=e['label'],values=values))
    assert len(records)==24 and len(checks)==8
    dump(HERE/'RESULTS.json',records);dump(HERE/'OFFICIAL_ALL_METRICS.json',official)
    wide=[dict(backbone=r['backbone'],method=r['method'],label=r['label'],**r['values']) for r in records]
    write_csv(HERE/'RESULTS.csv',wide)
    bases={r['backbone']:r for r in records if r['method']=='gmr'}
    delta_rows=[]
    for r in records:
        for k,v in r['values'].items():
            b=bases[r['backbone']]['values'][k]
            delta_rows.append(dict(backbone=r['backbone'],method=r['method'],metric=k,value=v,gmr_baseline=b,delta=v-b,
                                   unit='0-1' if k.startswith('G-mIoU') else 'percent; delta in percentage points'))
    write_csv(HERE/'DELTAS_VS_GMR.csv',delta_rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    preamble='\n'.join([r'\documentclass{article}',r'\usepackage[paperwidth=1850pt,paperheight=1120pt,margin=20pt]{geometry}',
                         r'\usepackage[T1]{fontenc}',r'\usepackage{booktabs,graphicx,xcolor}',
                         r'\definecolor{Gain}{HTML}{15803D}\definecolor{Loss}{HTML}{B91C1C}\definecolor{Zero}{HTML}{64748B}',
                         r'\pagestyle{empty}',r'\begin{document}'])
    groups=[('FULL_DECODABLE_TEST','SoccerGMR Full-Decodable Test (2970/2986) - primary metrics',COMMON+[f'{k}@tau={t}' for t in TAUS for k in PER]),
            ('OFFICIAL_LOCALIZATION_EXTRA','SoccerGMR Full-Decodable Test - additional official localization metrics',EXTRA),
            ('OFFICIAL_THRESHOLD_EXTRA','SoccerGMR Full-Decodable Test - additional official threshold metrics',[f'{k}@tau={t}' for t in TAUS for k in ('Acc','Rej-P','Rej-R','G-mIoU@5')])]
    tex=[]
    with PdfPages(HERE/'ALL_TABLES.pdf',metadata={'CreationDate':None,'ModDate':None}) as pdf:
        for stem,title,cols in groups:
            body=table(stem,title,records,cols,pdf);tex.append(body)
            (HERE/(stem+'.tex')).write_text(preamble+'\n'+body+'\n'+r'\end{document}'+'\n')
    (HERE/'ALL_TABLES.tex').write_text(preamble+'\n'+('\n'+r'\newpage'+'\n').join(tex)+'\n'+r'\end{document}'+'\n')
    lines=['# '+TITLE,'',
           '**24/24项正式Test已完成；每项覆盖2970/2970条（1633正例、1337负例），无新增跳过。原始Full Test为2986条，历史16条未覆盖（5正例、11负例），本结果不代表原始Full完整覆盖。**','',
           '四骨干均使用交接清单的原有Standard Train/Val best，SHA256逐文件核验、完整state_dict严格加载。全程eval()、禁用梯度，无训练、微调、重新选模或Test调参。GEQ Full是既有微调模式，和本次Full数据集不同。24项小批推理预检全部通过后，先完成8个GMR/Multi父模型的Full定位候选，再执行16项GEQ。','',
           '[主表 PNG](FULL_DECODABLE_TEST.png) · [主表 PDF](FULL_DECODABLE_TEST.pdf) · [全部表格 PDF](ALL_TABLES.pdf) · [LaTeX](ALL_TABLES.tex) · [CSV](RESULTS.csv) · [相对GMR增减](DELTAS_VS_GMR.csv) · [结果 JSON](RESULTS.json) · [全部官方指标及明细](OFFICIAL_ALL_METRICS.json)','',
           '![正式Test主表](FULL_DECODABLE_TEST.png)','',
           '公共指标每行只列一次。存在性阈值τ固定为0.4、0.5、0.6；定位mAP/mR在正例未门控候选上计算，AUROC/N-acc/Rej-F1/G-mIoU在完整正负域计算。N-acc=TN/负例数，不是Acc整体准确率。G-mIoU和其差值保持0–1；其余指标为百分数、差值为百分点。下方括号一律为“方法−同骨干GMR”；绿色增益、红色下降、灰色持平，GMR浅蓝加粗。CSV保留源精度，表格再四舍五入。官方Acc/Rej-P/Rej-R及其他定位指标在附表中完整报告。','',
           '[额外官方定位指标](OFFICIAL_LOCALIZATION_EXTRA.pdf) · [额外官方阈值指标](OFFICIAL_THRESHOLD_EXTRA.pdf)','',
           '八项Frozen逐张量验证原生权重与父检查点相同；正式预测直接复用对应父模型本轮Full候选的边界、顺序和分数，仅替换pred_exist_score。候选完全相同，官方mAP/mR/mR+/mIoU/mIoU+全部逐项一致，详见[AUDIT.json](AUDIT.json)。GEQ Full则严格加载其完整best（包含更新后的原生权重），独立生成候选。','',
           '输入使用Full独立qid文本特征，未复用Standard预测或文本特征；GT直接读取relevant_windows，不从moment再次扩窗。CLIP/SlowFast各自L2归一化，Moment-DETR按CLIP→SlowFast拼接，其余按SlowFast→CLIP；保留原TEF、32 token上限、10候选、排序、两秒取整及各骨干原生后处理。所有唯一特征的维度、有限值、时间步及SHA256已核验；推理时每批额外检查mask、非空有效token及输出有限性。','',
           '需要保留的输入限制：原配置视频上限75步（每步2秒）。Full中两段94/104步视频使用原配置截断前75步，仍按原duration缩放输出；Moment-DETR另外保留原生最大150秒时间后处理。没有调整模型序列上限、重采样或改写标签。此处“2970条覆盖”指每个查询均完成推理，不表示每段视频的全部时间范围都进入编码器。另有3段特征时长与标签duration相差超过2秒，沿用原生duration缩放；这些偏差会影响时序定位解释，未按Test修正或排除。详细视频、qid和长度见[输入审计](INPUT_AUDIT.json)。','',
           '原始Full Test与Standard Test共享1036个规范化视频stem+查询对（461段视频）；与Standard Train/Val的视频stem初查交集均为0。两组Test结果不能视为相互独立的实验样本。','',
           '历史16条排除来自两个ffprobe失败的视频，保持原始数据版本不变，逐条如下（完整记录：[JSONL](EXCLUDED_ORIGINAL_FULL_TEST_16.jsonl)）：','',
           '| qid | 视频 | GT数 |','|---:|---|---:|']
    for d in DATA['excluded_test_rows']:lines.append(f'| {d["qid"]} | {d["vid"]} | {len(d["relevant_windows"])} |')
    lines+=['','每项[runs/](runs/)含逐查询预测、官方三阈值指标、严格加载/小批预检、输入身份及完成标记；[logs/](logs/)保留日志。[INPUT_AUDIT.json](INPUT_AUDIT.json)含检查点、GT、manifest、代码及输入审计，[FEATURE_SHA256.json](FEATURE_SHA256.json)含全部4616个唯一特征哈希。任务使用TASK.lock防重，合法结果仅在输入、代码、检查点和产物哈希一致时复用。','',
            '复现：在本目录使用`/home/guoxiangyu/miniconda3/envs/gmr/bin/python -B run_all.py`。入口只执行推理；GPU调度计入其他CUDA进程，每卡至多2项、全机至多4项。`build_report.py`可离线重建表格。']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n')
    artifacts={str(p.relative_to(HERE)):sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.png','.pdf','.tex','.csv','.json','.md') and p.name not in ('AUDIT.json','COMPLETE.json','QUEUE_STATUS.json')}
    result=dict(passed=True,formal_completed=24,queries_each=2970,original_queries=2986,uncovered=16,
                frozen_checks=checks,inputs_unchanged=True,all_feature_hashes_rechecked=True,
                gmiou_unit='0-1',other_metric_unit='percent',delta_reference='same-backbone GMR',
                artifacts_sha256=artifacts,coverage_caveats=audit['over_75_steps']+audit['temporal_mismatches_gt_vs_features'])
    dump(HERE/'AUDIT.json',result)
    dump(HERE/'COMPLETE.json',dict(passed=True,title=TITLE,formal_completed=24,queries_each=2970,
                                 original_queries=2986,uncovered=16,input_audit_sha256=sha(HERE/'INPUT_AUDIT.json'),
                                 audit_sha256=sha(HERE/'AUDIT.json'),artifacts_sha256=artifacts))
    print('REPORT_COMPLETED 24/24; 2970/2986; frozen checks 8/8',flush=True)

if __name__=='__main__': main()
