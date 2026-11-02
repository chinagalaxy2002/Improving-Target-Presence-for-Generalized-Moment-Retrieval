#!/usr/bin/env python3
"""Rebuild all SoccerGMR tables and curves from packaged predictions, offline."""
import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
if not hasattr(np,'trapz'):np.trapz=np.trapezoid

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'lib'))
from official_eval.eval_main import evaluate_gmr
from official_eval.metrics import (_clean_pred_windows,_compute_set_iou_score,
                                  compute_G_mIoU,prepare_submission_for_gmiou,get_existence_score)
import table_style
import curve_style

BACKBONES=curve_style.BACKBONES
METHODS=curve_style.METHODS
LABELS=dict(zip(METHODS,curve_style.LABELS))
COMMON=['AUROC','mAP','mR@1','mR@5','mR+@5']
EXTRA=['mR@3','mR+@1','mR+@3','mIoU@1','mIoU@3','mIoU@5','mIoU+@1','mIoU+@3','mIoU+@5']
PER=['N-acc','Rej-F1','G-mIoU@1','G-mIoU@3']
TAUS=[.4,.5,.6]
GRID=[i/100 for i in range(101)]
SPLITS=dict(standard='Standard Test (1036/1036)',full_decodable='Full-Decodable Test (2970/2986)')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,list(records[0]));w.writeheader();w.writerows(records)

def load_prediction(run):
    path=ROOT/run['predictions'];assert sha(path)==run['predictions_gzip_sha256']
    raw=gzip.decompress(path.read_bytes())
    assert hashlib.sha256(raw).hexdigest()==run['predictions_uncompressed_sha256']
    return [json.loads(l) for l in raw.decode().splitlines() if l.strip()]

def to_values(reports):
    brief=reports['0.4']['brief'];v={k:brief[k] for k in COMMON+EXTRA}
    for t in TAUS:
        report=reports[str(t)];c=report['GMR-CLS']['per_threshold'][str(t)]
        values={'N-acc':100*c['TN']/(c['TN']+c['FP']),'Rej-F1':c['Rej-F1'],
                **{k:report['brief'][k]/100 for k in ('G-mIoU@1','G-mIoU@3','G-mIoU@5')},
                **{k:c[k] for k in ('Acc','Rej-P','Rej-R')}}
        for k,x in values.items():v[f'{k}@tau={t}']=x
    return v

def sweep(pred,gt):
    index={p['qid']:p for p in pred};scores=[];rejected=[];accepted={k:[] for k in (1,3)}
    for d in gt:
        p=index[d['qid']];s,source=get_existence_score(p);assert source=='pred_exist_score'
        scores.append(s);rejected.append(_compute_set_iou_score([],d['relevant_windows']))
        clean=_clean_pred_windows(p['pred_relevant_windows'],max_pred_windows=10)
        windows=[[w[0],w[1]] for w in clean]
        for k in (1,3):accepted[k].append(_compute_set_iou_score(windows[:k],d['relevant_windows']))
    y={k:[round(100*sum(a if s>t else r for a,s,r in zip(accepted[k],scores,rejected))/len(gt),2)/100 for t in GRID] for k in (1,3)}
    oracle={k:round(100*sum(a if d['relevant_windows'] else r for a,d,r in zip(accepted[k],gt,rejected))/len(gt),2)/100 for k in (1,3)}
    oracle_pred=[dict(index[d['qid']],pred_exist_score=float(bool(d['relevant_windows']))) for d in gt]
    gated,_=prepare_submission_for_gmiou(oracle_pred,.5,10)
    direct=compute_G_mIoU(gated,gt,k_list=(1,3))
    for k in (1,3):assert abs(oracle[k]-direct[f'G-mIoU@{k}']/100)<1e-12 and max(y[k])<=oracle[k]+1e-12
    return y,oracle

def compute():
    runs=read(ROOT/'data/RUNS.json');assert len(runs)==48
    models=read(ROOT/'data/CHECKPOINTS.json');assert len(models)==24
    model_index={m['id']:m for m in models};gt_cache={};pred_cache={};record_cache={};records=[];curve_rows=[];oracle_rows=[];frozen_checks=[]
    for r in runs:
        key=(r['test'],r['backbone'],r['method']);model=model_index[r['model_id']]
        assert model['sha256']==r['checkpoint_sha256']
        gt=gt_cache.setdefault(r['test'],rows(ROOT/r['gt']))
        pred=load_prediction(r);index={p['qid']:p for p in pred}
        expected_rows=1036 if r['test']=='standard' else 2970
        assert len(pred)==len(index)==len(gt)==expected_rows
        assert set(index)=={d['qid'] for d in gt}
        assert all((index[d['qid']]['vid'],index[d['qid']]['query'])==(d['vid'],d['query']) for d in gt)
        reports={str(t):evaluate_gmr(pred,gt,cls_thresholds=(.4,.5,.6),gmiou_cls_threshold=t,map_num_workers=0,verbose=False) for t in TAUS}
        values=to_values(reports)
        for k,expected in r['expected_published'].items():
            assert abs(values[k]-expected)<1e-9,(key,k,values[k],expected)
        for k,v in values.items():assert np.isfinite(v) and 0<=v<=(1 if k.startswith('G-mIoU') else 100)
        record=dict(test=r['test'],backbone=r['backbone'],method=r['method'],label=r['label'],
                    checkpoint_sha256=r['checkpoint_sha256'],rows=expected_rows,values=values)
        dump(ROOT/'metrics'/r['test']/r['backbone']/f'{r["method"]}.json',reports)
        records.append(record);record_cache[key]=reports;pred_cache[key]=pred
        if r['method'].endswith('_frozen'):
            parent='multi' if r['method'].startswith('multi') else 'gmr';parent_key=(r['test'],r['backbone'],parent)
            pp={p['qid']:p for p in pred_cache[parent_key]}
            assert all({k:v for k,v in p.items() if k!='pred_exist_score'}==
                       {k:v for k,v in pp[p['qid']].items() if k!='pred_exist_score'} for p in pred)
            for t in reports:
                for metric in ('mAP_detail','mR_detail','mR+_detail','mIoU_detail','mIoU+_detail'):
                    assert reports[t][metric]==record_cache[parent_key][t][metric],(key,metric)
            frozen_checks.append(dict(test=r['test'],backbone=r['backbone'],method=r['method'],parent=parent,
                                      candidates_exact=True,all_localization_metrics_exact=True))
        y,oracle=sweep(pred,gt)
        for t in TAUS:
            for k in (1,3):assert abs(y[k][int(100*t)]-values[f'G-mIoU@{k}@tau={t}'])<1e-12
        for i,t in enumerate(GRID):curve_rows.append(dict(test=r['test'],backbone=r['backbone'],method=r['method'],threshold=t,**{f'G-mIoU@{k}':y[k][i] for k in (1,3)}))
        oracle_rows.append(dict(test=r['test'],backbone=r['backbone'],method=r['method'],**{f'oracle_G-mIoU@{k}':oracle[k] for k in (1,3)}))
        print('VERIFIED',*key,expected_rows,flush=True)
    assert len(frozen_checks)==16
    dump(ROOT/'data/RESULTS.json',records)
    csvwrite(ROOT/'data/RESULTS.csv',[dict(test=r['test'],backbone=r['backbone'],method=r['method'],label=r['label'],rows=r['rows'],**r['values']) for r in records])
    csvwrite(ROOT/'data/CURVES.csv',curve_rows);csvwrite(ROOT/'data/ORACLE_BY_METHOD.csv',oracle_rows)
    index={(r['test'],r['backbone'],r['method']):r for r in records};deltas=[]
    for r in records:
        base=index[(r['test'],r['backbone'],'gmr')]
        for k,v in r['values'].items():deltas.append(dict(test=r['test'],backbone=r['backbone'],method=r['method'],metric=k,value=v,gmr_baseline=base['values'][k],delta=v-base['values'][k],unit='0-1' if k.startswith('G-mIoU') else 'percent; delta in percentage points'))
    csvwrite(ROOT/'data/DELTAS_VS_GMR.csv',deltas)
    dump(ROOT/'AUDIT.json',dict(passed=True,formal_records=48,distinct_checkpoints=24,standard_rows_each=1036,full_rows_each=2970,
                              original_full_rows=2986,original_full_excluded=16,published_values_reproduced=True,
                              query_identity_exact=True,frozen_checks=frozen_checks,gmiou_unit='0-1',
                              oracle_presence_verified=True,training_rerun=False,model_inference_rerun=False,
                              predicted_query_rows=sum(r['rows'] for r in records),fixed_thresholds=TAUS))
    return records,curve_rows,oracle_rows

def render(records,curve_rows,oracle_rows):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':11,
                         'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    preamble='\n'.join([r'\documentclass{article}',r'\usepackage[paperwidth=1850pt,paperheight=1120pt,margin=20pt]{geometry}',r'\usepackage[T1]{fontenc}',
                          r'\usepackage{booktabs,graphicx,xcolor}',r'\definecolor{Gain}{HTML}{15803D}\definecolor{Loss}{HTML}{B91C1C}\definecolor{Zero}{HTML}{64748B}',r'\pagestyle{empty}',r'\begin{document}'])
    tex=[]
    with PdfPages(ROOT/'tables/ALL_TEST_METRICS.pdf',metadata={'CreationDate':None,'ModDate':None}) as pdf:
        for split,title in SPLITS.items():
            rr=[r for r in records if r['test']==split]
            table_style.HERE=ROOT/'tables';table_style.SPLIT_TEXT='Standard-trained checkpoints | '+title+(' | 544 positive / 492 negative' if split=='standard' else ' | 1633 positive / 1337 negative | 16 original exclusions')
            groups=[('PRIMARY',COMMON+[f'{k}@tau={t}' for t in TAUS for k in PER]),('LOCALIZATION_EXTRA',EXTRA),
                    ('THRESHOLD_EXTRA',[f'{k}@tau={t}' for t in TAUS for k in ('Acc','Rej-P','Rej-R','G-mIoU@5')])]
            for suffix,columns in groups:
                stem=split.upper()+'_'+suffix;body=table_style.table(stem,'SoccerGMR '+title+' — '+suffix.replace('_',' ').lower(),rr,columns,pdf)
                tex.append(body);(ROOT/'tables'/f'{stem}.tex').write_text(preamble+'\n'+body+'\n'+r'\end{document}'+'\n')
            curve_style.HERE=ROOT/'figures'/split;curve_style.HERE.mkdir(parents=True,exist_ok=True);curve_style.SPLIT_LABEL=title
            curves={(b,m):{k:[float(d[f'G-mIoU@{k}']) for d in curve_rows if d['test']==split and d['backbone']==b and d['method']==m] for k in (1,3)} for b in BACKBONES for m in METHODS}
            maps={(r['backbone'],r['method']):r['values']['mAP'] for r in rr}
            ceilings={(b,k):max(d[f'oracle_G-mIoU@{k}'] for d in oracle_rows if d['test']==split and d['backbone']==b) for b in BACKBONES for k in (1,3)}
            with PdfPages(curve_style.HERE/'BY_BACKBONE.pdf',metadata={'CreationDate':None,'ModDate':None}) as pages:
                for b,name in BACKBONES.items():
                    fig,axes=plt.subplots(1,2,figsize=(11.4,4.6),sharey=True)
                    for ax,k in zip(axes,(1,3)):curve_style.panel(ax,curves,b,k,ceilings);ax.set_title(f'G-mIoU@{k}')
                    curve_style.decorate(fig,axes,f'{name}: G-mIoU versus existence threshold')
                    curve_style.map_legend(fig,axes[0],b,maps);curve_style.save(fig,b);pages.savefig(fig);plt.close(fig)
            fig,axes=plt.subplots(4,2,figsize=(11.4,14.2),sharex=True,sharey=True)
            for i,(b,name) in enumerate(BACKBONES.items()):
                for j,k in enumerate((1,3)):
                    curve_style.panel(axes[i,j],curves,b,k,ceilings);axes[i,j].set_title(f'{name} · G-mIoU@{k}')
                    if i<3:axes[i,j].set_xlabel('')
            curve_style.decorate(fig,axes,'G-mIoU by backbone · mAP and oracle ceiling',bottom=.15,top=.89)
            fig.subplots_adjust(hspace=.68);fig.texts[1].set_y(.945)
            for i,b in enumerate(BACKBONES):curve_style.map_legend(fig,axes[i,0],b,maps,anchor=axes[i,0].get_position().y0-(.055 if i==3 else .012),loc='upper center')
            curve_style.save(fig,'ALL_BACKBONES');plt.close(fig)
    (ROOT/'tables/ALL_TEST_METRICS.tex').write_text(preamble+'\n'+('\n'+r'\newpage'+'\n').join(tex)+'\n'+r'\end{document}'+'\n')

def result_markdown(records):
    index={(r['test'],r['backbone'],r['method']):r for r in records}
    lines=['# SoccerGMR两套Test：完整正式主指标','',
           '同一批24个Standard训练best在Standard Test及Full-Decodable Test上各24项，共48条。G-mIoU为0–1；其他指标为%，差值为百分点（G-mIoU差值仍为0–1）。括号一律相对同骨干、同Test的GMR。','',
           '[全部指标PDF](tables/ALL_TEST_METRICS.pdf) · [CSV](data/RESULTS.csv) · [JSON](data/RESULTS.json) · [全部增减](data/DELTAS_VS_GMR.csv)','',
           '| Test | 骨干 | GMR baseline | GEQ Frozen | GEQ Full | Multi | Multi+GEQ Frozen | Multi+GEQ Full |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for split,title in SPLITS.items():
        for b,name in BACKBONES.items():
            base=index[(split,b,'gmr')]['values']['mAP'];vals=[]
            for m in METHODS:
                v=index[(split,b,m)]['values']['mAP'];d=round(v-base,2)+0.
                vals.append(f'**{v:.2f}**' if m=='gmr' else f'{v:.2f}<br><sub>'+('🟢' if d>0 else '🔴' if d<0 else '⚪')+f' ({d:+.2f})</sub>')
            lines.append('| '+title+' | '+name+' | '+' | '.join(vals)+' |')
    for split,title in SPLITS.items():
        lines+=['','## '+title,'',f'![{title}完整主指标](tables/{split.upper()}_PRIMARY.png)','',
                f'[主表PDF](tables/{split.upper()}_PRIMARY.pdf) · [额外定位指标](tables/{split.upper()}_LOCALIZATION_EXTRA.pdf) · [额外阈值指标](tables/{split.upper()}_THRESHOLD_EXTRA.pdf)','',
                f'![{title}曲线](figures/{split}/ALL_BACKBONES.png)','',
                f'[曲线总览PDF](figures/{split}/ALL_BACKBONES.pdf) · [按骨干分页PDF](figures/{split}/BY_BACKBONE.pdf)']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--metrics-only',action='store_true');args=parser.parse_args()
    records,curves,oracle=compute()
    if not args.metrics_only:render(records,curves,oracle)
    result_markdown(records)
    print('REBUILD_COMPLETE: 48/48 official records, 16/16 Frozen checks, both Test domains.',flush=True)

if __name__=='__main__':main()
