#!/usr/bin/env python3
"""Publish formal 5x6 ablations: three thresholds, GMR-relative gains, CPU only."""
import argparse,csv,gzip,hashlib,html,json,sys
from decimal import Decimal
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'ablations/component_ablation_20261023'
OUT=SUITE/'tables'
sys.path.insert(0,str(ROOT/'backbonefolder/figures'))
from build_table2_pdf import font
NAMES={'moment_detr':'Moment-DETR','qd_detr':'QD-DETR','cg_detr':'CG-DETR','eatr':'EaTR','flashvtg':'FlashVTG'}
LABELS={1:'GMR (coupled baseline)',2:'+ Global Event Query',3:'+ LoRA',4:'+ Hard-RoI (no saliency)',5:'+ Gaussian-RoI (no saliency)',6:'Full DeRoI-GMR (+6D saliency)'}
METRICS=['N-acc','AUROC','Rej-F1','mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3']
GROUPS=[('Null-Set Rejection',0,3),('Temporal Localization',3,7),('Overall GMR',7,9)]
THRESHOLDS=['0.4','0.5','0.6']
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_predictions(path):return [json.loads(s) for s in gzip.decompress(path.read_bytes()).decode().splitlines() if s.strip()]
def rounded(value):return Decimal(f'{value:.2f}')
def metric_values(report,tau):
    b=report['brief'];return {k:rounded(b['Rej-F1@'+tau] if k=='Rej-F1' else b[k]) for k in METRICS}
def load():
    records={}
    for split in ['test','val']:
        rows=[]
        for backbone,name in NAMES.items():
            baseline=None
            for row in range(1,7):
                directory=SUITE/'results/formal'/backbone/f'row_{row:02d}'
                assert json.loads((directory/'COMPLETE.json').read_text())['row']==row
                reports=json.loads((directory/f'{split}_metrics.json').read_text())
                values={tau:metric_values(reports[tau],tau) for tau in THRESHOLDS}
                if row==1:baseline=values
                gains={tau:{key:values[tau][key]-baseline[tau][key] for key in METRICS} for tau in THRESHOLDS} if row>1 else None
                rows.append(dict(backbone=backbone,name=name,row=row,label=LABELS[row],values=values,gains=gains,reference=name+' / GMR (configuration 1)',source=str((directory/f'{split}_metrics.json').relative_to(ROOT)),source_sha256=digest(directory/f'{split}_metrics.json')))
        assert len(rows)==30 and {r['row'] for r in rows}==set(range(1,7));records[split]=rows
    return records

def recompute():
    sys.path.insert(0,str(SUITE))
    from official_eval.eval_main import evaluate_gmr
    from official_eval.metrics import prepare_submission_for_gmiou,compute_G_mIoU
    manifest=json.loads((SUITE/'provenance/SOURCE_COPY_MANIFEST.json').read_text())
    source_hashes={r['path']:r for r in manifest['files']}
    checks=[]
    for scope in ['formal','additional_diagnostic']:
        base=SUITE/'results/formal' if scope=='formal' else SUITE/'extra_diagnostics/results'
        configurations=range(1,7) if scope=='formal' else [7]
        for backbone in NAMES:
            for row in configurations:
                for split in ['val','test']:
                    directory=base/backbone/f'row_{row:02d}';path=directory/f'{split}_predictions.jsonl.gz'
                    meta=source_hashes[str(path.relative_to(SUITE))]
                    raw=gzip.decompress(path.read_bytes());assert hashlib.sha256(raw).hexdigest()==meta['uncompressed_sha256']
                    predictions=read_predictions(path);gt=[json.loads(x) for x in (SUITE/'data'/f'{split}.jsonl').read_text().splitlines() if x.strip()]
                    assert len(predictions)==len(gt)==(465 if split=='val' else 1036)
                    assert len({p['qid'] for p in predictions})==len(gt) and {p['qid'] for p in predictions}=={g['qid'] for g in gt}
                    reports=json.loads((directory/f'{split}_metrics.json').read_text())
                    # Localization/AUROC do not change with the hard gate. Compute them once.
                    actual=evaluate_gmr(predictions,gt,cls_thresholds=(.4,.5,.6),gmiou_cls_threshold=.4,map_num_workers=0,verbose=False)
                    negatives={g['qid'] for g in gt if not g['relevant_windows']}
                    for tau in THRESHOLDS:
                        gated,_=prepare_submission_for_gmiou(predictions,cls_threshold=float(tau),max_pred_windows=10)
                        g=compute_G_mIoU(gated,gt,k_list=(1,3))
                        actual_metrics={k:actual['brief'][k] for k in ['AUROC','mAP','mR@1','mR@5','mR+@5']}
                        actual_metrics.update(g)
                        actual_metrics['Rej-F1']=actual['GMR-CLS']['per_threshold'][tau]['Rej-F1']
                        actual_metrics['N-acc']=100*sum(p['pred_exist_score']<=float(tau) for p in predictions if p['qid'] in negatives)/len(negatives)
                        expected=metric_values(reports[tau],tau)
                        for key in METRICS:assert rounded(actual_metrics[key])==expected[key],(scope,backbone,row,split,tau,key)
                    checks.append(dict(scope=scope,backbone=backbone,row=row,split=split,queries=len(gt),prediction_sha256=meta['uncompressed_sha256'],all_nine_metrics_verified_at_thresholds=THRESHOLDS))
                print(f'CPU verified {scope}: {backbone} configuration {row}',flush=True)
    report=dict(passed=True,formal_experiments=30,additional_diagnostic_experiments=5,formal_split_pairs=60,formal_threshold_evaluations=180,additional_split_pairs=10,additional_threshold_evaluations=30,metric_columns=METRICS,checks=checks)
    (SUITE/'PUBLICATION_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')

def export(rows,split,taus):
    combined=len(taus)>1
    stem=f'ABLATION_{split.upper()}_'+('ALL_THRESHOLDS' if combined else 'tau_'+taus[0])
    target=OUT/stem
    columns=[(tau,key) for tau in taus for key in METRICS]
    ranks={(r['backbone'],tau,key):sorted({v['values'][tau][key] for v in rows if v['backbone']==r['backbone']},reverse=True) for r in rows for tau,key in columns}
    def emphasis(r,t,k):
        ordered=ranks[r['backbone'],t,k];v=r['values'][t][k]
        return 'best' if v==ordered[0] else 'second' if len(ordered)>1 and v==ordered[1] else ''
    caption=f'Formal component ablation on Soccer-GMR {split.title()}: 5 backbones x 6 configurations = 30 experiments; '+('tau = 0.4 / 0.5 / 0.6' if combined else 'tau = '+taus[0])+'.'
    left=32;backbone_width=96;configuration_width=210;step=64;metric_left=left+backbone_width+configuration_width
    width=metric_left+step*len(columns)+32
    row_heights=[26 if r['row']==1 else 32 for r in rows]
    data_height=sum(row_heights)+4*9;header_height=90 if combined else 68;height=32+54+header_height+data_height+100
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(width/72,height/72),facecolor='white');ax=fig.add_axes([0,0,1,1],xlim=(0,width),ylim=(0,height));ax.axis('off')
    def text(x,y,s,size=10.5,style='Regular',ha='left'):return ax.text(x,y,s,ha=ha,va='center',fontproperties=font(size,style))
    def rule(y,x0=left,x1=None,lw=.55):ax.plot([x0,width-32 if x1 is None else x1],[y,y],color='black',lw=lw,solid_capstyle='butt')
    text(left,height-29,'Formal component-wise ablation: five backbones, six configurations'+(' (tau = '+taus[0]+')' if not combined else ' (tau = 0.4 / 0.5 / 0.6)'),13,'Bold')
    text(left,height-49,f'{split.title()} split: '+('1,036 queries (544 positive / 492 negative)' if split=='test' else '465 queries (255 positive / 210 negative)')+'; all values: %. Parentheses: gain versus the same-backbone GMR, pp.',10.5)
    top=height-69;rule(top,lw=1.15)
    if combined:
        for i,tau in enumerate(taus):
            x0=metric_left+i*9*step;text(x0+4.5*step,top-14,f'Operating threshold tau = {tau}',11.5,'Bold','center');rule(top-26,x0+4,x0+9*step-4,.6)
        group_y=top-39;metric_y=top-65;bottom=top-80
    else:group_y=top-15;metric_y=top-42;bottom=top-57
    text(left+4,metric_y,'Backbone',10.5,'Bold');text(left+backbone_width+4,metric_y,'Configuration (cumulative)',10.5,'Bold')
    for ti,tau in enumerate(taus):
        start=metric_left+ti*9*step
        for label,a,b in GROUPS:text(start+(a+b)/2*step,group_y,label,10.5,'Italic','center');rule(group_y-10,start+a*step+4,start+b*step-4,.5)
        for mi,key in enumerate(METRICS):text(start+(mi+.5)*step,metric_y,key,10,'Regular','center')
    rule(bottom,lw=.75);y=bottom-16;underlines=[]
    for i,r in enumerate(rows):
        if i and r['row']==1:rule(y+13,lw=.7);y-=9
        yy=y+3 if r['row']>1 else y
        if r['row']==1:text(left+4,yy,r['name'],10.5,'Bold')
        text(left+backbone_width+4,yy,f"{r['row']}. {r['label']}",10.2)
        for ci,(tau,key) in enumerate(columns):
            x=metric_left+(ci+.5)*step;rank=emphasis(r,tau,key)
            artist=text(x,yy,f"{r['values'][tau][key]:.2f}",11,'Bold' if rank=='best' else 'Regular','center')
            if rank=='second':underlines.append(artist)
            if r['gains'] is not None:text(x,y-8.5,f"({r['gains'][tau][key]:+.2f})",8.4,ha='center')
        y-=row_heights[i]
    rule(y+10,lw=1.15)
    notes=['Formal rows only: configurations 1-6. The five fusion-enabled diagnostic runs (execution row 7) are published separately.',
           'Added rows: value (delta versus configuration-1 GMR, same backbone / split / threshold). Gains subtract displayed two-decimal values.',
           'Bold: within-backbone best; underline: second distinct best (ties retained). Fixed epoch 4; baseline frozen; seed 2026; no test selection.',
           'Native FlashVTG soft score calibration / rounding / NMS is retained. Matched CG/EaTR GMR exports differ from historical mAP by 0.01/0.02 pp.']
    for ni,note in enumerate(notes):text(left,y-9-ni*14,note,9)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for artist in underlines:
        box=artist.get_window_extent(renderer).transformed(ax.transData.inverted());rule(box.y0+.4,box.x0,box.x1,.5)
    for ext in ['pdf','svg','png']:
        kwargs={'dpi':180} if ext=='png' else {}
        if ext=='pdf':kwargs['metadata']={'Title':caption,'Creator':'reproduce_ablation_tables.py; reference style build_table2_pdf.py'}
        fig.savefig(str(target)+'.'+ext,**kwargs)
    plt.close(fig)
    # CSVs keep raw numeric values and gains separately, as the existing README does.
    ids=['Backbone','Configuration ID','Configuration','GMR reference']
    names=[(f'tau={tau} ' if combined else '')+key for tau,key in columns]
    with Path(str(target)+'.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(ids+names)
        for r in rows:w.writerow([r['name'],r['row'],r['label'],r['reference']]+[f"{r['values'][t][k]:.2f}" for t,k in columns])
    with Path(str(target)+'_GAINS.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(ids+[n+' delta (pp)' for n in names])
        for r in rows:
            if r['gains'] is not None:w.writerow([r['name'],r['row'],r['label'],r['reference']]+[f"{r['gains'][t][k]:+.2f}" for t,k in columns])
    md=['| '+' | '.join(['Backbone','Configuration']+names)+' |','| '+' | '.join(['---','---']+['---:']*len(columns))+' |']
    for r in rows:
        vals=[]
        for t,k in columns:
            label=f"{r['values'][t][k]:.2f}";rank=emphasis(r,t,k)
            if rank=='best':label=f'**{label}**'
            elif rank=='second':label=f'<u>{label}</u>'
            if r['gains'] is not None:label+=f" ({r['gains'][t][k]:+.2f})"
            vals.append(label)
        md.append('| '+' | '.join([r['name'],f"{r['row']}. {r['label']}"]+vals)+' |')
    Path(str(target)+'.md').write_text('\n'.join(md)+'\n')
    head='<tr><th rowspan="'+('3' if combined else '2')+'">Backbone</th><th rowspan="'+('3' if combined else '2')+'">Configuration</th>'
    if combined:head+=''.join(f'<th colspan="9">tau = {t}</th>' for t in taus)+'</tr><tr>'
    head+=''.join(f'<th colspan="{b-a}">{g}</th>' for t in taus for g,a,b in GROUPS)+'</tr><tr>'+''.join(f'<th>{html.escape(k)}</th>' for t,k in columns)+'</tr>'
    body=[]
    for r in rows:
        cells=[f'<th>{html.escape(r["name"])}</th>',f'<th>{r["row"]}. {html.escape(r["label"])}</th>']
        for t,k in columns:
            label=f"{r['values'][t][k]:.2f}";rank=emphasis(r,t,k)
            if rank=='best':label='<b>'+label+'</b>'
            elif rank=='second':label='<u>'+label+'</u>'
            if r['gains'] is not None:label+=f'<br><small>({r["gains"][t][k]:+.2f})</small>'
            cells.append('<td>'+label+'</td>')
        body.append('<tr'+(' class="block"' if r['row']==1 else '')+'>'+''.join(cells)+'</tr>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(caption)+'</title><style>body{font-family:"Liberation Serif","Times New Roman",serif;margin:28px;background:#fff;color:#000}.scroll{overflow:auto}table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid}th,td{padding:7px 10px;white-space:nowrap;font-weight:normal}td{text-align:center;font-variant-numeric:tabular-nums}thead{border-bottom:1px solid}.block{border-top:1px solid}thead th{font-style:italic}tbody th{text-align:left}small{font-size:80%}caption{text-align:left;line-height:1.5;padding-bottom:12px}</style><div class="scroll"><table><caption><b>'+html.escape(caption)+'</b><br>Value (delta versus same-backbone GMR), %. Formal configurations 1-6 only.</caption><thead>'+head+'</thead><tbody>'+''.join(body)+'</tbody></table></div><p>'+'<br>'.join(map(html.escape,notes))+'</p></html>'
    Path(str(target)+'.html').write_text(page)
    tex_rows=[]
    for r in rows:
        if r['row']==1 and tex_rows:tex_rows.append(r'\midrule')
        cells=[r['name'],str(r['row'])+'. '+r['label']]
        for t,k in columns:
            label=f"{r['values'][t][k]:.2f}";rank=emphasis(r,t,k)
            if rank=='best':label=r'\textbf{'+label+'}'
            elif rank=='second':label=r'\underline{'+label+'}'
            if r['gains'] is not None:label=r'\shortstack{'+label+r' \\ {\scriptsize ('+f"{r['gains'][t][k]:+.2f}"+r')}}'
            cells.append(label)
        tex_rows.append(' & '.join(cells)+r' \\')
    tex_head=(r'\multicolumn{2}{l}{} & '+' & '.join(r'\multicolumn{9}{c}{$\tau='+t+'$}' for t in taus)+r' \\'+'\n') if combined else ''
    tex_head+=r'\multicolumn{2}{l}{} & '+' & '.join(r'\multicolumn{'+str(b-a)+r'}{c}{\textit{'+g+'}}' for t in taus for g,a,b in GROUPS)+r' \\'+'\n'
    tex_head+='Backbone & Configuration & '+' & '.join(k.replace('mR+@5',r'mR$^+$@5') for t,k in columns)+r' \\'
    tex=r'\documentclass{article}'+'\n'+r'\usepackage[paperwidth='+f'{width:.1f}pt,paperheight={height:.1f}pt,margin=28pt'+r']{geometry}'+'\n'+r'\usepackage[T1]{fontenc}\usepackage{booktabs,graphicx}\pagestyle{empty}'+'\n'+r'\begin{document}\noindent\textbf{'+caption.replace('%',r'\%')+r'}\par\medskip\noindent\resizebox{\textwidth}{!}{\begin{tabular}{ll'+('r'*len(columns))+r'}\toprule'+'\n'+tex_head+'\n'+r'\midrule'+'\n'+'\n'.join(tex_rows)+'\n'+r'\bottomrule\end{tabular}}\par\medskip\footnotesize'+'\n'+r'\\'.join(n.replace('%',r'\%') for n in notes)+'\n'+r'\end{document}'+'\n'
    Path(str(target)+'.tex').write_text(tex)
    metadata=dict(scope='formal_component_ablation',formal_experiments=30,backbones=5,configurations=6,split=split,thresholds=taus,metrics=METRICS,gain_definition='Displayed two-decimal value minus displayed same-backbone configuration-1 GMR value, same split/threshold; percentage points.',ranking='Within each backbone: maximum bold; second distinct maximum underlined, including ties.',records=[dict(backbone=r['backbone'],configuration_id=r['row'],configuration=r['label'],reference=r['reference'],source=r['source'],source_sha256=r['source_sha256'],values={t:{k:f'{v:.2f}' for k,v in r['values'][t].items()} for t in taus},gains=None if r['gains'] is None else {t:{k:f'{v:+.2f}' for k,v in r['gains'][t].items()} for t in taus}) for r in rows])
    Path(str(target)+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Exported',stem,'30 formal rows; thresholds',','.join(taus),flush=True)

def verify_exports(records):
    count=0
    for split,rows in records.items():
        for taus in [[t] for t in THRESHOLDS]+[THRESHOLDS]:
            stem=f'ABLATION_{split.upper()}_'+('ALL_THRESHOLDS' if len(taus)>1 else 'tau_'+taus[0])
            table=list(csv.DictReader((OUT/(stem+'.csv')).open()));gains=list(csv.DictReader((OUT/(stem+'_GAINS.csv')).open()))
            assert len(table)==30 and len(gains)==25 and {int(r['Configuration ID']) for r in table}==set(range(1,7))
            for record,r in zip(table,rows):
                assert record['Backbone']==r['name'] and int(record['Configuration ID'])==r['row']
                for t in taus:
                    for k in METRICS:
                        field=(f'tau={t} ' if len(taus)>1 else '')+k
                        assert Decimal(record[field])==r['values'][t][k];count+=1
            for record,r in zip(gains,[r for r in rows if r['row']>1]):
                for t in taus:
                    for k in METRICS:
                        field=(f'tau={t} ' if len(taus)>1 else '')+k+' delta (pp)'
                        base=next(v for v in rows if v['backbone']==r['backbone'] and v['row']==1)
                        assert Decimal(record[field])==r['values'][t][k]-base['values'][t][k];count+=1
    report=dict(passed=True,formal_experiments=30,additional_diagnostics_in_formal_tables=False,table_count=8,values_and_gains_checked=count,thresholds=THRESHOLDS,split_tables=['test','val'])
    (SUITE/'TABLE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n');print('Table values/gains verified:',count)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--recompute-metrics',action='store_true');parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    if args.recompute_metrics:recompute()
    records=load();OUT.mkdir(parents=True,exist_ok=True)
    if not args.verify_only:
        for split,rows in records.items():
            for t in THRESHOLDS:export(rows,split,[t])
            export(rows,split,THRESHOLDS)
    verify_exports(records)
if __name__=='__main__':main()
