#!/usr/bin/env python3
"""Four requested configurations in the original 30-row table's exact layout.

Formats existing fixed-epoch-four Test results only; no training or selection.
"""
import argparse,csv,hashlib,html,json,sys
from decimal import Decimal
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'ablations/component_ablation_20261023'
NEW=ROOT/'ablations/global_hard_roi_20261023'
OUT=SUITE/'tables'
sys.path.insert(0,str(ROOT/'backbonefolder/figures'))
from build_table2_pdf import font
NAMES={'moment_detr':'Moment-DETR','qd_detr':'QD-DETR','cg_detr':'CG-DETR','eatr':'EaTR','flashvtg':'FlashVTG'}
LABELS={1:'+ Global Event Query',2:'+ Global Event Query + LoRA',3:'+ Global Event Query + Hard-RoI',4:'+ Global Event Query + LoRA + Hard-RoI'}
METRICS=['N-acc','AUROC','Rej-F1','mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3']
GROUPS=[('Null-Set Rejection',0,3),('Temporal Localization',3,7),('Overall GMR',7,9)]
THRESHOLDS=['0.4','0.5','0.6']
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def metric_values(report,tau):
    b=report['brief'];return {k:Decimal(f"{b['Rej-F1@'+tau if k=='Rej-F1' else k]:.2f}") for k in METRICS}
def load():
    rows=[]
    for backbone,name in NAMES.items():
        base=SUITE/'results/formal'/backbone/'row_01'
        baseline_reports=json.loads((base/'test_metrics.json').read_text())
        baseline={t:metric_values(baseline_reports[t],t) for t in THRESHOLDS}
        initial_sha=json.loads((base/'CONFIG.json').read_text())['initialization']['sha256']
        sources={1:SUITE/'results/formal'/backbone/'row_02',2:SUITE/'results/formal'/backbone/'row_03',3:NEW/'results'/backbone/'global_hard',4:SUITE/'results/formal'/backbone/'row_04'}
        for row,directory in sources.items():
            complete=json.loads((directory/'COMPLETE.json').read_text())
            config=json.loads((directory/'CONFIG.json').read_text())
            assert complete['status']=='complete' and complete['epochs']==4
            assert config['job']['seed']==2026 and config['initialization']['sha256']==initial_sha
            effective=json.loads((directory/'EFFECTIVE_CONFIG.json').read_text());flags=effective.get('effective',effective)
            assert effective['passed'] and flags['global_event_query'] and not flags.get('saliency_6d') and not flags.get('candidate_presence_fusion')
            assert flags['lora']==(row in [2,4])
            assert bool(flags.get('hard_roi',flags.get('boundary_refiner')))==(row in [3,4])
            path=directory/'test_metrics.json';reports=json.loads(path.read_text())
            values={t:metric_values(reports[t],t) for t in THRESHOLDS}
            gains={t:{k:values[t][k]-baseline[t][k] for k in METRICS} for t in THRESHOLDS}
            rows.append(dict(backbone=backbone,name=name,row=row,label=LABELS[row],values=values,gains=gains,reference=name+' / frozen GMR',source=str(path.relative_to(ROOT)),source_sha256=digest(path),baseline_source=str((base/'test_metrics.json').relative_to(ROOT)),baseline_source_sha256=digest(base/'test_metrics.json')))
    assert len(rows)==20
    return rows

def export(rows,split,taus):
    combined=len(taus)>1
    stem=f'ABLATION_GLOBAL_LORA_HARD_{split.upper()}_'+('ALL_THRESHOLDS' if combined else 'tau_'+taus[0])
    target=OUT/stem
    columns=[(tau,key) for tau in taus for key in METRICS]
    ranks={(r['backbone'],tau,key):sorted({v['values'][tau][key] for v in rows if v['backbone']==r['backbone']},reverse=True) for r in rows for tau,key in columns}
    def emphasis(r,t,k):
        ordered=ranks[r['backbone'],t,k];v=r['values'][t][k]
        return 'best' if v==ordered[0] else 'second' if len(ordered)>1 and v==ordered[1] else ''
    caption=f'Global Event Query / LoRA / Hard-RoI on Soccer-GMR {split.title()}: 5 backbones x 4 configurations = 20 rows; '+('tau = 0.4 / 0.5 / 0.6' if combined else 'tau = '+taus[0])+'.'
    left=32;backbone_width=96;configuration_width=210;step=64;metric_left=left+backbone_width+configuration_width
    width=metric_left+step*len(columns)+32
    row_heights=[32 for r in rows]
    data_height=sum(row_heights)+4*9;header_height=90 if combined else 68;height=32+54+header_height+data_height+100
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(width/72,height/72),facecolor='white');ax=fig.add_axes([0,0,1,1],xlim=(0,width),ylim=(0,height));ax.axis('off')
    def text(x,y,s,size=10.5,style='Regular',ha='left'):return ax.text(x,y,s,ha=ha,va='center',fontproperties=font(size,style))
    def rule(y,x0=left,x1=None,lw=.55):ax.plot([x0,width-32 if x1 is None else x1],[y,y],color='black',lw=lw,solid_capstyle='butt')
    text(left,height-29,'Global Event Query / LoRA / Hard-RoI ablation: five backbones, four configurations'+(' (tau = '+taus[0]+')' if not combined else ' (tau = 0.4 / 0.5 / 0.6)'),13,'Bold')
    text(left,height-49,f'{split.title()} split: '+('1,036 queries (544 positive / 492 negative)' if split=='test' else '465 queries (255 positive / 210 negative)')+'; all values: %. Parentheses: gain versus the same-backbone GMR, pp.',10.5)
    top=height-69;rule(top,lw=1.15)
    if combined:
        for i,tau in enumerate(taus):
            x0=metric_left+i*9*step;text(x0+4.5*step,top-14,f'Operating threshold tau = {tau}',11.5,'Bold','center');rule(top-26,x0+4,x0+9*step-4,.6)
        group_y=top-39;metric_y=top-65;bottom=top-80
    else:group_y=top-15;metric_y=top-42;bottom=top-57
    text(left+4,metric_y,'Backbone',10.5,'Bold');text(left+backbone_width+4,metric_y,'Configuration (all include GMR)',10.5,'Bold')
    for ti,tau in enumerate(taus):
        start=metric_left+ti*9*step
        for label,a,b in GROUPS:text(start+(a+b)/2*step,group_y,label,10.5,'Italic','center');rule(group_y-10,start+a*step+4,start+b*step-4,.5)
        for mi,key in enumerate(METRICS):text(start+(mi+.5)*step,metric_y,key,10,'Regular','center')
    rule(bottom,lw=.75);y=bottom-16;underlines=[]
    for i,r in enumerate(rows):
        if i and r['row']==1:rule(y+13,lw=.7);y-=9
        yy=y+3
        if r['row']==1:text(left+4,yy,r['name'],10.5,'Bold')
        text(left+backbone_width+4,yy,f"{r['row']}. {r['label']}",10.2)
        for ci,(tau,key) in enumerate(columns):
            x=metric_left+(ci+.5)*step;rank=emphasis(r,tau,key)
            artist=text(x,yy,f"{r['values'][tau][key]:.2f}",11,'Bold' if rank=='best' else 'Regular','center')
            if rank=='second':underlines.append(artist)
            if r['gains'] is not None:text(x,y-8.5,f"({r['gains'][tau][key]:+.2f})",8.4,ha='center')
        y-=row_heights[i]
    rule(y+10,lw=1.15)
    notes=['Rows: Global Event Query; Global + LoRA; Global + Hard-RoI; Global + LoRA + Hard-RoI. All start independently from the same pretrained GMR.',
           'Every cell: value (delta versus the same-backbone GMR, same Test / threshold), pp. GMR reference is not displayed as a separate row.',
           'Bold: within-backbone best among these four rows; underline: second distinct best (ties retained). Fixed adaptation epoch 4; seed 2026.',
           'Gaussian-RoI, 6D saliency and candidate presence fusion: OFF. Native FlashVTG calibration / rounding / NMS retained. Single-seed exploratory results.']
    for ni,note in enumerate(notes):text(left,y-9-ni*14,note,9)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for artist in underlines:
        box=artist.get_window_extent(renderer).transformed(ax.transData.inverted());rule(box.y0+.4,box.x0,box.x1,.5)
    for ext in ['pdf','svg','png']:
        kwargs={'dpi':180} if ext=='png' else {}
        if ext=='pdf':kwargs['metadata']={'Title':caption,'Creator':'reproduce_lora_hard_ablation.py; same style as original formal ablation table'}
        fig.savefig(str(target)+'.'+ext,**kwargs)
    plt.close(fig)
    # CSVs keep raw numeric values and gains separately, as the existing README does.
    ids=['Backbone','Configuration ID','Configuration','GMR reference']
    names=[(f'tau={tau} ' if combined else '')+key for tau,key in columns]
    with Path(str(target)+'.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(ids+names)
        for r in rows:w.writerow([r['name'],r['row'],r['label'],r['reference']]+[f"{r['values'][t][k]:.2f}" for t,k in columns])
    with Path(str(target)+'_GAINS.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator="\n");w.writerow(ids+[n+' delta (pp)' for n in names])
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
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>'+html.escape(caption)+'</title><style>body{font-family:"Liberation Serif","Times New Roman",serif;margin:28px;background:#fff;color:#000}.scroll{overflow:auto}table{border-collapse:collapse;border-top:2px solid;border-bottom:2px solid}th,td{padding:7px 10px;white-space:nowrap;font-weight:normal}td{text-align:center;font-variant-numeric:tabular-nums}thead{border-bottom:1px solid}.block{border-top:1px solid}thead th{font-style:italic}tbody th{text-align:left}small{font-size:80%}caption{text-align:left;line-height:1.5;padding-bottom:12px}</style><div class="scroll"><table><caption><b>'+html.escape(caption)+'</b><br>Value (delta versus same-backbone GMR), %. Four matched configurations; no additional training.</caption><thead>'+head+'</thead><tbody>'+''.join(body)+'</tbody></table></div><p>'+'<br>'.join(map(html.escape,notes))+'</p></html>'
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
    metadata=dict(scope='global_lora_hard_comparison',displayed_configurations=20,new_training_runs=0,backbones=5,configurations=4,split=split,thresholds=taus,metrics=METRICS,gain_definition='Displayed two-decimal value minus displayed same-backbone frozen GMR value, same split/threshold; percentage points.',ranking='Within each backbone: maximum bold; second distinct maximum underlined, including ties.',records=[dict(backbone=r['backbone'],configuration_id=r['row'],configuration=r['label'],reference=r['reference'],source=r['source'],source_sha256=r['source_sha256'],values={t:{k:f'{v:.2f}' for k,v in r['values'][t].items()} for t in taus},gains=None if r['gains'] is None else {t:{k:f'{v:+.2f}' for k,v in r['gains'][t].items()} for t in taus}) for r in rows])
    Path(str(target)+'.json').write_text(json.dumps(metadata,indent=2)+'\n')
    for suffix in ['.csv','_GAINS.csv','.svg','.html','.md','.tex','.json']:
        artifact=Path(str(target)+suffix)
        artifact.write_text('\n'.join(line.rstrip() for line in artifact.read_text().splitlines())+'\n')
    print('Exported',stem,'20 comparison rows; thresholds',','.join(taus),flush=True)

def verify(rows):
    stem=OUT/'ABLATION_GLOBAL_LORA_HARD_TEST_ALL_THRESHOLDS'
    table=list(csv.DictReader(Path(str(stem)+'.csv').open()))
    gains=list(csv.DictReader(Path(str(stem)+'_GAINS.csv').open()))
    assert len(table)==len(gains)==20
    count=0
    for actual,delta,row in zip(table,gains,rows):
        assert actual['Backbone']==row['name'] and int(actual['Configuration ID'])==row['row']
        for tau in THRESHOLDS:
            for key in METRICS:
                field=f'tau={tau} {key}'
                assert Decimal(actual[field])==row['values'][tau][key]
                assert Decimal(delta[field+' delta (pp)'])==row['gains'][tau][key]
                count+=2
    report=dict(passed=True,displayed_configurations=20,backbones=5,configurations_per_backbone=4,new_training_runs=0,split='test',queries=1036,seed=2026,adaptation_epoch=4,thresholds=THRESHOLDS,metric_columns=27,values_and_gains_verified=count,configuration_order=LABELS,source_metric_hashes_verified=True,gain_reference='same-backbone frozen GMR, not displayed as a separate row',baseline_sources={r['backbone']:dict(path=r['baseline_source'],sha256=r['baseline_source_sha256']) for r in rows})
    Path(str(stem)+'_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified',count,'numeric values and GMR-relative gains.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    rows=load()
    if not a.verify_only:export(rows,'test',THRESHOLDS)
    verify(rows)
if __name__=='__main__':main()
