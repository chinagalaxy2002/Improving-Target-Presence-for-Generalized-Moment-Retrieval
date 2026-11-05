#!/usr/bin/env python3
"""Plot every completed fixed-checkpoint Test without waiting for other stages."""
import argparse,hashlib,json,os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soccergmr.io import read_jsonl,sha256
from soccergmr.evaluation.official.metrics import _clean_pred_windows,_compute_set_iou_score

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--batch',choices=['first','second'],required=True);args=p.parse_args()
root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
names=({'bm_detr':'BM-DETR','bam_detr':'BAM-DETR','uvcom':'UVCOM'} if args.batch=='first' else {'tr_detr':'TR-DETR','taskweave':'TaskWeave','ld_detr':'LD-DETR'})
styles={'gmr':('#555555','GMR baseline','-','o'),'geq_full':('#0072B2','GMR + GEQ Full','--','D')}
cachepath=out/('.'+args.batch+'_incremental_cache.json')
try:cache=json.loads(cachepath.read_text())
except (OSError,ValueError):cache={}

def data(name,stage,split):
    part=root/name/(stage+'_test')/('test_'+split);marker=part/'complete.json';metrics=part/'metrics.json';predictions=part/'predictions.jsonl';complete=root/name/stage/'complete.json';cfg=root/'configs'/(name+'.json')
    if not all(p.is_file() for p in (marker,metrics,predictions,complete,cfg)):return None
    try:
        m=json.loads(marker.read_text());cm=json.loads(complete.read_text());met=json.loads(metrics.read_text());config=json.loads(cfg.read_text());gtpath=Path(config['data']['splits']['test_'+split])
        if m['metrics_sha256']!=sha256(metrics) or m['predictions_sha256']!=sha256(predictions) or met['provenance']['checkpoint_sha256']!=cm['best_sha256'] or met['provenance']['label_sha256']!=sha256(gtpath):return None
        key='|'.join((m['predictions_sha256'],met['provenance']['label_sha256'],'v1'))
        if key in cache:return cache[key]
        gt=read_jsonl(gtpath);rows=read_jsonl(predictions);byid={r['qid']:r for r in rows}
        if len(gt)!=m['rows'] or len(byid)!=len(gt) or set(byid)!={g['qid'] for g in gt}:return None
        reject=np.array([_compute_set_iou_score([],g['relevant_windows']) for g in gt]);positive=np.array([bool(g['relevant_windows']) for g in gt]);scores=np.array([byid[g['qid']]['pred_exist_score'] for g in gt]);assert np.isfinite(scores).all()
        accepted={}
        for k in (1,3):
            accepted[k]=np.array([_compute_set_iou_score([w[:2] for w in _clean_pred_windows(byid[g['qid']]['pred_relevant_windows'],max_pred_windows=10)[:k]],g['relevant_windows']) for g in gt])
        curves={str(k):[100*float(np.where(scores>t/100,accepted[k],reject).mean()) for t in range(101)] for k in (1,3)}
        oracle={str(k):100*float(np.where(positive,accepted[k],reject).mean()) for k in (1,3)}
        result={'curves':curves,'oracle':oracle,'mAP':met['brief']['mAP'],'rows':len(gt)};cache[key]=result;return result
    except (OSError,KeyError,ValueError,json.JSONDecodeError,AssertionError):return None

now=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S CST')
for split,title in [('standard','Standard Test (1036)'),('full','Full-Decodable Test (2970/2986)')]:
    fig,axes=plt.subplots(3,2,figsize=(13,12),sharex=True,layout='constrained')
    fig.suptitle(f'{args.batch.title()} batch · {title} · fixed Val-selected checkpoints\nGMR vs GEQ Full · Test never used for selection · {now}',fontweight='bold',fontsize=13)
    for i,(name,label) in enumerate(names.items()):
        rows={stage:data(name,stage,split) for stage in styles}
        for j,k in enumerate((1,3)):
            ax=axes[i,j];valid=[]
            for stage,(color,method,linestyle,marker) in styles.items():
                item=rows[stage]
                if item is None:continue
                valid.append(item);ax.plot(np.linspace(0,1,101),item['curves'][str(k)],color=color,linestyle=linestyle,marker=marker,markevery=[40,50,60],markersize=4,label=f"{method} · mAP {item['mAP']:.2f}")
            if not valid:ax.text(.5,.5,'Waiting for completed Test',ha='center',va='center',transform=ax.transAxes,color='#64748b')
            else:
                oracle=max(a['oracle'][str(k)] for a in valid);ax.axhline(oracle,color='#334155',linestyle=':',linewidth=1,label=f'Oracle envelope {oracle:.2f}')
                ax.legend(fontsize=7.5,loc='best')
            if all(rows.values()):
                a,b=rows['gmr'],rows['geq_full'];delta=np.mean([b['curves'][str(k)][t]-a['curves'][str(k)][t] for t in (40,50,60)]);dm=b['mAP']-a['mAP']
                ax.text(.02,.03,f'ΔG{k} mean {delta:+.2f} pp',transform=ax.transAxes,color='#13804a' if delta>0 else '#c43c39' if delta<0 else '#64748b',fontsize=8)
                ax.text(.53,.03,f'ΔmAP {dm:+.2f} pp',transform=ax.transAxes,color='#13804a' if dm>0 else '#c43c39' if dm<0 else '#64748b',fontsize=8)
            ax.set_title(f'{label} · G-mIoU@{k}');ax.set_xlabel('Existence threshold');ax.set_ylabel('G-mIoU (0–100)');ax.set_xlim(0,1);ax.set_ylim(0,100);ax.grid(alpha=.2)
    for ext in ('png','svg'):
        target=out/(args.batch+'_batch_'+split+'_live.'+ext);tmp=out/(target.stem+'.tmp.'+ext);fig.savefig(tmp,dpi=160,facecolor='white');os.replace(tmp,target)
    plt.close(fig)
tmp=cachepath.with_suffix('.tmp');tmp.write_text(json.dumps(cache));os.replace(tmp,cachepath)
