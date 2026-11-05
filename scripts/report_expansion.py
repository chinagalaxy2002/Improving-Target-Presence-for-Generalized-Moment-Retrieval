#!/usr/bin/env python3
"""Verify and report finalized paired expansion experiments; never select on Test."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soccergmr.io import read_jsonl, sha256, dump_json
from soccergmr.evaluation import evaluate
from soccergmr.evaluation.official.metrics import _clean_pred_windows, _compute_set_iou_score

NAMES={'bm_detr':'BM-DETR','bam_detr':'BAM-DETR','uvcom':'UVCOM'}
METHODS=('gmr','geq_full')


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root).resolve(); dest=root/'report';dest.mkdir(parents=True,exist_ok=True)
    summary=[];curves=[];audit=[];snapshot={}
    for split,count,label in [('standard',1036,'Standard Test (1036)'),('full',2970,'Full-Decodable Test (2970/2986)')]:
        snapshot[split]={}
        for name in NAMES:
            cfg=json.loads((root/'configs'/f'{name}.json').read_text())
            gt=read_jsonl(cfg['data']['splits']['test_'+split]);assert len(gt)==count
            ids={g['qid'] for g in gt};assert len(ids)==count
            snapshot[split][name]={}
            baseline_sha=json.loads((root/name/'gmr'/'complete.json').read_text())['best_sha256']
            for method in METHODS:
                complete=json.loads((root/name/method/'complete.json').read_text())
                assert complete['status']=='complete' and sha256(root/name/method/'best.pt')==complete['best_sha256']
                if method=='geq_full':assert complete['baseline_sha256']==baseline_sha
                part=root/name/(method+'_test')/('test_'+split)
                marker=json.loads((part/'complete.json').read_text());metrics=json.loads((part/'metrics.json').read_text());predictions=read_jsonl(part/'predictions.jsonl')
                assert marker['rows']==count and marker['predictions_sha256']==sha256(part/'predictions.jsonl') and marker['metrics_sha256']==sha256(part/'metrics.json')
                assert metrics['provenance']['checkpoint_sha256']==complete['best_sha256'] and metrics['provenance']['label_sha256']==sha256(Path(cfg['data']['splits']['test_'+split]))
                byid={p['qid']:p for p in predictions};assert len(byid)==count and set(byid)==ids
                ordered=[byid[g['qid']] for g in gt]; assert all(p['vid']==g['vid'] for p,g in zip(ordered,gt))
                assert evaluate(ordered,gt)['brief']==metrics['brief']
                truth=np.array([bool(g['relevant_windows']) for g in gt]); scores=np.array([p['pred_exist_score'] for p in ordered]);assert np.isfinite(scores).all()
                rejected=np.array([_compute_set_iou_score([],g['relevant_windows']) for g in gt])
                accepted={k:np.array([_compute_set_iou_score([w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'],max_pred_windows=10)[:k]],g['relevant_windows']) for p,g in zip(ordered,gt)]) for k in (1,3)}
                oracle={k:round(100*float(np.where(truth,accepted[k],rejected).mean()),2) for k in (1,3)}
                reject_all=round(100*float(rejected.mean()),2)
                vals={k:[] for k in (1,3)}
                for i in range(101):
                    tau=i/100;pred=scores>tau
                    row={'split':split,'backbone':name,'method':method,'threshold':tau}
                    for k in (1,3):
                        value=round(100*float(np.where(pred,accepted[k],rejected).mean()),2);vals[k].append(value);row['G'+str(k)]=value
                    curves.append(row)
                record={'split':split,'backbone':name,'method':method,'mAP':metrics['brief']['mAP'],'oracle_G1':oracle[1],'oracle_G3':oracle[3],'all_rejected_G1_G3':reject_all,'checkpoint_sha256':complete['best_sha256'],'selected_epoch':complete['best_epoch'],'stop_reason':complete['reason']}
                for t in (40,50,60):
                    pred=scores>t/100;tp=int((truth & pred).sum());fn=int((truth & ~pred).sum());fp=int((~truth & pred).sum());tn=int((~truth & ~pred).sum());official=metrics['GMR-CLS']['per_threshold'][str(t/100)]
                    assert (tp,fn,fp,tn)==(official['TP'],official['FN'],official['FP'],official['TN'])
                    for k in (1,3):
                        assert vals[k][t]==metrics['gmiou'][str(t/100)][f'G-mIoU@{k}'];record[f'G{k}_{t}']=vals[k][t];record[f'oracle_gap_G{k}_{t}']=oracle[k]-vals[k][t]
                    record.update({f'TP_{t}':tp,f'FN_{t}':fn,f'FP_{t}':fp,f'TN_{t}':tn,f'positive_rejection_rate_{t}':fn/(tp+fn),f'positive_retention_rate_{t}':tp/(tp+fn),f'null_retention_rate_{t}':fp/(fp+tn)})
                for k in (1,3):record[f'G{k}_mean']=np.mean([vals[k][i] for i in (40,50,60)])
                summary.append(record);snapshot[split][name][method]={'mAP':record['mAP'],'curves':vals,'oracle':oracle}
                audit.append({'split':split,'backbone':name,'method':method,'checkpoint_sha256':complete['best_sha256'],'label_sha256':metrics['provenance']['label_sha256'],'prediction_sha256':marker['predictions_sha256'],'rows':count})
                print('verified',split,name,method,flush=True)
            base=summary[-2];full=summary[-1]
            for key in ('mAP','G1_40','G1_50','G1_60','G1_mean','G3_40','G3_50','G3_60','G3_mean'):
                full['delta_'+key]=full[key]-base[key];base['delta_'+key]=0.
        fig,axes=plt.subplots(3,2,figsize=(12,13),sharex=True,sharey=True,layout='constrained')
        fig.suptitle(f'{label} · Standard-trained · GEQ 10-epoch patience',fontweight='bold')
        for i,(name,title) in enumerate(NAMES.items()):
            a,b=snapshot[split][name]['gmr'],snapshot[split][name]['geq_full']
            for j,k in enumerate((1,3)):
                ax=axes[i,j]
                for method,color in [('gmr','#555555'),('geq_full','#009e73')]:
                    entry=snapshot[split][name][method];ax.plot(np.linspace(0,1,101),entry['curves'][k],label=f"{method} mAP {entry['mAP']:.2f}",color=color)
                ax.axhline(max(a['oracle'][k],b['oracle'][k]),color='black',ls=':',lw=1,label='Oracle envelope')
                delta=np.mean([b['curves'][k][x]-a['curves'][k][x] for x in (40,50,60)]);dm=b['mAP']-a['mAP']
                ax.set_title(f'{title} · G-mIoU@{k}');ax.text(.02,.04,f'ΔG mean {delta:+.2f} pp',transform=ax.transAxes,color='#13804a' if delta>0 else '#c43c39' if delta<0 else '#64748b');ax.text(.53,.04,f'ΔmAP {dm:+.2f} pp',transform=ax.transAxes,color='#13804a' if dm>0 else '#c43c39' if dm<0 else '#64748b');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlabel('Existence threshold');ax.set_ylabel('G-mIoU (0–100)')
        for ext in ('png','pdf','svg'):fig.savefig(dest/(split.upper()+'_TEST.'+ext),dpi=200)
        plt.close(fig)
    # Rows have identical fields after delta assignment; preserve negative changes.
    write_csv(dest/'METRICS.csv',summary);write_csv(dest/'CURVES.csv',curves)
    dump_json(dest/'AUDIT.json',{'passed':True,'runs':len(audit),'no_test_selection':True,'source':'formal 10-patience GEQ protocol','checks':audit})
    print('report complete',dest,flush=True)

if __name__=='__main__':main()
