#!/usr/bin/env python3
"""Validate full-coverage paired Test runs and publish a portable plot snapshot."""
import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path
import numpy as np
from soccergmr.io import read_jsonl,sha256,dump_json,fingerprint
from soccergmr.evaluation import evaluate
from soccergmr.evaluation.official.metrics import (_clean_pred_windows,_compute_set_iou_score,
    compute_G_mIoU,get_existence_score,prepare_submission_for_gmiou)

NAMES=('moment_detr','eatr','qd_detr','cg_detr','sim_detr')
METHODS=('gmr','geq_full')


def compress(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    with Path(source).open('rb') as data,destination.open('wb') as file:
        with gzip.GzipFile(filename='',mode='wb',fileobj=file,mtime=0) as zipped:shutil.copyfileobj(data,zipped)


def csv_write(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runs',required=True);parser.add_argument('--inputs',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();runs=Path(args.runs);dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(args.inputs).read_text());snapshot=dict(schema=1,protocol='Historical Standard Val selected GEQ best; patience 6, max epochs 20',
        train_split='Standard',test_used_for_selection=False,methods=list(METHODS),splits={})
    audit=[];curves=[];summary=[];oracle_rows=[]
    portable=dict(backbones={},data={})
    for split in ('standard','full'):
        gt_path=Path(manifest['data'][split]['labels']);gt=read_jsonl(gt_path)
        count=1036 if split=='standard' else 2970
        assert len(gt)==count and len({g['qid'] for g in gt})==count
        compress(gt_path,dest/'ground_truth'/(split+'.jsonl.gz'))
        label='SoccerGMR Standard Test (1036 queries)' if split=='standard' else 'SoccerGMR Full-Decodable Test (2970/2986 queries)'
        split_data=dict(label=label,rows=count,positive=sum(bool(g['relevant_windows']) for g in gt),gt_sha256=sha256(gt_path),backbones={})
        portable['data'][split]=dict(labels=f'assets/labels/{"Standard" if split=="standard" else "Full"}/test.jsonl',features=f'assets/{split}_features')
        for name in NAMES:
            series={};paired_hashes=[]
            for method in METHODS:
                root=runs/name/split/method;marker=json.loads((root/'complete.json').read_text())
                predictions=read_jsonl(root/'predictions.jsonl');metrics=json.loads((root/'metrics.json').read_text())
                assert marker['passed'] and marker['rows']==count and marker['label_sha256']==sha256(gt_path)
                assert marker['predictions_sha256']==sha256(root/'predictions.jsonl') and marker['metrics_sha256']==sha256(root/'metrics.json')
                assert marker['checkpoint_sha256']==manifest['backbones'][name][method]['sha256']
                assert marker['strict_state_load'] and marker['gradients_disabled'] and not marker['training_called'] and not marker['test_selection']
                paired_hashes.append(marker['checkpoint_sha256'])
                qids=[p['qid'] for p in predictions];assert len(qids)==count and len(set(qids))==count and set(qids)=={g['qid'] for g in gt}
                by_qid={p['qid']:p for p in predictions};predictions=[by_qid[g['qid']] for g in gt]
                scores=[];rejected=[];accepted={1:[],3:[]}
                for p,g in zip(predictions,gt):
                    assert p['vid']==g['vid'] and p['query']==g['query']
                    score,field=get_existence_score(p);assert field=='pred_exist_score' and np.isfinite(score) and 0<=score<=1
                    scores.append(score);rejected.append(_compute_set_iou_score([],g['relevant_windows']))
                    clean=_clean_pred_windows(p['pred_relevant_windows'],max_pred_windows=10)
                    for k in (1,3):accepted[k].append(_compute_set_iou_score([w[:2] for w in clean[:k]],g['relevant_windows']))
                ts=[i/100 for i in range(101)];ys={str(k):[] for k in (1,3)}
                for tau in ts:
                    for k in (1,3):
                        value=round(100*sum(a if s>tau else r for a,s,r in zip(accepted[k],scores,rejected))/count,2)/100
                        ys[str(k)].append(value)
                oracle={str(k):round(100*sum(a if g['relevant_windows'] else r for a,g,r in zip(accepted[k],gt,rejected))/count,2)/100 for k in (1,3)}
                for tau in (0.,.4,.5,.6,1.):
                    gated,_=prepare_submission_for_gmiou(predictions,cls_threshold=tau,max_pred_windows=10)
                    official=compute_G_mIoU(gated,gt,k_list=(1,3))
                    for k in (1,3):
                        value=ys[str(k)][int(round(tau*100))]
                        assert abs(value-official[f'G-mIoU@{k}']/100)<1e-12
                        if tau in (.4,.5,.6):assert abs(value-metrics['gmiou'][str(tau)][f'G-mIoU@{k}']/100)<1e-12
                ideal=[dict(p,pred_exist_score=float(bool(g['relevant_windows']))) for p,g in zip(predictions,gt)]
                gated,_=prepare_submission_for_gmiou(ideal,cls_threshold=.5,max_pred_windows=10)
                official=compute_G_mIoU(gated,gt,k_list=(1,3))
                for k in (1,3):
                    assert abs(oracle[str(k)]-official[f'G-mIoU@{k}']/100)<1e-12
                    assert oracle[str(k)]+1e-12>=max(ys[str(k)])
                # Independent full official recomputation also checks positive-only mAP.
                verified=evaluate(predictions,gt)
                assert verified['brief']==metrics['brief']
                row=dict(thresholds=ts,gmiou=ys,oracle=oracle,mAP=metrics['brief']['mAP'],
                    checkpoint_sha256=marker['checkpoint_sha256'],selected_epoch=marker['selected_epoch'],
                    prediction_sha256=marker['predictions_sha256'],max_video_tokens=marker['max_video_tokens'],export_profile=marker['export_profile'])
                series[method]=row
                for i,tau in enumerate(ts):curves.append(dict(split=split,backbone=name,method=method,threshold=tau,G1=ys['1'][i],G3=ys['3'][i]))
                oracle_rows.append(dict(split=split,backbone=name,method=method,oracle_G1=oracle['1'],oracle_G3=oracle['3']))
                record=dict(split=split,backbone=name,method=method,mAP=row['mAP'])
                for k in (1,3):
                    for index in (40,50,60):record[f'G{k}_{index}']=ys[str(k)][index]
                    record[f'G{k}_mean']=float(np.mean([ys[str(k)][i] for i in (40,50,60)]))
                summary.append(record)
                target=dest/'runs'/name/split/method;target.mkdir(parents=True,exist_ok=True)
                compress(root/'predictions.jsonl',target/'predictions.jsonl.gz')
                for file in ('metrics.json','complete.json'):shutil.copy2(root/file,target/file)
                # Copy original Standard metric reference into a portable rerun manifest.
                if split=='standard':
                    target_ref=dest/'references'/name/(method+'.json');target_ref.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(manifest['backbones'][name][method]['standard_metrics'],target_ref)
                    portable['backbones'].setdefault(name,dict(batch_size=manifest['backbones'][name]['batch_size']))[method]=dict(
                        path=f'checkpoints/{name}.{method}.ckpt',sha256=marker['checkpoint_sha256'],standard_metrics=str(target_ref))
                audit.append(dict(**marker,artifact=f'runs/{name}/{split}/{method}',official_metric_recomputation=True,
                    official_sweep_checks=[0.,.4,.5,.6,1.],oracle_official_exact=True,compressed_prediction_sha256=sha256(target/'predictions.jsonl.gz')))
                print('VERIFIED',split,name,method,flush=True)
            split_data['backbones'][name]=series
        snapshot['splits'][split]=split_data
    for name in NAMES:
        for method in METHODS:assert snapshot['splits']['standard']['backbones'][name][method]['checkpoint_sha256']==snapshot['splits']['full']['backbones'][name][method]['checkpoint_sha256']
    for row in summary:
        base=next(r for r in summary if r['split']==row['split'] and r['backbone']==row['backbone'] and r['method']=='gmr')
        for key in ('mAP','G1_mean','G3_mean'):row['delta_'+key+'_vs_gmr']=row[key]-base[key]
    dump_json(dest/'PLOT_DATA.json',snapshot);dump_json(dest/'AUDIT.json',dict(passed=True,runs=20,checks=audit,same_checkpoints_across_both_tests=True,training_called=False,test_selection=False))
    dump_json(dest/'INPUTS.example.json',portable)
    csv_write(dest/'CURVES.csv',curves);csv_write(dest/'METRICS.csv',summary);csv_write(dest/'ORACLE.csv',oracle_rows)
    print('All twenty full-coverage Test runs verified',flush=True)

if __name__=='__main__':main()
