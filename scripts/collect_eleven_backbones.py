#!/usr/bin/env python3
"""Audit fixed selected Test artifacts and collect 11 paired backbones; no training."""
import argparse,csv,gzip,json,hashlib,shutil
from pathlib import Path
import numpy as np
from soccergmr.io import read_jsonl,sha256,dump_json
from soccergmr.evaluation import evaluate
from soccergmr.evaluation.official.metrics import _clean_pred_windows,_compute_set_iou_score
NAMES=dict(moment_detr='Moment-DETR',eatr='EaTR',qd_detr='QD-DETR',cg_detr='CG-DETR',sim_detr='Sim-DETR',bm_detr='BM-DETR',bam_detr='BAM-DETR',uvcom='UVCOM',tr_detr='TR-DETR',taskweave='TaskWeave',ld_detr='LD-DETR')

def main():
 p=argparse.ArgumentParser();p.add_argument('--first-root',type=Path,required=True);p.add_argument('--second-root',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results/test_thresholds_11_backbones'));a=p.parse_args()
 old=Path('results/test_thresholds');dest=a.output;dest.mkdir(parents=True,exist_ok=True)
 oldsnap=json.loads((old/'PLOT_DATA.json').read_text());oldchecks=json.loads((old/'AUDIT.json').read_text())['checks']
 snapshot={'schema':2,'train_split':'Standard','test_used_for_selection':False,'protocol':'Historical 5: GEQ patience6,75 video tokens. Expansion6: GEQ patience10,128 video tokens. Within-backbone comparisons only.','backbone_labels':NAMES,'splits':{}}
 audit=[];metrics=[];curves=[];oracle_rows=[];lineage=[]
 for split,count in [('standard',1036),('full',2970)]:
  base=oldsnap['splits'][split];snapshot['splits'][split]={k:v for k,v in base.items() if k!='backbones'};snapshot['splits'][split]['backbones']={}
  for name in NAMES:
   family='historical' if name in list(NAMES)[:5] else 'expansion'
   pair={};rawpair={}
   for method in ['gmr','geq_full']:
    if family=='historical':
     prior=next(c for c in oldchecks if (c['backbone'],c['split'],c['method'])==(name,split,method));part=old/prior['artifact'];met=json.loads((part/'metrics.json').read_text())
     compressed=part/'predictions.jsonl.gz';raw=gzip.decompress(compressed.read_bytes());pred=[json.loads(l) for l in raw.decode().splitlines()];source=prior
     assert hashlib.sha256(raw).hexdigest()==prior['predictions_sha256'];assert sha256(part/'metrics.json')==prior['metrics_sha256'];assert sha256(compressed)==prior['compressed_prediction_sha256']
     # Historical labels are already portable and independently hash bound.
     gtraw=gzip.decompress((old/'ground_truth'/(split+'.jsonl.gz')).read_bytes());assert hashlib.sha256(gtraw).hexdigest()==prior['label_sha256'];gt=[json.loads(l) for l in gtraw.decode().splitlines()]
     entry=dict(base['backbones'][name][method]);entry.update(protocol='historical GEQ patience 6; max20; video limit75',artifact=str(part.relative_to(dest.parent.parent)),selected_on='Standard Val')
    else:
     root=a.first_root if name in ['bm_detr','bam_detr','uvcom'] else a.second_root
     cfg=json.loads((root/'configs'/f'{name}.json').read_text());part=root/name/(method+'_test')/('test_'+split)
     stage=root/name/method;complete=json.loads((stage/'complete.json').read_text());marker=json.loads((part/'complete.json').read_text());met=json.loads((part/'metrics.json').read_text());raw=(part/'predictions.jsonl').read_bytes();pred=read_jsonl(part/'predictions.jsonl');gtpath=Path(cfg['data']['splits']['test_'+split]);gt=read_jsonl(gtpath)
     assert complete['status']=='complete' and sha256(stage/'best.pt')==complete['best_sha256']
     assert marker['rows']==count and marker['predictions_sha256']==hashlib.sha256(raw).hexdigest() and marker['metrics_sha256']==sha256(part/'metrics.json')
     assert met['provenance']['checkpoint_sha256']==complete['best_sha256'] and met['provenance']['source']==complete['source']
     assert met['provenance']['label_sha256']==sha256(gtpath) and sha256(gtpath)==base['gt_sha256']
     if method=='geq_full':assert complete['baseline_sha256']==json.loads((root/name/'gmr/complete.json').read_text())['best_sha256']
     saved=dest/'runs'/name/split/method;saved.mkdir(parents=True,exist_ok=True)
     with (saved/'predictions.jsonl.gz').open('wb') as handle:handle.write(gzip.compress(raw,mtime=0))
     for f in ['metrics.json','complete.json']:shutil.copy2(part/f,saved/f)
     dump_json(saved/'stage_complete.json',complete)
     history=dest/'training'/name/method;history.mkdir(parents=True,exist_ok=True)
     for f in ['history.jsonl','best_val_metrics.json','complete.json']:shutil.copy2(stage/f,history/f)
     architecture=json.loads(Path(cfg['architecture']).read_text());dump_json(history/'architecture.json',architecture)
     source={'source':complete['source'],'checkpoint_sha256':complete['best_sha256'],'label_sha256':sha256(gtpath),'predictions_sha256':marker['predictions_sha256'],'metrics_sha256':marker['metrics_sha256'],'compressed_prediction_sha256':sha256(saved/'predictions.jsonl.gz')}
     entry={'thresholds':[i/100 for i in range(101)],'mAP':met['brief']['mAP'],'checkpoint_sha256':complete['best_sha256'],'selected_epoch':complete['best_epoch'],'prediction_sha256':marker['predictions_sha256'],'max_video_tokens':128,'export_profile':'native family ranking, top10, actual duration','protocol':'GEQ patience10; max200; video limit128; seed2023','artifact':str(saved.relative_to(dest.parent.parent)),'selected_on':'Standard Val'}
     if split=='standard':lineage.append({'backbone':name,'method':method,**complete})
    assert len(gt)==count and len(pred)==count
    byid={r['qid']:r for r in pred};assert len(byid)==count and set(byid)=={r['qid'] for r in gt}
    pred=[byid[r['qid']] for r in gt];assert all(g['vid']==r['vid'] for g,r in zip(gt,pred))
    assert evaluate(pred,gt)['brief']==met['brief']
    truth=np.array([bool(g['relevant_windows']) for g in gt]);scores=np.array([r['pred_exist_score'] for r in pred]);assert np.isfinite(scores).all()
    rejected=(~truth).astype(float);accepted={k:np.array([_compute_set_iou_score([w[:2] for w in _clean_pred_windows(r['pred_relevant_windows'],max_pred_windows=10)[:k]],g['relevant_windows']) for r,g in zip(pred,gt)]) for k in [1,3]}
    sweep={str(k):[float(np.where(scores>i/100,accepted[k],rejected).mean()) for i in range(101)] for k in [1,3]};oracle={str(k):float(np.where(truth,accepted[k],rejected).mean()) for k in [1,3]}
    for i in [40,50,60]:
     for k in [1,3]:assert round(100*sweep[str(k)][i],2)==met['gmiou'][str(i/100)][f'G-mIoU@{k}']
    if family=='historical':
     assert all(np.allclose(entry['gmiou'][str(k)],np.round(sweep[str(k)],4),atol=1e-12) for k in [1,3])
     assert all(abs(entry['oracle'][str(k)]-round(oracle[str(k)],4))<1e-12 for k in [1,3])
     sweep=entry['gmiou'];oracle=entry['oracle']
    else:
     sweep={k:[round(x*100,2)/100 for x in v] for k,v in sweep.items()};oracle={k:round(v*100,2)/100 for k,v in oracle.items()}
    entry.update(gmiou=sweep,oracle=oracle,all_rejected=float(rejected.mean()));pair[method]=entry
    record={'split':split,'backbone':name,'method':method,'protocol_group':family,'mAP':entry['mAP'],'AUROC':met['GMR-CLS']['AUROC'],'max_video_tokens':entry['max_video_tokens']}
    for i in [40,50,60]:
     t=str(i/100);dec=scores>i/100;tp=int((truth&dec).sum());fn=int((truth&~dec).sum());fp=int((~truth&dec).sum());tn=int((~truth&~dec).sum())
     assert all(met['GMR-CLS']['per_threshold'][t][key]==value for key,value in [('TP',tp),('FN',fn),('FP',fp),('TN',tn)])
     record.update({f'TP_{i}':tp,f'FN_{i}':fn,f'FP_{i}':fp,f'TN_{i}':tn,f'positive_retention_{i}':tp/(tp+fn),f'positive_rejection_{i}':fn/(tp+fn),f'null_retention_{i}':fp/(fp+tn)})
     for k in [1,3]:record[f'G{k}_{i}']=sweep[str(k)][i]
    for k in [1,3]:
     record[f'G{k}_mean']=np.mean([sweep[str(k)][i] for i in [40,50,60]])
     record[f'oracle_G{k}']=oracle[str(k)]
     record[f'oracle_gap_G{k}_mean']=oracle[str(k)]-record[f'G{k}_mean']
    record['reject_all_G1_G3']=float(rejected.mean());rawpair[method]=record
    for i in range(101):curves.append({'split':split,'backbone':name,'method':method,'threshold':i/100,'G1':sweep['1'][i],'G3':sweep['3'][i]})
    oracle_rows.append({'split':split,'backbone':name,'method':method,'oracle_G1':oracle['1'],'oracle_G3':oracle['3'],'reject_all_G1_G3':float(rejected.mean())})
    audit.append({'passed':True,'backbone':name,'method':method,'split':split,'rows':count,'protocol_group':family,'qid_exact':True,'official_metric_recomputation':True,'official_fixed_threshold_checks':[.4,.5,.6],**{k:v for k,v in source.items() if k in ['source','checkpoint_sha256','label_sha256','predictions_sha256','metrics_sha256','compressed_prediction_sha256']},'artifact':entry['artifact']})
    print('verified',split,name,method,flush=True)
   for method in ['gmr','geq_full']:
    r=rawpair[method];b=rawpair['gmr']
    for key in ['mAP','G1_40','G1_50','G1_60','G1_mean','G3_40','G3_50','G3_60','G3_mean']:r['delta_'+key+'_vs_gmr']=r[key]-b[key]
    metrics.append(r)
   snapshot['splits'][split]['backbones'][name]=pair
 for name in NAMES:
  for method in ['gmr','geq_full']:assert snapshot['splits']['standard']['backbones'][name][method]['checkpoint_sha256']==snapshot['splits']['full']['backbones'][name][method]['checkpoint_sha256']
 for file,rows in [('METRICS.csv',metrics),('CURVES.csv',curves),('ORACLE.csv',oracle_rows)]:
  with (dest/file).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
 dump_json(dest/'PLOT_DATA.json',snapshot);dump_json(dest/'AUDIT.json',{'passed':True,'runs':len(audit),'backbones':11,'no_test_selection':True,'checks':audit});dump_json(dest/'TRAINING_LINEAGE.json',lineage)
 print('44 Test evaluations audited; 11 backbone pairs on each split',flush=True)
if __name__=='__main__':main()
