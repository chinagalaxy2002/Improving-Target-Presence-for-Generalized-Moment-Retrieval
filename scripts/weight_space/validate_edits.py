#!/usr/bin/env python3
"""Frozen-candidate evaluation using original predictions and analytic head logits.

Development selects a direction edit before any Val metrics are computed.
This does not substitute for a separate full model forward invariance audit.
"""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from core import read_jsonl,write_json,sha256,video_partition,label

p=argparse.ArgumentParser();p.add_argument('--legacy-code',required=True);p.add_argument('--train',required=True);p.add_argument('--val',required=True);p.add_argument('--analysis',required=True);p.add_argument('--edits',required=True);p.add_argument('--output',required=True);p.add_argument('--ffn-predictions-root',help='Actual fixed-candidate forwards at NAME/{train,val}')
a=p.parse_args();sys.path.insert(0,str(Path(a.legacy_code).resolve()))
from common import OLD
sys.path.insert(0,str(OLD));from official_eval.eval_main import evaluate_gmr
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
analysis=json.loads((Path(a.analysis)/'analysis.json').read_text());idx=json.loads((Path(a.edits)/'index.json').read_text());assert idx['analysis_sha256']==sha256(Path(a.analysis)/'analysis.json')
tau=analysis['tau'];hard=analysis['hard_null_threshold']
def load(path,split):
 p=Path(path);audit=json.loads((p/'audit.json').read_text());assert audit['split']==split and not audit['limited'] and audit['checkpoint_sha256']==idx['source_sha256']
 for f,digest in audit['output_sha256'].items():assert sha256(p/f)==digest
 rows=read_jsonl(p/'rows.jsonl');pred=read_jsonl(p/'predictions.jsonl');gt=read_jsonl(p/'ground_truth.jsonl');h=np.load(p/'activations.npy')
 assert len(rows)==len(pred)==len(gt)==len(h)
 assert all(x['qid']==y['qid']==z['qid'] for x,y,z in zip(rows,pred,gt))
 return rows,pred,gt,h
tr=load(a.train,'train');val=load(a.val,'val');assert {x['vid'] for x in tr[0]}.isdisjoint(x['vid'] for x in val[0])
ids=[i for i,r in enumerate(tr[0]) if video_partition(r['vid'])=='development']
def subset(data,ids):return tuple([x[i] for i in ids] if not isinstance(x,np.ndarray) else x[ids] for x in data)
dev=subset(tr,ids)
source=torch.load(Path(a.edits)/'unedited'/'edited.ckpt',map_location='cpu',weights_only=False)['model'];w=source['geq.global_head.weight'].numpy().reshape(-1);b=float(source['geq.global_head.bias'].item())
for data in (dev,val):
 error=np.max(np.abs(data[3]@w+b-np.array([r['logit'] for r in data[0]])))
 if error>3e-5:raise ValueError(f'Head reconstruction mismatch {error}')

def evaluate_one(name,data,base,split):
 rows,preds,gt,h=data;folder=Path(a.edits)/name;m=json.loads((folder/'manifest.json').read_text())
 assert sha256(folder/'edited.ckpt')==m['checkpoint_sha256'] and sha256(folder/'delta.pt')==m['delta_sha256']
 delta=torch.load(folder/'delta.pt',map_location='cpu',weights_only=True)
 if m['kind']=='ffn':
    if not a.ffn_predictions_root:raise ValueError('FFN edits require actual fixed-candidate forwards')
    f=Path(a.ffn_predictions_root)/name/split;fa=json.loads((f/'audit.json').read_text())
    assert fa['split']==split and fa['reference_sha256']==idx['source_sha256'] and fa['edited_sha256']==m['checkpoint_sha256']
    assert sha256(f/'predictions.jsonl')==fa['prediction_sha256']
    ff={str(p['qid']):p for p in read_jsonl(f/'predictions.jsonl')}
    assert len(ff)==fa['rows']
    for original in preds:
     changed=ff[str(original['qid'])]
     assert original['pred_relevant_windows']==changed['pred_relevant_windows'] and original['pred_relevant_windows_pre_exist']==changed['pred_relevant_windows_pre_exist']
    scores=np.array([ff[str(r['qid'])]['pred_exist_score'] for r in rows])
 else:
    assert set(delta)<= {'geq.global_head.weight','geq.global_head.bias'}
    dw=delta.get('geq.global_head.weight',torch.zeros(1,256)).numpy().reshape(-1);db=float(delta.get('geq.global_head.bias',torch.zeros(1)).item())
    # Preserve exact baseline probabilities for no-edit control; edits add analytic deltas to captured logits.
    logits=np.array([r['logit'] for r in rows])+h@dw+db
    scores=np.array([r['probability'] for r in rows]) if not delta else torch.sigmoid(torch.as_tensor(logits)).numpy()
 altered=copy.deepcopy(preds)
 for p,s in zip(altered,scores):p['pred_exist_score']=float(s)
 # All official candidate bytes are frozen at the original E1 export, including its historical score fusion and NMS.
 assert all(x['pred_relevant_windows']==y['pred_relevant_windows'] and x['pred_relevant_windows_pre_exist']==y['pred_relevant_windows_pre_exist'] for x,y in zip(preds,altered))
 y=np.array([r['presence'] for r in rows]);pred=scores>tau;old=np.array([r['probability']>tau for r in rows]);bl=np.array([label(r,tau,hard)=='B' for r in rows]);cn=np.array([label(r,tau,hard)=='C' for r in rows])
 metrics={}
 for t in (.4,.5,.6):
    z=evaluate_gmr(altered,gt,cls_thresholds=(t,),gmiou_cls_threshold=t,map_num_workers=0,verbose=False)['brief']
    metrics[str(t)]={'G-mIoU@1':z['G-mIoU@1'],'G-mIoU@3':z['G-mIoU@3'],'mAP':z['mAP']}
 if base is not None:
    assert all(metrics[t]['mAP']==base['metrics'][t]['mAP'] for t in metrics),'mAP changed'
 rescued=[r['qid'] for i,r in enumerate(rows) if bl[i] and pred[i]]
 newfp=[r['qid'] for i,r in enumerate(rows) if not y[i] and not old[i] and pred[i]]
 newfn=[r['qid'] for i,r in enumerate(rows) if y[i] and old[i] and not pred[i]]
 return {'name':name,'kind':m['kind'],'alpha':m['alpha'],'auroc':float(roc_auc_score(y,scores)),
 'confusion':{'TP':int((y&pred).sum()),'FN':int((y&~pred).sum()),'FP':int((~y&pred).sum()),'TN':int((~y&~pred).sum())},
 'fn_loc_count':int(bl.sum()),'fn_loc_rescued':len(rescued),'fn_loc_rescue_rate':len(rescued)/int(bl.sum()) if bl.sum() else None,
 'hard_null_count':int(cn.sum()),'hard_null_false_accept_rate':float(pred[cn].mean()) if cn.sum() else None,
 'rescued_qids':rescued,'new_false_accept_qids':newfp,'new_false_reject_qids':newfn,'metrics':metrics,
 'changed_parameters':m['changed_parameters'],'changed_fraction':m['changed_fraction'],'delta_l2':m['delta_l2'],
 'map_exact_on_fixed_predictions':True,'native_rerun_verified':False}

results={}
for name in idx['candidates']:
 results[name]=evaluate_one(name,dev,results.get('unedited'),'train')
base=results['unedited'];eligible=[]
for name,r in results.items():
 if r['kind'] not in ('direction','ffn'):r['development_decision']='control';continue
 if r['fn_loc_rescued']==0:r['development_decision']='reject: no FN-Loc rescue'
 elif r['confusion']['FP']>base['confusion']['FP']:r['development_decision']='reject: increases false accepts'
 elif r['hard_null_false_accept_rate']>base['hard_null_false_accept_rate']:r['development_decision']='reject: increases hard-null false accept rate'
 elif r['confusion']['FN']>base['confusion']['FN']:r['development_decision']='reject: increases total false rejects'
 else:r['development_decision']='eligible';eligible.append(r)
selected=max(eligible,key=lambda x:(x['fn_loc_rescued'],-x['delta_l2']))['name'] if eligible else None
for r in eligible:r['development_decision']='selected' if r['name']==selected else 'reject: less rescue or larger edit than selected'
write_json(out/'development.json',results)
write_json(out/'selection.json',{'selected':selected,'rule':'maximize FN-Loc rescue, then minimize edit norm; no increase in FP, FN or hard-null false accept; direction or causally gated FFN only','selected_using':'video-disjoint Train development only','development_sha256':sha256(out/'development.json')})
# Selection is frozen and written before any Val metrics are calculated.
independent={}
for name in idx['candidates']:independent[name]=evaluate_one(name,val,independent.get('unedited'),'val')
write_json(out/'val.json',independent)
print(json.dumps({'selected':selected,'candidates':len(results),'dev':len(dev[0]),'val':len(val[0])}))
