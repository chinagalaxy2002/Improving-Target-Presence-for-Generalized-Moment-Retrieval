#!/usr/bin/env python3
"""Whole-split endpoint metrics and E1-relative diagnostics from saved predictions."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
EXP=Path(__file__).resolve().parent;SRC=EXP.parent/'end_to_end_geq_pv_20261026';OLD=EXP.parent/'global_hard_roi_20261023'

def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):
    p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def aligned(path,gt):
    pred=rows(path);mp={p['qid']:p for p in pred};assert len(mp)==len(pred)==len(gt) and set(mp)=={g['qid'] for g in gt}
    return [mp[g['qid']] for g in gt]
def measure(pred,gt):
    y=np.array([bool(g['relevant_windows']) for g in gt]);s=np.array([p['pred_exist_score'] for p in pred],dtype=float)
    assert y.any() and (~y).any() and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
    pos=float(s[y].mean());neg=float(s[~y].mean());dp=1-pos;dn=neg;d=(dp+dn)/2;score=100*(1-d)
    error=np.abs(s-y.astype(float));weights=np.where(y,.5/y.sum(),.5/(~y).sum());assert abs(weights@error-d)<1e-12 and abs(score-50*(1+pos-neg))<1e-12
    stats={}
    for name,mask in [('positive',y),('negative',~y)]:
        stats[name]={'score_mean':float(s[mask].mean()),'score_std':float(s[mask].std()),'score_quantiles':np.quantile(s[mask],[0,.1,.25,.5,.75,.9,1]).tolist()}
        if all('pred_exist_logit_raw' in p for p in pred):
            z=np.array([p['pred_exist_logit_raw'] for p in pred]);stats[name].update(logit_mean=float(z[mask].mean()),logit_std=float(z[mask].std()),logit_quantiles=np.quantile(z[mask],[0,.1,.25,.5,.75,.9,1]).tolist())
    return dict(n=len(gt),n_positive=int(y.sum()),n_negative=int((~y).sum()),mu_pos=pos,mu_neg=neg,D_pos=dp,D_neg=dn,D_balanced=d,S=score,mean_gap=pos-neg,AUROC=100*roc_auc_score(y,s),AUPRC=100*average_precision_score(y,s),distribution=stats)
def hit(p,g,k):
    windows=np.array([w[:2] for w in p['pred_relevant_windows'][:k] if w[1]>w[0]],float).reshape(-1,2);target=np.array(g['relevant_windows'],float).reshape(-1,2)
    if not len(windows) or not len(target):return False
    inter=np.maximum(0,np.minimum(windows[:,None,1],target[None,:,1])-np.maximum(windows[:,None,0],target[None,:,0]));union=(windows[:,1]-windows[:,0])[:,None]+(target[:,1]-target[:,0])[None,:]-inter
    return bool((inter/np.maximum(union,1e-8)>=.5).any())
def compare(pred,reference,gt):
    out={'presence':{},'localization':{}}
    for tau in [.4,.5,.6]:
        lost=[];restored=[]
        for p,r,g in zip(pred,reference,gt):
            if not g['relevant_windows']:continue
            before=r['pred_exist_score']>tau;after=p['pred_exist_score']>tau
            if before and not after:lost.append(g['qid'])
            if after and not before:restored.append(g['qid'])
        out['presence'][str(tau)]=dict(lost_count=len(lost),restored_count=len(restored),lost_qids=lost,restored_qids=restored)
    for k in [1,5]:
        lost=[];restored=[]
        for p,r,g in zip(pred,reference,gt):
            if not g['relevant_windows']:continue
            before=hit(r,g,k);after=hit(p,g,k)
            if before and not after:lost.append(g['qid'])
            if after and not before:restored.append(g['qid'])
        out['localization'][str(k)]=dict(lost_count=len(lost),restored_count=len(restored),lost_qids=lost,restored_qids=restored)
    return out

def evaluate_setting(setting):
    directory=EXP/'runs'/setting;result={};sources={}
    for split in ['val','test']:
        gtpath=SRC/f'data/{split}.jsonl';gt=rows(gtpath);path=directory/f'{split}_predictions.jsonl';pred=aligned(path,gt);reference_path=SRC/f'runs/flashvtg/E1/{split}_predictions.jsonl';reference=aligned(reference_path,gt)
        result[split]=measure(pred,gt)
        result[split]['vs_e1']=compare(pred,reference,gt)
        existing=json.loads((directory/f'{split}_diagnostics.json').read_text());assert existing['independent_metric_verification']['passed']
        official=json.loads((directory/f'{split}_metrics.json').read_text())
        assert abs(official['0.4']['brief']['AUROC']-result[split]['AUROC'])<.011
        sources[split]={str(p):digest(p) for p in [gtpath,path,reference_path,directory/f'{split}_metrics.json',directory/f'{split}_diagnostics.json']}
    dump(directory/'endpoint_metrics.json',result);dump(directory/'ENDPOINT_AUDIT.json',dict(passed=True,whole_split_class_means=True,exact_qid_alignment=True,class_balanced_error_verified=True,no_test_selection=True,source_sha256=sources))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setting',choices=['GAP','ENDPOINT'],required=True);args=p.parse_args();evaluate_setting(args.setting)
