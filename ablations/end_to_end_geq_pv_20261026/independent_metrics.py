"""Independent NumPy implementation; never imports official metric helpers."""
import numpy as np

def overlaps(pred, gt):
    p=np.asarray(pred,dtype=float).reshape(-1,2);g=np.asarray(gt,dtype=float).reshape(-1,2)
    intersection=np.maximum(0,np.minimum(p[:,None,1],g[None,:,1])-np.maximum(p[:,None,0],g[None,:,0]))
    union=(p[:,1]-p[:,0])[:,None]+(g[:,1]-g[:,0])[None]-intersection
    return np.divide(intersection,union,out=np.zeros_like(intersection),where=union>0)

def match(mat, threshold, ap_order=False):
    used=set();hits=np.zeros(len(mat));values=[]
    for i,row in enumerate(mat):
        if ap_order:
            candidates=np.argsort(row)[::-1]
            j=next((int(j) for j in candidates if row[j]>=threshold and int(j) not in used),None)
        else:
            candidates=[j for j in range(len(row)) if j not in used]
            j=max(candidates,key=lambda j:row[j]) if candidates else None
            if j is not None and row[j]<threshold:j=None
        if j is not None:used.add(j);hits[i]=1;values.append(float(row[j]))
    return hits,values

def localization(pred,gt):
    mapping={p['qid']:p for p in pred};aps=[];recalls={1:[],3:[],5:[]};plus={1:[],3:[],5:[]}
    thresholds=np.round(np.linspace(.5,.95,10),2)
    for g in gt:
        windows=g['relevant_windows']
        if not windows:continue
        ranked=mapping[g['qid']]['pred_relevant_windows'];mat=overlaps([w[:2] for w in ranked],windows)
        apmat=overlaps([w[:2] for w in sorted(ranked[:10],key=lambda w:-w[2])],windows)
        for threshold in thresholds:
            hits,_=match(apmat,threshold,True)
            if len(hits):
                recall=np.cumsum(hits)/len(windows);precision=np.cumsum(hits)/np.arange(1,len(hits)+1)
                envelope=np.maximum.accumulate(np.r_[precision,0][::-1])[::-1][:-1]
                aps.append(float((np.diff(np.r_[0,recall])*envelope).sum()))
            else:aps.append(0.)
            for k in recalls:
                count=match(mat[:k],threshold)[0].sum();recalls[k].append(count/len(windows))
                if len(windows)>=2:plus[k].append(max(0,count-1)/(len(windows)-1))
    result={'mAP':100*float(np.mean(aps))}
    result.update({f'mR@{k}':100*float(np.mean(v)) for k,v in recalls.items()})
    result.update({f'mR+@{k}':100*float(np.mean(v)) if v else 0. for k,v in plus.items()})
    return result

def gmiou(pred,gt,tau):
    mapping={p['qid']:p for p in pred};result={}
    for k in (1,3,5):
        scores=[]
        for g in gt:
            p=mapping[g['qid']]
            windows=[w[:2] for w in p['pred_relevant_windows'] if w[1]>w[0]][:10] if p['pred_exist_score']>tau else []
            windows=windows[:k];target=g['relevant_windows']
            if not windows or not target:scores.append(float(not windows and not target));continue
            hits,values=match(overlaps(windows,target),0.)
            scores.append(sum(values)/(len(windows)+len(target)-int(hits.sum())))
        result[f'G-mIoU@{k}']=100*sum(scores)/len(scores)
    return result

def verify(pred,gt,reports):
    loc=localization(pred,gt)
    for tau,report in reports.items():
        independent=dict(loc,**gmiou(pred,gt,float(tau)))
        for key,val in independent.items():
            assert abs(val-report['brief'][key])<=.011,(key,val,report['brief'][key])
    return dict(passed=True,independent_localization=loc,thresholds=list(reports),tolerance_pp=.011)
