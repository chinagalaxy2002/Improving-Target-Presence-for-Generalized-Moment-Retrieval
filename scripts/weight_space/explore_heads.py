#!/usr/bin/env python3
"""Gradient-free geometric head exploration; discovery builds, development selects."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from core import read_jsonl,write_json,sha256,video_partition,label
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=Path(a.root);o=Path(a.output);o.mkdir(parents=True,exist_ok=False)
analysis=json.loads((r/'analysis/analysis.json').read_text());rows=read_jsonl(r/'train/rows.jsonl');h=np.load(r/'train/activations.npy').astype(float);tau=.5;hard=analysis['hard_null_threshold'];dis=np.array([video_partition(x['vid'])=='discovery' for x in rows]);dev=~dis;groups=np.array([label(x,tau,hard) for x in rows]);y=np.array([x['presence'] for x in rows]);logit=np.array([x['logit'] for x in rows]);means={g:h[dis&(groups==g)].mean(0) for g in 'ABCD'}
B=means['B'];C=means['C'];D=means['D'];A=means['A'];null=h[dis&~y];anchor=null.mean(0);raw=B-C
# Protect null variations using their leading principal directions; exclude all development samples.
_,sing,V=np.linalg.svd(null-anchor,full_matrices=False)
directions={'B_C':raw,'B_all_null':B-anchor,'B_D':B-D}
for rank in (1,2,4,8,16,32,64):directions[f'null_pca_protect_{rank}']=raw-V[:rank].T@(V[:rank]@raw)
# Mean constraints preserve A-vs-null and D-vs-C logits at first order.
for name,vectors in [('protect_D',[D-C]),('protect_A_D',[A-C,D-C])]:
 M=np.stack(vectors).T;directions[name]=raw-M@np.linalg.pinv(M)@raw
cov=(null-anchor).T@(null-anchor)/len(null)
for reg in (.1,1.,10.):directions[f'null_cov_{reg:g}']=np.linalg.solve(cov+reg*np.eye(256),B-anchor)
base=directions['protect_A_D']
for k in (4,8,16,32,64):
 v=np.zeros(256);ix=np.argsort(-np.abs(base))[:k];v[ix]=base[ix];directions[f'sparse_{k}']=v
# Boundary-local prototype uses only discovery FN-Loc near the decision boundary.
near=dis&(groups=='B')&(logit>np.log(.25/.75))
if near.sum()>3:directions['boundary_B_null']=h[near].mean(0)-anchor
results=[];vectors={};original_gate=logit>0;mask=dev;baseline_fp=int((mask&~y&original_gate).sum());baseline_fn=int((mask&y&~original_gate).sum());baseline_hard=int((mask&(groups=='C')&original_gate).sum())
for family,u in directions.items():
 if np.linalg.norm(u)<1e-10:continue
 u=u/np.linalg.norm(u);vectors[family]=u
 for alpha in (.01,.025,.05,.1,.2,.4,.8):
  for centering in ('hard_mean','null_mean','null_q95','null_q99'):
   center={'hard_mean':float(C@u),'null_mean':float(anchor@u),'null_q95':float(np.quantile(null@u,.95)),'null_q99':float(np.quantile(null@u,.99))}[centering]
   shift=alpha*(h@u-center);gate=logit+shift>0;rescue=int((mask&(groups=='B')&gate).sum());newfn=int((mask&y&original_gate&~gate).sum());fp=int((mask&~y&gate).sum());fn=int((mask&y&~gate).sum());hn=int((mask&(groups=='C')&gate).sum());safe=rescue>0 and fp<=baseline_fp and fn<=baseline_fn and hn<=baseline_hard
   results.append(dict(name=f'{family}_{centering}_a{alpha:g}',family=family,alpha=alpha,center=center,centering=centering,rescued=rescue,new_false_rejects=newfn,FP=fp,FN=fn,hard_false_accepts=hn,eligible=safe,net_correct=baseline_fp+baseline_fn-fp-fn))
# Freeze selection using Train development only. Preserve best distinct families, including rejected diagnostics.
ranked=sorted(results,key=lambda x:(x['eligible'],x['net_correct'],x['rescued'],-x['alpha']),reverse=True);short=[];seen=set()
for x in ranked:
 if x['family'] in seen:continue
 short.append(x);seen.add(x['family'])
 if len(short)==8:break
selected=next((x['name'] for x in ranked if x['eligible']),None)
for x in results:x['decision']='eligible' if x['eligible'] else 'reject: no rescue or increased FP/FN/hard-null false accepts'
write_json(o/'sweep.json',dict(discovery_queries=int(dis.sum()),development_queries=int(dev.sum()),test_used=False,val_used_for_selection=False,gradient_free=True,candidates=results,baseline=dict(FP=baseline_fp,FN=baseline_fn,hard_false_accepts=baseline_hard),selected=selected,shortlist=short,source_sha256=sha256(a.checkpoint),methods='Mean contrasts, null PCA orthogonal complement, mean protection, covariance linear solve, sparse coordinates. Closed-form data-derived edits, not a replication of the blog agent.'))
np.savez(o/'directions.npz',**vectors)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);state=saved['model'];wk='geq.global_head.weight';bk='geq.global_head.bias';edits=o/'edits';edits.mkdir();names=['unedited']+[x['name'] for x in short];denom=sum(v.numel() for k,v in state.items() if k.startswith('geq.'))
for name,x in [('unedited',None)]+[(x['name'],x) for x in short]:
 folder=edits/name;folder.mkdir();edited={k:v.clone() for k,v in state.items()}
 if x:
  edited[wk]+=torch.tensor(x['alpha']*vectors[x['family']],dtype=state[wk].dtype).reshape(1,-1);edited[bk]-=x['alpha']*x['center']
 delta={k:edited[k]-state[k] for k in (wk,bk) if not torch.equal(edited[k],state[k])};payload={k:v for k,v in saved.items() if k not in ('model','optimizer','scheduler','rng')};payload.update(model=edited,weight_space_edit=dict(parent_sha256=sha256(a.checkpoint),kind='direction',not_resumable_for_training=True));torch.save(payload,folder/'edited.ckpt');torch.save(delta,folder/'delta.pt');n=sum(int(v.count_nonzero()) for v in delta.values());norm=math.sqrt(sum(float((v.double()**2).sum()) for v in delta.values()))
 write_json(folder/'manifest.json',dict(name=name,kind='direction' if x else 'none',alpha=x['alpha'] if x else 0,checkpoint_sha256=sha256(folder/'edited.ckpt'),delta_sha256=sha256(folder/'delta.pt'),changed_parameters=n,changed_fraction=n/denom,delta_l2=norm,source_sha256=sha256(a.checkpoint),changed_tensors=list(delta)))
write_json(edits/'index.json',dict(candidates=names,source_sha256=sha256(a.checkpoint),analysis_sha256=sha256(r/'analysis/analysis.json')))
print(json.dumps(dict(tested=len(results),eligible=sum(x['eligible'] for x in results),selected=selected,shortlist=short)),flush=True)
