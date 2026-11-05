#!/usr/bin/env python3
"""Analytic GEQ head edits, with immutable source and explicit weight deltas."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from core import sha256,write_json,read_jsonl,video_partition,label,head_delta,suppress_columns


p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--analysis',required=True);p.add_argument('--train',required=True);p.add_argument('--output',required=True)
p.add_argument('--alpha',type=float,nargs='+',default=[.05,.1,.2,.4,.8]);p.add_argument('--seed',type=int,default=2026)
p.add_argument('--ffn-causal',help='Causal intervention report; only passing strengths can be edited')
p.add_argument('--ffn-only',action='store_true',help='Materialize baseline plus causally passing FFN edits only')
a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
analysis=json.loads((Path(a.analysis)/'analysis.json').read_text());source_sha=sha256(a.checkpoint)
assert source_sha==analysis['checkpoint_sha256'] and sha256(Path(a.analysis)/'direction.npz')==analysis['direction_sha256']
raw=np.load(Path(a.analysis)/'direction.npz');u=raw['direction'];mu=raw['mean_hard_null'];assert abs(np.linalg.norm(u)-1)<1e-5
rows=read_jsonl(Path(a.train)/'rows.jsonl');h=np.load(Path(a.train)/'activations.npy')
# Bias-only matches the direction edit's average logit shift for B in development.
ids=[i for i,r in enumerate(rows) if video_partition(r['vid'])=='development' and label(r,analysis['tau'],analysis['hard_null_threshold'])=='B']
if not ids:raise ValueError('No development B examples for control')
bias_per_alpha=float(((h[ids]-mu)@u).mean())
rng=np.random.default_rng(a.seed);rand=rng.standard_normal(256);rand/=np.linalg.norm(rand)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert saved.get('setting')=='E1'
state=saved['model'];wk='geq.global_head.weight';bk='geq.global_head.bias';assert tuple(state[wk].shape)==(1,256) and tuple(state[bk].shape)==(1,)
candidates=[('unedited',0.,'none')]+[(f'{kind}_a{alpha:g}',alpha,kind) for alpha in a.alpha for kind in ('direction','bias_only','random')]
if a.ffn_only:
    if not a.ffn_causal:raise ValueError('--ffn-only requires a passing causal report')
    candidates=[('unedited',0.,'none')]
causal=None
if a.ffn_causal:
    causal=json.loads(Path(a.ffn_causal).read_text())
    if causal['source_sha256']!=source_sha or causal['direction_sha256']!=analysis['direction_sha256'] or not causal['independent_development_videos']:
        raise ValueError('Causal report does not match source and direction')
    passed=[x['strength'] for x in causal['results'] if x['pass_causal_gate']]
    if not causal['passed'] or not passed: raise ValueError('No FFN intervention passed the causal gate')
    candidates += [(f'ffn_a{alpha:g}',alpha,'ffn') for alpha in passed]
for name,alpha,kind in candidates:
    out=root/name;out.mkdir();w,b=(np.zeros(256),0.)
    if kind=='direction':w,b=head_delta(u,mu,alpha)
    if kind=='random':
        # Match the joint weight-and-bias L2 norm of the direction edit.
        scale=math.sqrt(1+float(u@mu)**2)/math.sqrt(1+float(rand@mu)**2)
        w,b=head_delta(rand,mu,alpha*scale)
    if kind=='bias_only':b=alpha*bias_per_alpha
    edited={k:v.clone() for k,v in state.items()}
    edited[wk]+=torch.as_tensor(w,dtype=state[wk].dtype).reshape(1,256);edited[bk]+=torch.as_tensor([b],dtype=state[bk].dtype)
    changed_keys=[wk,bk]
    if kind=='ffn':
        fk=f"geq.video_reads.{causal['layer']}.feedforward.3.weight"
        value=suppress_columns(state[fk].numpy(),u,causal['columns'],alpha)
        edited[fk]=torch.as_tensor(value,dtype=state[fk].dtype);changed_keys.append(fk)
    delta={k:edited[k]-state[k] for k in changed_keys if not torch.equal(edited[k],state[k])}
    assert all(k.startswith('geq.') for k in delta)
    assert all(torch.equal(state[k],edited[k]) for k in state if not k.startswith('geq.'))
    payload={k:v for k,v in saved.items() if k not in ('model','optimizer','scheduler','rng')}
    payload.update(model=edited,weight_space_edit=dict(parent_sha256=source_sha,kind=kind,alpha=alpha,not_resumable_for_training=True))
    torch.save(payload,out/'edited.ckpt');torch.save(delta,out/'delta.pt')
    n=sum(int(x.count_nonzero()) for x in delta.values());denom=sum(x.numel() for k,x in state.items() if k.startswith('geq.'))
    norm=math.sqrt(sum(float((x.double()**2).sum()) for x in delta.values()))
    write_json(out/'manifest.json',dict(name=name,kind=kind,alpha=alpha,source_checkpoint=str(Path(a.checkpoint).resolve()),source_sha256=source_sha,checkpoint_sha256=sha256(out/'edited.ckpt'),delta_sha256=sha256(out/'delta.pt'),changed_tensors=list(delta),changed_parameters=n,geq_parameters=denom,changed_fraction=n/denom,delta_l2=norm,weight_l2=float(delta.get(wk,torch.zeros(1)).norm()),bias_delta=float(delta.get(bk,torch.zeros(1)).sum()),gradient_free=True,bias_only_control_matches='development B mean direction shift',random_seed=a.seed,causal_report_sha256=sha256(a.ffn_causal) if kind=='ffn' else None))
assert sha256(a.checkpoint)==source_sha
write_json(root/'index.json',dict(source_sha256=source_sha,analysis_sha256=sha256(Path(a.analysis)/'analysis.json'),candidates=[x[0] for x in candidates],dev_bias_per_alpha=bias_per_alpha))
