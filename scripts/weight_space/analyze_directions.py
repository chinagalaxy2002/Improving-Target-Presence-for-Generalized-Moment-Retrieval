#!/usr/bin/env python3
"""Discover B-vs-hard-null mean direction on Train discovery videos only."""
import argparse,json
from pathlib import Path
import numpy as np
from core import read_jsonl,write_json,sha256,video_partition,label,direction,distribution,summarize

p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--val',required=True);p.add_argument('--output',required=True)
p.add_argument('--tau',type=float,default=.5);p.add_argument('--hard-quantile',type=float,default=.75)
a=p.parse_args();o=Path(a.output);o.mkdir(parents=True,exist_ok=False)
def load(path,split):
    p=Path(path);audit=json.loads((p/'audit.json').read_text());assert audit['split']==split and not audit['limited']
    for file,checksum in audit['output_sha256'].items(): assert sha256(p/file)==checksum
    r=read_jsonl(p/'rows.jsonl');h=np.load(p/'activations.npy'); assert h.shape==(len(r),256)
    return r,h,audit
tr,th,ta=load(a.train,'train');va,vh,ve=load(a.val,'val');assert ta['checkpoint_sha256']==ve['checkpoint_sha256']
assert {r['vid'] for r in tr}.isdisjoint({r['vid'] for r in va}), 'Val video leakage'
ids={key:np.array([i for i,r in enumerate(tr) if video_partition(r['vid'])==key]) for key in ('discovery','development')}
assert set(tr[i]['vid'] for i in ids['discovery']).isdisjoint(tr[i]['vid'] for i in ids['development'])
null_scores=[tr[i]['candidate_score'] for i in ids['discovery'] if not tr[i]['presence']]
if not null_scores:raise ValueError('No null discovery samples')
hard=float(np.quantile(null_scores,a.hard_quantile)); dr=[tr[i] for i in ids['discovery']]
u,mu=direction(th[ids['discovery']],dr,a.tau,hard)
np.savez(o/'direction.npz',direction=u,mean_hard_null=mu)
reports={}
for name,rows,h in [('train_discovery',dr,th[ids['discovery']]),('train_development',[tr[i] for i in ids['development']],th[ids['development']]),('val_independent',va,vh)]:
    summary=summarize(rows,h,a.tau,hard); proj=(h-mu)@u
    summary['projection']={group:distribution([float(proj[i]) for i,r in enumerate(rows) if label(r,a.tau,hard)==group]) for group in ('A','B','C','D','other_positive')}
    summary['videos']=len(set(r['vid'] for r in rows));reports[name]=summary
write_json(o/'analysis.json',dict(checkpoint_sha256=ta['checkpoint_sha256'],train_sha256=ta['label_sha256'],val_sha256=ve['label_sha256'],tau=a.tau,hard_null_definition='top candidate relevance >= discovery-train null score quantile',hard_null_quantile=a.hard_quantile,hard_null_threshold=hard,partition='sha256(2026:video) mod 10: <6 discovery, otherwise development',direction='unit difference between B and C mean global-head input; exploratory on discovery only',direction_sha256=sha256(o/'direction.npz'),reports=reports))
try:
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
 for ax,(name,rows,h) in zip(axes,[('Train discovery',dr,th[ids['discovery']]),('Train development',[tr[i] for i in ids['development']],th[ids['development']]),('Val independent',va,vh)]):
    proj=(h-mu)@u
    groups=[[proj[i] for i,r in enumerate(rows) if label(r,a.tau,hard)==g] for g in 'ABCD']
    for g,values in zip('ABCD',groups):
        if values:ax.hist(values,bins=25,alpha=.45,density=True,label=f'{g} (n={len(values)})')
    ax.set_title(name);ax.set_xlabel('Projection centered at hard-null mean');ax.legend(fontsize=7)
 axes[0].set_ylabel('Density');fig.tight_layout();fig.savefig(o/'projection.png',dpi=160);plt.close(fig)
except ImportError: pass
print(json.dumps({'hard_null_threshold':hard,'counts':{n:{k:v['score']['n'] for k,v in z['groups'].items()} for n,z in reports.items()}}))
