"""Pure, gradient-free diagnostics shared by the FlashVTG E1 experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path) as f: return [json.loads(x) for x in f if x.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def video_partition(video, seed=2026):
    h = hashlib.sha256(f'{seed}:{video}'.encode()).digest()
    return 'discovery' if int.from_bytes(h[:8], 'big') % 10 < 6 else 'development'


def temporal_iou(window, targets):
    if window is None or not targets: return 0.0
    a,b = map(float, window[:2]); best = 0.0
    for x,y in targets:
        intersection = max(0., min(b,y)-max(a,x))
        union = max(0.,b-a)+max(0.,y-x)-intersection
        if union > 0: best = max(best,intersection/union)
    return best


def label(row, tau, hard_threshold):
    if row['presence']:
        if row['top1_iou'] < .5: return 'other_positive'
        return 'A' if row['probability'] > tau else 'B'
    return 'C' if row['candidate_score'] >= hard_threshold else 'D'


def distribution(x):
    x = np.asarray(x, dtype=float)
    if not len(x): return {'n': 0}
    return {'n':len(x),'mean':float(x.mean()),'std':float(x.std()),'quantiles':np.quantile(x,[0,.1,.25,.5,.75,.9,1]).tolist()}


def summarize(rows, activations, tau, hard_threshold):
    labels=[label(r,tau,hard_threshold) for r in rows]
    out={}
    for group in ('A','B','C','D','other_positive'):
        ids=[i for i,v in enumerate(labels) if v==group]
        scores=[rows[i]['probability'] for i in ids]
        hs=activations[ids]
        out[group]={'score':distribution(scores),'logit':distribution([rows[i]['logit'] for i in ids]),
                    'candidate_score':distribution([rows[i]['candidate_score'] for i in ids]),
                    'near_threshold_0.4_0.6':sum(.4 <= x <= .6 for x in scores),
                    'activation_mean':hs.mean(0).tolist() if len(ids) else None,
                    'activation_std':hs.std(0).tolist() if len(ids) else None,
                    'activation_l2':distribution(np.linalg.norm(hs,axis=1)) if len(ids) else {'n':0}}
    y=np.array([r['presence'] for r in rows],bool); p=np.array([r['probability']>tau for r in rows],bool)
    return {'groups':out,'confusion':{'TP':int((y&p).sum()),'FN':int((y&~p).sum()),'FP':int((~y&p).sum()),'TN':int((~y&~p).sum())}}


def direction(h, rows, tau, hard_threshold):
    b=np.array([label(r,tau,hard_threshold)=='B' for r in rows]); c=np.array([label(r,tau,hard_threshold)=='C' for r in rows])
    if b.sum()<2 or c.sum()<2: raise ValueError(f'Insufficient B/C discovery examples: {b.sum()}/{c.sum()}')
    d=h[b].mean(0)-h[c].mean(0); norm=np.linalg.norm(d)
    if norm <= 1e-12: raise ValueError('Degenerate direction')
    return d/norm,h[c].mean(0)


def head_delta(u,mu,alpha):
    u=np.asarray(u,dtype=np.float64);mu=np.asarray(mu,dtype=np.float64)
    if u.shape!=(256,) or mu.shape!=(256,) or not np.isfinite(u).all() or not np.isfinite(mu).all():raise ValueError('invalid direction')
    return alpha*u, -alpha*float(u@mu)


def suppress_columns(weight, direction, columns, strength):
    """Project chosen output columns away from a unit direction."""
    weight=np.asarray(weight);direction=np.asarray(direction)
    cols=np.asarray(columns,dtype=int)
    if weight.ndim!=2 or weight.shape[0]!=len(direction): raise ValueError('shape mismatch')
    if not np.isclose(np.linalg.norm(direction),1,atol=1e-5): raise ValueError('direction must be unit')
    if not len(cols) or len(set(cols.tolist()))!=len(cols) or (cols<0).any() or (cols>=weight.shape[1]).any(): raise ValueError('invalid columns')
    result=weight.copy();selected=weight[:,cols]
    result[:,cols]=selected-strength*np.outer(direction,direction@selected)
    return result
