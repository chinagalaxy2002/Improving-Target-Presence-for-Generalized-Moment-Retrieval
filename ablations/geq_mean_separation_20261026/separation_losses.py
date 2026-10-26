"""Fixed, attached GEQ class-mean constraints. No RNG or cross-batch state."""
import torch

def mean_constraint(logits,labels,setting):
    z=logits.reshape(-1);y=labels.reshape(-1).bool();s=z.sigmoid()
    pos=s[y].mean() if y.any() else None
    neg=s[~y].mean() if (~y).any() else None
    if pos is None or neg is None:return z.sum()*0.,pos,neg,False
    if setting=='GAP':loss=(.5-(pos-neg)).clamp_min(0).square()
    elif setting=='ENDPOINT':loss=.5*((.75-pos).clamp_min(0).square()+(neg-.25).clamp_min(0).square())
    else:raise ValueError(setting)
    return loss,pos,neg,True

def analytical_checks(setting):
    records=[]
    for values,labels,expected in [([.5,.5],[1,0],.25 if setting=='GAP' else .0625),([.8,.2],[1,0],0.),([.2,.3],[1,1],0.),([.7,.8],[0,0],0.)]:
        z=torch.logit(torch.tensor(values,dtype=torch.float64)).requires_grad_();y=torch.tensor(labels)
        loss,_,_,mixed=mean_constraint(z,y,setting);g=torch.autograd.grad(loss,z)[0]
        assert abs(loss.item()-expected)<1e-12 and torch.isfinite(g).all()
        if expected>0:assert (g[y.bool()]<0).all() and (g[~y.bool()]>0).all()
        else:assert not g.count_nonzero()
        records.append(dict(probabilities=values,labels=labels,loss=loss.item(),gradient=g.tolist(),mixed=mixed))
    return records
