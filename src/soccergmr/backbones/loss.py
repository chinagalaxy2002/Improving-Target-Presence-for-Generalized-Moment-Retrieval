"""Keep native localization losses and supervise real null queries with BCE."""
import torch
from torch import nn
from torch.nn import functional as F


def select(value, indices, batch, key=''):
    if torch.is_tensor(value):
        return value[indices] if value.ndim and value.shape[0]==batch and key!='global_rep_tokens' else value
    if isinstance(value,dict):return {k:select(v,indices,batch,k) for k,v in value.items()}
    if isinstance(value,list):
        if key in ('span_labels','mask_labels','pseudo_event_spans'):
            return [value[i] for i in indices.tolist()]
        return [select(v,indices,batch) for v in value]
    return value


def stable_cg_criterion(criterion):
    """Evaluate native attention BCE only over valid frames, never padding."""
    base=type(criterion)
    class ValidFrameCriterion(base):
        def loss_labels(self,outputs,targets,indices,log=True):
            return super().loss_labels(outputs,targets,indices,log=False)
        def loss_saliency(self,outputs,targets,indices,log=True):
            pieces=[]
            batch=outputs['saliency_scores'].shape[0]
            for j in range(batch):
                idx=torch.tensor([j],device=outputs['saliency_scores'].device)
                one=select(outputs,idx,batch)
                target=select(targets,idx,batch)
                length=int(one['video_mask'].sum())
                for key in ('saliency_scores','video_mask','t2vattnvalues'):
                    one[key]=one[key][:,:length]
                for key in ('saliency_all_labels','relevant_clips'):
                    target[key]=target[key][:,:length]
                pieces.append(super().loss_saliency(one,target,None)['loss_saliency'])
            return {'loss_saliency':torch.stack(pieces).mean()}
    criterion.__class__=ValidFrameCriterion
    return criterion


class GMRLoss(nn.Module):
    def __init__(self,native,name):
        super().__init__()
        self.native,self.name=native,name
        self.weight_dict={**native.weight_dict,'loss_exist':1.0}
    def forward(self,outputs,targets,epoch=0):
        if self.name=='sim_detr':
            losses=self.native(outputs,targets)
        else:
            logit=outputs.get('pred_exist_logits',outputs.get('pred_exist_logit')).reshape(-1)
            batch=len(logit)
            indices=torch.where(targets['exist_label']>0)[0]
            if len(indices):
                positive_outputs=select(outputs,indices,batch)
                positive_targets=select(targets,indices,batch)
                if self.name=='eatr':
                    # Historical SoccerGMR EaTR profile has no saliency pair supervision.
                    for key in ('saliency_pos_labels','saliency_neg_labels'):
                        positive_targets.pop(key,None)
                    losses=self.native(positive_outputs,positive_targets,epoch)
                else:losses=self.native(positive_outputs,positive_targets)
            else:
                losses={k:logit.sum()*0 for k in self.native.weight_dict}
            losses['loss_exist']=F.binary_cross_entropy_with_logits(logit,targets['exist_label'])
        total=sum(value*self.weight_dict[key] for key,value in losses.items() if key in self.weight_dict)
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite native losses: '+str({k:float(v.detach()) if torch.is_tensor(v) else v for k,v in losses.items()}))
        return total,losses
