"""LD-DETR GMR wrapper, preserving momentum-alignment state in native model."""
import torch
from torch import nn
from torch.nn import functional as F
from .loss import select


class LDModel(nn.Module):
    def __init__(self,native,dimension):
        super().__init__();self.native=native;self.features=None
        self.exist_head=nn.Sequential(nn.Linear(dimension,dimension),nn.ReLU(),nn.Linear(dimension,1))
        native.prediction_heads.video_moment_retrieval_prediction_head.class_embed.register_forward_pre_hook(self.capture)
    def capture(self,module,args):self.features=args[0][-1]
    def forward(self,epoch=None,batch_idx=0,train_loader_length=0,targets=None,**inputs):
        self.features=None
        try:
            out=self.native(**inputs,epoch_i=epoch or 0,batch_idx=batch_idx,train_loader_length=train_loader_length,targets=targets,is_training=self.training and targets is not None)
            if self.features is None:raise RuntimeError('Missing LD-DETR decoder features')
            out['pred_exist_logits']=self.exist_head(self.features.amax(1)).squeeze(-1)
            return out
        finally:self.features=None


class LDLoss(nn.Module):
    def __init__(self,native):super().__init__();self.native=native;self.weight_dict={**native.weight_dict,'loss_exist':1.}
    def forward(self,outputs,targets,epoch=0):
        logit=outputs['pred_exist_logits'].reshape(-1);batch=len(logit);pos=torch.where(targets['exist_label']>0)[0];losses={}
        for level,out in enumerate([outputs]+list(outputs.get('aux_outputs',[]))):
            suffix='' if level==0 else f'_{level-1}'
            matches=[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)) for _ in range(batch)]
            if len(pos):
                po=select({k:v for k,v in out.items() if k!='aux_outputs'},pos,batch);pt=select(targets,pos,batch);pm=self.native.matcher(po,pt)
                for i,match in zip(pos.tolist(),pm):matches[i]=match
                for k,v in self.native.loss_spans(po,pt,pm).items():losses[k+suffix]=v
            else:losses['loss_span'+suffix]=logit.sum()*0;losses['loss_giou'+suffix]=logit.sum()*0
            losses['loss_label'+suffix]=self.native.loss_labels(out,targets,matches,log=False)['loss_label']
        scores=outputs['saliency_scores'];mask=outputs['video_mask'].bool();fg=targets['saliency_all_labels'].bool() & mask
        logp=F.log_softmax((scores/.5).masked_fill(~mask,float('-inf')),dim=1)
        rank=-(logp.masked_fill(~fg,0).sum(1)/fg.sum(1).clamp_min(1)).mean()/12
        valid=targets['saliency_pair_mask'].bool()
        if valid.any():
            ix=torch.arange(batch,device=logit.device)[:,None];pair=(self.native.saliency_margin+scores[ix,targets['saliency_neg_labels']]-scores[ix,targets['saliency_pos_labels']]).clamp_min(0)[valid].mean()*2
        else:pair=scores.sum()*0
        losses.update(loss_saliency=rank+pair,loss_align=outputs['loss_align'],loss_sim=outputs['loss_sim'],loss_exist=F.binary_cross_entropy_with_logits(logit,targets['exist_label']))
        total=sum(v*self.weight_dict[k] for k,v in losses.items() if k in self.weight_dict)
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite LD-DETR loss')
        return total,losses
