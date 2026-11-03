"""TaskWeave GMR adaptation with learned original task weighting."""
import torch
from torch import nn
from torch.nn import functional as F
from .bam_adapter import BAMLoss
from .loss import select


class TaskModel(nn.Module):
    def __init__(self,native,dimension):
        super().__init__();self.native=native
        self.exist_head=nn.Sequential(nn.Linear(dimension,dimension),nn.ReLU(),nn.Linear(dimension,1))
        self.features=None;self.native.class_embed.register_forward_pre_hook(self.capture)
    def capture(self,module,args):self.features=args[0][-1]
    def forward(self,epoch=None,**inputs):
        self.features=None
        try:
            output,(hd,mr)=self.native(epoch_i=epoch,**inputs)
            if self.features is None:raise RuntimeError('Missing TaskWeave decoder features')
            output['pred_exist_logits']=self.exist_head(self.features.amax(1)).squeeze(-1)
            output['hd_log_var']=hd;output['mr_log_var']=mr
            return output
        finally:self.features=None


class TaskLoss(BAMLoss):
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
        losses['loss_saliency']=self.saliency(outputs,targets)
        losses['loss_exist']=F.binary_cross_entropy_with_logits(logit,targets['exist_label'])
        mr=sum(v for k,v in losses.items() if k.startswith(('loss_span','loss_giou','loss_label')) and k in self.native.weight_dict)
        hd=losses['loss_saliency'];h=outputs['hd_log_var'];m=outputs['mr_log_var']
        total=(2*hd*torch.exp(-h)+mr*torch.exp(-m)+h+m).sum()+losses['loss_exist']
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite TaskWeave loss')
        return total,losses
