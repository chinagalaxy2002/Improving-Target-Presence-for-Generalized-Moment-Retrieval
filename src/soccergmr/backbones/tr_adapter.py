"""TR-DETR native losses with real-null GMR supervision."""
import torch
from torch import nn
from torch.nn import functional as F
from .loss import select
from .uvcom_adapter import UVModel


class TRModel(UVModel):
    def forward(self, **inputs):
        out=super().forward(**inputs)
        out['text_mask']=inputs['src_txt_mask']
        return out


class TRLoss(nn.Module):
    def __init__(self,native):
        super().__init__();self.native=native
        self.weight_dict={**native.weight_dict,'loss_exist':1.,'loss_vtc':.3,'loss_ctc':.5}

    def forward(self,outputs,targets,epoch=0):
        logit=outputs['pred_exist_logits'].reshape(-1);batch=len(logit);positive=torch.where(targets['exist_label']>0)[0]
        losses={};levels=[outputs]+list(outputs.get('aux_outputs',[]))
        for level,out in enumerate(levels):
            suffix='' if level==0 else f'_{level-1}'
            indices=[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)) for _ in range(batch)]
            if len(positive):
                po=select({k:v for k,v in out.items() if k!='aux_outputs'},positive,batch);pt=select(targets,positive,batch);matches=self.native.matcher(po,pt)
                for i,match in zip(positive.tolist(),matches):indices[i]=match
                for key,value in self.native.loss_spans(po,pt,matches).items():losses[key+suffix]=value
            else:
                losses['loss_span'+suffix]=logit.sum()*0;losses['loss_giou'+suffix]=logit.sum()*0
            losses['loss_label'+suffix]=self.native.loss_labels(out,targets,indices,log=False)['loss_label']
        scores=outputs['saliency_scores'];mask=outputs['video_mask'].bool();fg=targets['saliency_all_labels'].bool() & mask
        logp=F.log_softmax((scores/.5).masked_fill(~mask,float('-inf')),dim=1)
        ranking=-(logp.masked_fill(~fg,0).sum(1)/fg.sum(1).clamp_min(1)).mean()/12
        valid=targets['saliency_pair_mask'].bool()
        if valid.any():
            ix=torch.arange(batch,device=logit.device)[:,None]
            pair=(self.native.saliency_margin+scores[ix,targets['saliency_neg_labels']]-scores[ix,targets['saliency_pos_labels']]).clamp_min(0)[valid].mean()*2
        else:pair=scores.sum()*0
        losses['loss_saliency']=ranking+pair
        txt=F.normalize(outputs['src_txt_ed'],dim=-1);vid=F.normalize(outputs['src_vid_ed'],dim=-1);tm=outputs['text_mask'].unsqueeze(-1)
        txt=(txt*tm).sum(1)/tm.sum(1).clamp_min(1)
        frame=(vid*txt[:,None,:]).sum(-1)/.07
        losses['loss_ctc']=(F.binary_cross_entropy_with_logits(frame,targets['saliency_all_labels'],reduction='none')*mask).sum()/mask.sum().clamp_min(1)
        if len(positive):
            tx=F.normalize(outputs['src_txt_cls_ed'][positive],dim=-1);vi=F.normalize(outputs['src_vid_cls_ed'][positive],dim=-1)
            sim=tx@vi.T/.07;groups=targets['video_group'][positive];duplicate=groups[:,None].eq(groups[None,:]);duplicate.fill_diagonal_(False);sim=sim.masked_fill(duplicate,float('-inf'));labels=torch.arange(len(sim),device=logit.device)
            losses['loss_vtc']=(F.cross_entropy(sim,labels)+F.cross_entropy(sim.T,labels))/2
        else:losses['loss_vtc']=logit.sum()*0
        losses['loss_exist']=F.binary_cross_entropy_with_logits(logit,targets['exist_label'])
        total=sum(v*self.weight_dict[k] for k,v in losses.items() if k in self.weight_dict)
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite TR-DETR loss')
        return total,losses
