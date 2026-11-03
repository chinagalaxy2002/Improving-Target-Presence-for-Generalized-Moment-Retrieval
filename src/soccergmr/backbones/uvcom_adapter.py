"""UVCOM adaptation retaining global/local similarity and saliency losses."""
import torch
from torch import nn
from torch.nn import functional as F
from .bam_adapter import BAMLoss
from .loss import select


class UVModel(nn.Module):
    def __init__(self, native, dimension):
        super().__init__()
        self.native = native
        self.exist_head = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, 1))
        self._decoder_features = None
        self.native.class_embed.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        self._decoder_features = args[0][-1]

    def forward(self, **inputs):
        self._decoder_features = None
        try:
            out = self.native(**inputs)
            if self._decoder_features is None:
                raise RuntimeError('Missing UVCOM final decoder representation')
            out['pred_exist_logits'] = self.exist_head(self._decoder_features.amax(dim=1)).squeeze(-1)
            return out
        finally:
            self._decoder_features = None


class UVLoss(BAMLoss):
    def similarity(self, outputs, targets):
        sim = outputs['sim']['VLA_Sim']
        clip, mask = outputs['sim']['CTA_Sim']
        positive = torch.where(targets['exist_label'] > 0)[0]
        if len(positive):
            inter = sim[positive][:, positive]
            groups = targets['video_group'][positive]
            duplicate = groups[:,None].eq(groups[None,:])
            duplicate.fill_diagonal_(False)
            inter = inter.masked_fill(duplicate, float('-inf'))
            label = torch.arange(len(positive), device=sim.device)
            inter_loss = (F.cross_entropy(inter, label) + F.cross_entropy(inter.T, label)) / 2
        else:
            inter_loss = sim.sum() * 0
        # All annotated windows contribute, with padding excluded.
        mask = mask.bool()
        clip_logp = F.log_softmax(clip.masked_fill(~mask, float('-inf')),dim=-1)
        gt = targets['saliency_all_labels'].bool() & mask
        intra = -(clip_logp.masked_fill(~gt, 0)).mean()
        return .5 * (inter_loss + intra)

    def forward(self, outputs, targets, epoch=0):
        logit=outputs['pred_exist_logits'].reshape(-1)
        batch=len(logit)
        positive=torch.where(targets['exist_label']>0)[0]
        losses={}
        levels=[outputs]+list(outputs.get('aux_outputs',[]))
        for level,out in enumerate(levels):
            suffix='' if level==0 else f'_{level-1}'
            matches=[(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)) for _ in range(batch)]
            if len(positive):
                pos_out=select({k:v for k,v in out.items() if k not in ('aux_outputs','sim')},positive,batch)
                pos_targets=select(targets,positive,batch)
                pos_matches=self.native.matcher(pos_out,pos_targets)
                for index,match in zip(positive.tolist(),pos_matches):matches[index]=match
                for key,value in self.native.loss_spans(pos_out,pos_targets,pos_matches).items():losses[key+suffix]=value
            else:
                losses['loss_span'+suffix]=logit.sum()*0
                losses['loss_giou'+suffix]=logit.sum()*0
            losses['loss_label'+suffix]=self.native.loss_labels(out,targets,matches,log=False)['loss_label']
        losses['loss_saliency']=self.saliency(outputs,targets)
        losses['loss_sim']=self.similarity(outputs,targets)
        losses['loss_exist']=F.binary_cross_entropy_with_logits(logit,targets['exist_label'])
        total=sum(value*self.weight_dict[key] for key,value in losses.items() if key in self.weight_dict)
        if not torch.isfinite(total):raise FloatingPointError('Nonfinite UVCOM loss')
        return total,losses
