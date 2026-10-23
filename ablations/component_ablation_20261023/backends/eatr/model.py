"""Native EaTR with independent Global Event Query and Gaussian RoI refinement."""
import torch
from torch import nn
from torch.nn import functional as F
from reference_components import EvidenceRead, CandidateMomentExtractor, TemporalBoundaryRefiner, LoRALinear, compute_boundary_refine_loss

class GlobalEventQueryHead(nn.Module):
    """Exact global path of the reference MomentConditionedVerifier, without local heads."""
    def __init__(self, depth):
        super().__init__()
        self.token=nn.Parameter(torch.randn(1,1,256)*0.02)
        self.text_read=EvidenceRead(256)
        self.video_reads=nn.ModuleList([EvidenceRead(256) for _ in range(depth)])
        self.global_head=nn.Linear(256,1)
    def forward(self,text,text_valid,video,video_valid):
        state,_=self.text_read(self.token.expand(text.shape[0],-1,-1),text,text_valid)
        assert len(video)==len(self.video_reads)
        for reader,memory in zip(self.video_reads,video):state,_=reader(state,memory,video_valid)
        return self.global_head(state[:,0]).squeeze(-1)

class S4EaTR(nn.Module):
    def __init__(self,base,config):
        super().__init__();self.base=base;self.config=dict(config)
        assert config['presence_source']=='global_refine' and config['pooling_type']=='gaussian'
        assert config['use_lora'] and config['lora_rank']==8 and config['multi_weight']==0 and not config['hard_negative']
        assert not any(config.get(k,False) for k in ('augmentation','qcd','query_swap','background_replacement','synthetic_negatives','synthetic_samples'))
        assert base.transformer.d_model==256
        layers=base.transformer.encoder.layers
        self.global_query=GlobalEventQueryHead(len(layers))
        self.extractor=CandidateMomentExtractor(256,config['top_k'],pooling_type='gaussian')
        self.refiner=TemporalBoundaryRefiner(256,config['refine_max_delta'])
        for layer in layers:
            layer.linear1=LoRALinear(layer.linear1,rank=8);layer.linear2=LoRALinear(layer.linear2,rank=8)
        for name,param in self.named_parameters():param.requires_grad_(not name.startswith('base.') or '.lora_' in name)
        self._context=None
        self._handles=[base.input_txt_proj.register_forward_hook(self._text),base.transformer.register_forward_hook(self._transformer)]
        self._handles.extend(layer.register_forward_hook(self._video) for layer in layers)
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def _text(self,module,args,result):
        if self._context is not None:self._context['text']=result
    def _video(self,module,args,result):
        if self._context is not None:self._context['video'].append(result.transpose(0,1)[:,:self._video_length])
    def _transformer(self,module,args,result):
        if self._context is not None:self._context['final']=result[1][:,:self._video_length]
    def forward(self,*,durations,candidate_override=None,**inputs):
        assert self._context is None,'Reentrant calls unsupported'
        assert torch.all((durations>0)&(durations<=150))
        tv=inputs['src_txt_mask'].bool();vv=inputs['src_vid_mask'].bool()
        assert tv.any(1).all() and vv.any(1).all()
        self._video_length=vv.shape[1];context={'video':[]};self._context=context
        try:
            out=self.base(**inputs)
            center,width=out['pred_spans'].unbind(-1)
            xy=torch.stack([center-width/2,center+width/2],-1)*durations[:,None,None]
            xy=torch.minimum(xy.clamp_min(0),durations[:,None,None])
            candidates=torch.cat([xy,out['pred_logits'].softmax(-1)[...,:1]],-1)
            indices=candidates[...,2].topk(min(self.config['top_k'],candidates.shape[1]),dim=1).indices
            selected=candidates.gather(1,indices[...,None].expand(-1,-1,3))
            if candidate_override is not None:
                assert candidate_override.shape==selected.shape
                selected=candidate_override
            # Presence never reads candidate coordinates, scores, RoI tokens or refined proposals.
            logit=self.global_query(context['text'],tv,context['video'],vv)
            tokens,_,_=self.extractor(context['final'],vv,selected)
            refined,deltas=self.refiner(tokens,context['text'],tv,selected,out['saliency_scores'].masked_fill(~vv,0))
            end=torch.minimum(refined[...,1],durations[:,None])
            start=torch.minimum(refined[...,0],(end-1).clamp_min(0))
            refined=torch.stack([start,end,refined[...,2]],-1)
            merged=candidates.scatter(1,indices[...,None].expand(-1,-1,3),refined)
            out.update(pred_exist_logits=logit,pred_exist_score=logit.sigmoid(),global_logit=logit,
                       refined_spans=refined,candidate_seconds=merged,selected_candidates=selected,refine_deltas=deltas)
            return out
        finally:self._context=None

def loss_for(output,metadata,config):
    labels=output['pred_exist_logits'].new_tensor([bool(d['relevant_windows']) for d in metadata])
    presence=F.binary_cross_entropy_with_logits(output['pred_exist_logits'],labels)
    refine=compute_boundary_refine_loss(output['refined_spans'],metadata,labels.device,config)
    return presence+config['refine_weight']*refine,{'loss_exist':presence,'loss_refine':refine}
