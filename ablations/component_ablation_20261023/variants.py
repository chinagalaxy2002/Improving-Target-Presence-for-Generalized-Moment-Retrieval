"""Matched component switches; all parameters materialized before switches."""
import importlib.util
from pathlib import Path
import torch
from torch import nn
REF=Path(__file__).resolve().parent/'code/model.py'
spec=importlib.util.spec_from_file_location('ablation_reference',REF)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)

ROWS={1:'Coupled GMR Adapter',2:'+ Decoupled Global Event Query',3:'+ LoRA',4:'+ Hard-RoI Refiner (without saliency)',5:'+ Gaussian-RoI Refiner (without saliency)',6:'Full DeRoI-GMR (minusFusion)',7:'Full + candidate presence fusion control'}

class IdentityRefiner(nn.Module):
    def forward(self,moments,text,mask,spans,*args,**kwargs):
        return spans,torch.zeros_like(spans)

class NativeVariant(nn.Module):
    def __init__(self,inner,row):
        super().__init__();self.inner=inner;self.row=row;self.cache={}
        head=next(getattr(inner,n) for n in ['event_query','global_query','global_event_query'] if hasattr(inner,n))
        # Exact reference fusion modules, identically materialized for every row.
        self.fusion=reference.MomentConditionedVerifier(len(head.video_reads),dict(use_moment_conditioning=True,use_boundary_refine=False))
        self.fusion.requires_grad_(False)
        if row==7:
            for name in ['moment_cross_attn','moment_match_mlp','local_head','fusion_gate']:
                getattr(self.fusion,name).requires_grad_(True)
        inner.base.register_forward_hook(self._base)
        head.text_read.register_forward_hook(self._text)
        inner.extractor.register_forward_hook(self._moments)
        for name,p in inner.named_parameters():
            active=(row>1 and not name.startswith('base.')) or (row>=3 and '.lora_' in name)
            if name.startswith(('extractor.','refiner.')):active=row>=4
            if '.sal_proj.' in name:active=row>=6
            p.requires_grad_(active)
        for m in inner.modules():
            if isinstance(m,reference.LoRALinear) or (hasattr(m,'lora_A') and hasattr(m,'scaling')):
                if row<3:m.scaling=0.
        inner.extractor.pooling_type='uniform_box' if row==4 else 'gaussian'
        if row<4:inner.refiner=IdentityRefiner()
        elif row<6:inner.refiner.sal_proj.register_forward_hook(lambda m,a,o:torch.zeros_like(o))

    def _base(self,m,a,o):self.cache['native_logit']=o.get('pred_exist_logits',o.get('pred_exist_logit'))
    def _text(self,m,a,o):self.cache['text_state']=o[0][:,0]
    def _moments(self,m,a,o):self.cache['moments']=o[0];self.cache['conf']=o[1]
    def train(self,mode=True):
        super().train(mode);self.inner.base.eval();return self
    def forward(self,**inputs):
        self.cache={};out=self.inner(**inputs)
        if self.row==1:
            out['pred_exist_logits']=self.cache['native_logit'].reshape(-1)
        if self.row<4:
            c,w=out['pred_spans'].unbind(-1);d=inputs['durations'][:,None,None]
            xy=torch.minimum((torch.stack([c-w/2,c+w/2],-1)*d).clamp_min(0),d)
            out['candidate_seconds']=torch.cat([xy,out['pred_logits'].softmax(-1)[...,:1]],-1)
            out.pop('refined_spans',None)
        if self.row==7:
            q=self.cache['text_state'];moments=self.cache['moments'];conf=self.cache['conf'];f=self.fusion
            local,_=f.moment_cross_attn(q[:,None],moments,moments)
            local=f.local_head(local[:,0]).squeeze(-1)
            qr=q[:,None].expand_as(moments)
            match=f.moment_match_mlp(torch.cat([qr,moments,qr*moments,conf[...,None]],-1)).squeeze(-1)
            k=min(3,conf.shape[1]);g=out['pred_exist_logits']
            gate=torch.stack([g,local,match.max(1).values,match[:,:k].mean(1),conf.max(1).values,conf[:,:k].mean(1)],1)
            out['pred_exist_logits']=g+f.fusion_gate(gate).squeeze(-1)
        out['pred_exist_score']=out['pred_exist_logits'].sigmoid()
        return out
