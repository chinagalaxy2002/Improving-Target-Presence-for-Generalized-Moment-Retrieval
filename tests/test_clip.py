import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from soccergmr.clip import ClipHead, GMRWithClipHead, masked_clip_loss
from soccergmr.config import stage_config_identity


def test_dense_loss_per_video_padding_and_all_null_gradients():
    logits=torch.zeros(2,4,requires_grad=True)
    labels=torch.tensor([[0.,1.,0.,1.],[0.,0.,0.,0.]])
    valid=torch.tensor([[1.,1.,1.,1.],[1.,0.,0.,0.]])
    loss=masked_clip_loss(logits,labels,valid)
    assert float(loss)==pytest.approx(0.69314718)
    loss.backward()
    assert logits.grad[1,0]==pytest.approx(.25)
    assert not logits.grad[1,1:].any()
    z=torch.randn(2,4,requires_grad=True)
    masked_clip_loss(z,torch.zeros_like(z),valid).backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    z=torch.randn(1,3,requires_grad=True)
    masked_clip_loss(z,torch.zeros_like(z),torch.zeros_like(z)).backward()
    assert not z.grad.any()


def test_text_mask_and_query_dependence():
    torch.manual_seed(9)
    head=ClipHead(12,8,16)
    v=torch.randn(2,5,12);q=torch.randn(2,3,8)
    vm=torch.ones(2,5);tm=torch.tensor([[1,1,0],[1,1,1]])
    first=head(v,q,vm,tm)
    changed=q.clone();changed[0,2]+=1000
    assert torch.equal(first,head(v,changed,vm,tm))
    changed=q.clone();changed[0,0]+=10
    assert not torch.equal(first[0],head(v,changed,vm,tm)[0])


def test_stage_identity_sharing_only_stage_two_clip_weight():
    a={'identity':'A','clip':{'gmr_weight':0.,'geq_weight':0.},'protocol':{'seed':2023}}
    b=copy.deepcopy(a);b['identity']='B';b['clip']['geq_weight']=.3
    assert stage_config_identity(a,'gmr')==stage_config_identity(b,'gmr')
    assert stage_config_identity(a,'geq_full')!=stage_config_identity(b,'geq_full')
    b['clip']['gmr_weight']=.3
    assert stage_config_identity(a,'gmr')!=stage_config_identity(b,'gmr')
    b=copy.deepcopy(a);b['protocol']['seed']=7
    assert stage_config_identity(a,'gmr')!=stage_config_identity(b,'gmr')


def test_encoder_offset_multiple_passes_and_clip_gradient():
    class Layer(nn.Module):
        def __init__(self):
            super().__init__();self.self_attn=SimpleNamespace(embed_dim=8);self.proj=nn.Linear(8,8)
        def forward(self,x):return self.proj(x)
    class Base(nn.Module):
        def __init__(self):
            super().__init__();self.input_txt_proj=nn.Linear(8,8)
            self.transformer=nn.Module();self.transformer.encoder=nn.Module()
            self.transformer.encoder.layers=nn.ModuleList([Layer(),Layer()])
        def forward(self,src_txt,src_txt_mask,src_vid,src_vid_mask):
            text=self.input_txt_proj(src_txt)
            x=torch.cat([src_vid[:,:1]*0,src_vid,text],1).transpose(0,1)
            for _ in range(2):
                for layer in self.transformer.encoder.layers:x=layer(x)
            return {'tokens':x}
    base=Base();spec=SimpleNamespace(name='moment_detr',encoder_offset=1,encoder_passes=2,extra_encoder_input='')
    model=GMRWithClipHead(base,spec)
    x=dict(src_txt=torch.randn(2,3,8),src_txt_mask=torch.ones(2,3),src_vid=torch.randn(2,5,8),src_vid_mask=torch.ones(2,5))
    out=model(**x)
    assert out['clip_logits'].shape==(2,5)
    masked_clip_loss(out['clip_logits'],torch.zeros(2,5),x['src_vid_mask']).backward()
    assert base.transformer.encoder.layers[0].proj.weight.grad.abs().sum()>0
    model.close()
