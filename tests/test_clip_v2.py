import copy,math
import pytest
import torch
from soccergmr.clip import balanced_reduce,clip_auxiliary_loss
from soccergmr.config import validate_clip_v2
from soccergmr.data import soft_clip_labels


def test_balanced_video_and_missing_groups():
 z=torch.tensor([[2.,-1.,4.,100.],[-2.,1.,100.,100.]],requires_grad=True)
 y=torch.tensor([[1.,0.,0.,0.],[0.,0.,0.,0.]])
 m=torch.tensor([[1,1,1,0],[1,1,0,0]])
 c=validate_clip_v2({'method':'balanced'})
 loss,_=clip_auxiliary_loss(z,{'saliency_all_labels':y},m,c)
 b=torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')
 expected=(.5*b[0,0]+.5*b[0,1:3].mean()+b[1,:2].mean())/2
 assert torch.allclose(loss,expected)
 loss.backward();assert z.grad[0,3]==0 and not z.grad[1,2:].any()
 # A positive video containing no negative clips keeps unit sample weight.
 v=balanced_reduce(torch.tensor([[3.,5.]]),torch.ones(1,2),torch.ones(1,2))
 assert v==4


@pytest.mark.parametrize('method',['balanced','focal','dice','ranking','soft','native'])
def test_edges_all_null_all_positive_and_padding(method):
 for labels in [torch.zeros(2,7),torch.ones(2,7),torch.tensor([[1,0,1,0,0,0,0],[0,0,0,0,0,0,0.]])]:
  z=torch.randn(2,7,requires_grad=True);valid=torch.tensor([[1,1,1,1,1,0,0],[1,1,0,0,0,0,0]])
  c=validate_clip_v2({'method':method})
  loss,_=clip_auxiliary_loss(z,dict(saliency_all_labels=labels,clip_soft_labels=labels*.7),valid,c)
  loss.backward();assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
  assert not z.grad[~valid.bool()].any()


def test_soft_union_duration_and_boundary():
 # 3-second clips with overlap counted once.
 y=soft_clip_labels([[1,5],[3,7],[10,11]],12,4)
 assert torch.allclose(y,torch.tensor([2/3,1.,1/3,1/3]))
 assert not soft_clip_labels([],12,4).any()
 assert torch.equal(soft_clip_labels([[3,6]],12,4),torch.tensor([0.,1.,0.,0.]))


def test_focal_dice_and_topk_numerical_definitions():
 z=torch.tensor([[1.,-1.,2.,3.,-3.,0.,4.]],requires_grad=True);y=torch.tensor([[1.,0.,0.,0.,0.,0.,0.]])
 m=torch.ones_like(y);t={'saliency_all_labels':y}
 base,_=clip_auxiliary_loss(z,t,m,validate_clip_v2({'method':'balanced'}))
 rank,_=clip_auxiliary_loss(z,t,m,validate_clip_v2({'method':'ranking','topk':2}))
 expected=base+.2*torch.nn.functional.softplus(torch.tensor(.5-1+(4+3)/2))
 assert torch.allclose(rank,expected)
 focal,_=clip_auxiliary_loss(z,t,m,validate_clip_v2({'method':'focal','gamma':0}))
 assert torch.equal(base,focal)
 dice,_=clip_auxiliary_loss(z,t,m,validate_clip_v2({'method':'dice'}))
 expected=base+.2*(1-(2*(z.sigmoid()*y).sum()+1e-6)/(z.sigmoid().sum()+y.sum()+1e-6))
 assert torch.allclose(dice,expected)


@pytest.mark.parametrize('bad',[{'method':'unknown'},{'method':'balanced','topk':1.5},{'method':'balanced','gamma':float('nan')},{'method':'balanced','freeze_aux':1}])
def test_config_rejects_invalid(bad):
 with pytest.raises(ValueError):validate_clip_v2(bad)


def test_native_quality_keeps_bam_candidate_order():
    from soccergmr.predict import export
    out=dict(pred_logits=torch.tensor([[[9.,0.],[0.,9.]]]),pred_quality=torch.tensor([[[.1],[.9]]]),pred_spans=torch.tensor([[[.2,.1],[.8,.1]]]),pred_exist_logits=torch.zeros(1))
    result=export(out,[dict(qid=1,vid='a',duration=100)])[0]
    assert result['pred_relevant_windows'][0][0]>70
