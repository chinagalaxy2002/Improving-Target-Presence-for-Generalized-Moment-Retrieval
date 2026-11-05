import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/weight_space'))
import numpy as np
from core import temporal_iou,label,video_partition,direction,summarize

def test_iou_multiple_and_empty():
    assert temporal_iou([0,2],[[0,1],[0,2]])==1
    assert temporal_iou(None,[])==0

def test_groups_and_strict_threshold():
    p={'presence':True,'top1_iou':.5,'probability':.5,'candidate_score':.1}
    assert label(p,.5,.2)=='B'
    assert label(dict(p,probability=.51),.5,.2)=='A'
    assert label(dict(p,presence=False,candidate_score=.2),.5,.2)=='C'
    assert label(dict(p,presence=False,candidate_score=.1),.5,.2)=='D'

def test_direction_and_diagnostic():
    rows=[{'presence':True,'top1_iou':.6,'probability':.4,'candidate_score':.1,'logit':0}]*2+ [{'presence':False,'top1_iou':0,'probability':.2,'candidate_score':.9,'logit':-1}]*2
    h=np.array([[2.,0],[2.,0],[0.,0],[0.,0]])
    u,mu=direction(h,rows,.5,.5)
    np.testing.assert_array_equal(u,[1,0]);np.testing.assert_array_equal(mu,[0,0])
    assert summarize(rows,h,.5,.5)['confusion']=={'TP':0,'FN':2,'FP':0,'TN':2}
    assert video_partition('vid')==video_partition('vid')


def test_head_edit_anchors_hard_null_and_is_linear():
    from core import head_delta
    u=np.zeros(256);u[0]=1;mu=np.arange(256,dtype=float)
    dw,db=head_delta(u,mu,.3)
    assert abs(float(dw@mu)+db)<1e-12
    x=mu.copy();x[0]+=2
    assert abs(float(dw@x)+db-.6)<1e-12


def test_ffn_column_suppression_matches_causal_activation_intervention():
    from core import suppress_columns
    rng=np.random.default_rng(7);w=rng.normal(size=(4,6));u=np.array([1.,0,0,0]);cols=[1,4];x=rng.normal(size=6);alpha=.4
    edited=suppress_columns(w,u,cols,alpha)
    expected=w@x-alpha*u*((u@w[:,cols])@x[cols])
    np.testing.assert_allclose(edited@x,expected)
    np.testing.assert_array_equal(edited[:,[0,2,3,5]],w[:,[0,2,3,5]])


def test_random_joint_edit_norm_matches():
    from core import head_delta
    rng=np.random.default_rng(11);u=rng.normal(size=256);u/=np.linalg.norm(u);r=rng.normal(size=256);r/=np.linalg.norm(r);mu=rng.normal(size=256)
    dw,db=head_delta(u,mu,.2)
    scale=np.sqrt(1+(u@mu)**2)/np.sqrt(1+(r@mu)**2)
    rw,rb=head_delta(r,mu,.2*scale)
    np.testing.assert_allclose(np.linalg.norm(np.r_[dw,db]),np.linalg.norm(np.r_[rw,rb]))

if __name__ == '__main__':
    tests=[value for name,value in list(globals().items()) if name.startswith('test_') and callable(value)]
    for test in tests:test()
    print(f'{len(tests)} tests passed')
