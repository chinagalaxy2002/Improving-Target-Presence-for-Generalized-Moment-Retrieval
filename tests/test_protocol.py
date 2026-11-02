import copy
import math
import pytest
from soccergmr.selection import Selector,score
from soccergmr.evaluation import evaluate


def test_exact_ten_consecutive_epochs_ties_and_resume():
    selector=Selector('Val G1',10)
    assert selector.update(1,.5)['improved']
    for epoch in range(2,7):assert not selector.update(epoch,.5)['stop']
    selector=Selector.from_state(selector.state_dict())
    for epoch in range(7,11):assert not selector.update(epoch,.4)['stop']
    decision=selector.update(11,.5)
    assert decision['stop'] and decision['stale']==10 and decision['best_epoch']==1
    assert selector.update(12,.6)['stale']==0
    for value in (float('nan'),float('inf')):
        with pytest.raises(ValueError):selector.update(13,value)


def test_selection_uses_three_point_mean_not_max_or_map():
    first={'brief':{'mAP':99},'gmiou':{str(t):{'G-mIoU@1':v} for t,v in zip((.4,.5,.6),(.9,.1,.1))}}
    second={'brief':{'mAP':0},'gmiou':{str(t):{'G-mIoU@1':.4} for t in (.4,.5,.6)}}
    assert score('geq_full',second)>score('geq_full',first)
    assert score('gmr',first)>score('gmr',second)


def test_official_metrics_full_coverage_and_presence_gate():
    gt=[dict(qid=1,vid='a',duration=10,relevant_windows=[[2,4]]),dict(qid=2,vid='b',duration=10,relevant_windows=[])]
    pred=[dict(qid=1,pred_relevant_windows=[[2,4,.9]],pred_exist_score=.45),
          dict(qid=2,pred_relevant_windows=[[2,4,.9]],pred_exist_score=.2)]
    metrics=evaluate(pred,gt)
    assert metrics['brief']['mAP']==100
    assert metrics['gmiou']['0.4']['G-mIoU@1']>metrics['gmiou']['0.5']['G-mIoU@1']
    for bad in (pred[:1],pred+[pred[0]]):
        with pytest.raises(ValueError):evaluate(bad,gt)
    changed=copy.deepcopy(pred);changed[0]['pred_exist_score']=.99
    assert evaluate(changed,gt)['brief']['mAP']==metrics['brief']['mAP']
