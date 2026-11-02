"""Strict official evaluation: require exactly one prediction for every GT qid."""
import numpy as np
if not hasattr(np,'trapz'):np.trapz=np.trapezoid
from .official.eval_main import evaluate_gmr
from .official.metrics import prepare_submission_for_gmiou,compute_G_mIoU


def evaluate(predictions,ground_truth):
    qids=[p['qid'] for p in predictions]
    truth=[g['qid'] for g in ground_truth]
    if len(qids)!=len(set(qids)) or len(truth)!=len(set(truth)) or set(qids)!=set(truth):
        raise ValueError('Prediction/GT qids must match exactly with no duplicates or drops')
    result=evaluate_gmr(predictions,ground_truth,cls_thresholds=(.4,.5,.6),
                        gmiou_cls_threshold=.4,map_num_workers=1,verbose=False)
    result['gmiou']={}
    for threshold in (.4,.5,.6):
        gated,_=prepare_submission_for_gmiou(predictions,cls_threshold=threshold,max_pred_windows=10)
        result['gmiou'][str(threshold)]=compute_G_mIoU(gated,ground_truth,k_list=(1,3,5))
    return result
