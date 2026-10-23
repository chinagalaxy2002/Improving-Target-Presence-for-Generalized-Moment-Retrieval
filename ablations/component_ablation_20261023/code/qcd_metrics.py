"""WorldCup-only discrimination and recall at fixed empirical specificity."""
import numpy as np
from sklearn.metrics import roc_curve
from evaluation import Evaluation
from qcd import worldcup


def worldcup_report(pre, post, gt):
    ev = Evaluation(pre, post, gt)
    wc = np.array([worldcup(r) for r in gt])
    single, multi = wc & (ev.count == 1), wc & (ev.count > 1)
    result = {name: ev.metrics(mask=mask) for name,mask in
              [('all',wc),('single',single),('multi',multi)] if mask.any()}
    # ROC constraint uses exact observed thresholds; never interpolates through ties.
    # These within-split values describe ranking, and are NOT deployed thresholds.
    fpr,tpr,thresholds = roc_curve(ev.y[wc], ev.logit[wc], drop_intermediate=False)
    fixed = {}
    for specificity in (.90,.95,.975):
        valid = np.flatnonzero(fpr <= (1-specificity)+1e-12)
        index = valid[np.argmax(tpr[valid])]
        gate = ev.logit >= thresholds[index]
        fixed[str(specificity)] = dict(target_N_acc=100*specificity,
            achieved_N_acc=100*(1-fpr[index]),positive_recall=100*tpr[index],
            single_recall=100*float(gate[single].mean()),multi_recall=100*float(gate[multi].mean()))
    result['fixed_N_acc_ROC_diagnostic'] = fixed
    return result
