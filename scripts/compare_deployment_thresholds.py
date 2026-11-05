#!/usr/bin/env python3
"""Compare gain magnitudes under three fixed deployment threshold objectives.

All thresholds are selected on Standard Val before any Test is read. Every
model gets its own threshold under the same objective. The 101-point grid is
primary, matching the existing analysis; exact Val score breakpoints are a
separately reported sensitivity analysis. No Test-based threshold selection.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_presence_thresholds import (
    NAMES, METHODS, THRESHOLDS, arrays, dump, read_rows, sha, values, write_csv,
)

RULES = ('max_gmiou1', 'retain_positive_95', 'null_fpr_05')
SEARCHES = ('grid_101', 'val_breakpoints')


def choose(scores, utility, y, rule, search):
    candidates = THRESHOLDS if search == 'grid_101' else np.unique(np.r_[0., scores, 1.])
    accepted = scores[:, None] > candidates[None, :]
    tpr = accepted[y].mean(0)
    fpr = accepted[~y].mean(0)
    g1 = np.where(accepted, utility[:, None], (~y)[:, None]).mean(0)
    if rule == 'max_gmiou1':
        tied = np.flatnonzero(np.isclose(g1, g1.max(), atol=1e-12, rtol=0))
        index = min(tied, key=lambda i: (abs(candidates[i]-.5), candidates[i]))
    elif rule == 'retain_positive_95':
        feasible = np.flatnonzero(tpr >= .95-1e-12)
        if not len(feasible):
            raise ValueError('95% positive retention is infeasible on Val')
        index = feasible[-1]  # Largest threshold: minimizes FPR subject to retention.
    elif rule == 'null_fpr_05':
        feasible = np.flatnonzero(fpr <= .05+1e-12)
        index = feasible[0]  # Smallest threshold: maximizes localization recall.
    else:
        raise ValueError(rule)
    return dict(threshold=float(candidates[index]), val_TPR=100*float(tpr[index]),
                val_FPR=100*float(fpr[index]), val_G1=100*float(g1[index]),
                candidate_thresholds=len(candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=Path('results/presence_thresholds/INPUTS.local.json'))
    parser.add_argument('--val', type=Path, default=Path('results/presence_thresholds/val_predictions_11'))
    parser.add_argument('--tests', type=Path, default=Path('results/test_thresholds'))
    parser.add_argument('--output', type=Path, default=Path('results/deployment_thresholds'))
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(args.inputs.read_text())
    protocol = dict(rule_definitions={
        'max_gmiou1':'Maximize Val G-mIoU@1; ties within 1e-12: closest to .5, then smaller.',
        'retain_positive_95':'Largest Val threshold with positive retention >=95%; minimizes FPR.',
        'null_fpr_05':'Smallest Val threshold with null FPR <=5%; maximizes positive and successful-localization recall.'},
        primary_search='grid_101', sensitivity_search='all Val score breakpoints plus 0 and 1',
        gate='pred_exist_score > threshold', checkpoint_selection='Existing fixed bests; no epoch reselection',
        shared_objective_independent_thresholds=True, top_k=[1,3],
        relative_gain='(GEQ - GMR)/GMR * 100; also report percentage-point differences',
        summary='Equal-backbone mean scores; relative gain of means and mean per-backbone relative gain are both saved.',
        significance_tests=False, test_threshold_selection=False, new_training=False)
    dump(out/'PROTOCOL.json', protocol)
    val_gt = read_rows(inputs['val']['labels'])
    yval = np.array([bool(g['relevant_windows']) for g in val_gt])
    selected, selections, audit = {}, [], []
    for name in NAMES:
        selected[name] = {}
        for model in METHODS:
            path = args.val/name/model
            done = json.loads((path/'complete.json').read_text())
            assert done['checkpoint_sha256'] == inputs['backbones'][name][model]['sha256']
            assert done['identity']['label_sha256'] == sha(inputs['val']['labels'])
            assert done['predictions_sha256'] == sha(path/'predictions.jsonl.gz')
            score, utility = arrays(val_gt,read_rows(path/'predictions.jsonl.gz'))
            selected[name][model] = {}
            for search in SEARCHES:
                selected[name][model][search] = {}
                for rule in RULES:
                    result = choose(score,utility[1],yval,rule,search)
                    selected[name][model][search][rule] = result
                    selections.append(dict(backbone=name,model=model,search=search,rule=rule,**result))
            audit.append(dict(backbone=name,model=model,val_prediction_sha256=done['predictions_sha256'],
                              checkpoint_sha256=done['checkpoint_sha256']))
    dump(out/'VAL_SELECTION.json',dict(protocol=protocol,selections=selected))
    write_csv(out/'VAL_SELECTION.csv',selections)
    selection_hash = sha(out/'VAL_SELECTION.json')
    detailed=[]
    from soccergmr.evaluation.official.metrics import compute_G_mIoU,prepare_submission_for_gmiou
    for split in ('standard','full'):
        gt = read_rows(args.tests/'ground_truth'/f'{split}.jsonl.gz')
        assert not ({(g['vid'],g['query']) for g in gt}&{(g['vid'],g['query']) for g in val_gt})
        y = np.array([bool(g['relevant_windows']) for g in gt])
        for name in NAMES:
            data = {}
            for model in METHODS:
                root = Path(inputs['backbones'][name].get('test_root',args.tests))/'runs'/name/split/model
                done = json.loads((root/'complete.json').read_text())
                metrics = json.loads((root/'metrics.json').read_text())
                checkpoint = done.get('checkpoint_sha256',metrics.get('provenance',{}).get('checkpoint_sha256'))
                assert checkpoint == inputs['backbones'][name][model]['sha256']
                with gzip.open(root/'predictions.jsonl.gz','rb') as stream:
                    assert hashlib.sha256(stream.read()).hexdigest() == done['predictions_sha256']
                predictions = read_rows(root/'predictions.jsonl.gz')
                score,utility = arrays(gt,predictions)
                data[model] = (score,utility,predictions)
            for search in SEARCHES:
                for rule in RULES:
                    for k in (1,3):
                        row = dict(search=search,rule=rule,split=split,backbone=name,k=k)
                        for model in METHODS:
                            score,utility,predictions = data[model]
                            threshold = selected[name][model][search][rule]['threshold']
                            accept = score>threshold
                            v = values(score,utility[k],y,threshold)
                            row.update({f'{model}_threshold':threshold,
                                f'{model}_G':100*float(v.mean()),
                                f'{model}_TPR':100*float(accept[y].mean()),
                                f'{model}_FPR':100*float(accept[~y].mean())})
                            gated,_ = prepare_submission_for_gmiou(predictions,cls_threshold=threshold)
                            official = compute_G_mIoU(gated,gt,k_list=(k,))
                            assert abs(round(row[f'{model}_G'],2)-official[f'G-mIoU@{k}'])<1e-9
                        row['delta_pp']=row['geq_full_G']-row['gmr_G']
                        assert row['gmr_G']>0
                        row['relative_gain_percent']=100*row['delta_pp']/row['gmr_G']
                        detailed.append(row)
            print('EVALUATED',split,name,flush=True)
    summary=[]
    for search in SEARCHES:
        for rule in RULES:
            for split in ('standard','full'):
                for k in (1,3):
                    family=[r for r in detailed if (r['search'],r['rule'],r['split'],r['k'])==(search,rule,split,k)]
                    a=float(np.mean([r['gmr_G'] for r in family]))
                    b=float(np.mean([r['geq_full_G'] for r in family]))
                    summary.append(dict(search=search,rule=rule,split=split,k=k,mean_GMR=a,mean_GEQ=b,
                        mean_delta_pp=b-a,relative_gain_of_means_percent=100*(b-a)/a,
                        mean_paired_relative_gain_percent=float(np.mean([r['relative_gain_percent'] for r in family])),
                        positive_backbones=sum(r['delta_pp']>0 for r in family),
                        mean_GMR_TPR=float(np.mean([r['gmr_TPR'] for r in family])),
                        mean_GEQ_TPR=float(np.mean([r['geq_full_TPR'] for r in family])),
                        mean_GMR_FPR=float(np.mean([r['gmr_FPR'] for r in family])),
                        mean_GEQ_FPR=float(np.mean([r['geq_full_FPR'] for r in family]))))
    write_csv(out/'DETAIL.csv',detailed)
    write_csv(out/'SUMMARY.csv',summary)
    assert sha(out/'VAL_SELECTION.json')==selection_hash
    dump(out/'AUDIT.json',dict(passed=True,selection_sha256=selection_hash,
        validation_predictions=audit,official_metric_checks=len(detailed)*2,
        script_sha256=sha(__file__),source_helper_sha256=sha(Path(__file__).with_name('analyze_presence_thresholds.py')),
        significance_tests=False,
        caveats=['Historical 5: patience6/video75; new6: patience10/video128. Single trained model pair.',
                 'Full overlaps Standard; do not pool the two as independent evidence.',
                 'Application constraints selected on Val may not hold on shifted Test data.',
                 'Comparison of objectives is exploratory; numerical gain is not a significance claim.']))
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    main()
