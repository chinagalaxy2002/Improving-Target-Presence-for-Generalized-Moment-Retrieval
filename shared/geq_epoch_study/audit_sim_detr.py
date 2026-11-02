#!/usr/bin/env python3
"""Audit published SoccerGMR Sim-DETR predictions, curves, sources and figures."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode=True

import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'results/geq_epoch_study/sim_detr'
if not hasattr(np,'trapz'):
    np.trapz=np.trapezoid
sys.path.insert(0,str(REPO/'backbones/sim_detr/source'))
from eval.eval_main import evaluate_gmr
from eval.metrics import _compute_set_iou_score


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    gt_path=Path('/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/test.jsonl')
    gt=[json.loads(x) for x in gt_path.read_text().splitlines() if x.strip()]
    assert len(gt)==1036
    positive=np.array([bool(g['relevant_windows']) for g in gt])
    assert positive.sum()==544
    snapshot=read(OUT/'full_frozen_comparison_snapshot.json')
    assert snapshot['backbone']=='sim_detr'
    all_predictions={}
    checks={}
    for mode in ('baseline','full','frozen'):
        dest=OUT/'run_metadata'/mode
        marker=read(dest/'COMPLETE.json')
        assert marker['rows']==1036 and marker['positive']==544 and marker['negative']==492
        assert marker['ground_truth_sha256']==sha(gt_path)
        assert sha(dest/'metrics.json')==marker['metrics_sha256']
        raw=gzip.decompress((dest/'predictions.jsonl.gz').read_bytes())
        assert hashlib.sha256(raw).hexdigest()==marker['prediction_sha256']
        predictions=[json.loads(x) for x in raw.splitlines() if x.strip()]
        assert len(predictions)==1036 and len({p['qid'] for p in predictions})==1036
        assert [p['qid'] for p in predictions]==[g['qid'] for g in gt]
        all_predictions[mode]=predictions
        reports=read(dest/'metrics.json')
        for tau in (.4,.5,.6):
            actual=evaluate_gmr(predictions,gt,cls_thresholds=(.4,.5,.6),gmiou_cls_threshold=tau,map_num_workers=1,verbose=False)
            for metric,value in reports[str(tau)]['brief'].items():
                assert actual['brief'][metric]==value,(mode,tau,metric)
        checks[mode+'_three_fixed_points_officially_recomputed']=True
        series=snapshot['data']['series'][mode]
        scores=np.array([p['pred_exist_score'] for p in predictions])
        raw_iou={k:np.array([_compute_set_iou_score([w[:2] for w in p['pred_relevant_windows'][:k]],g['relevant_windows']) for p,g in zip(predictions,gt)]) for k in (1,3)}
        for i in range(101):
            point=series['grid'][i];keep=scores>(i/100)
            tn=int((~keep&~positive).sum());fp=int((keep&~positive).sum());fn=int((~keep&positive).sum())
            for k in (1,3):
                expected=round(100*float(np.where(keep,raw_iou[k],~positive).mean()),2)/100
                assert math.isclose(point[f'gmiou_{k}'],expected,abs_tol=1e-12)
            assert math.isclose(point['negative_accuracy'],tn/492,abs_tol=1e-12)
            assert math.isclose(point['rejection_f1'],round(100*2*tn/max(1,2*tn+fp+fn),2)/100,abs_tol=1e-12)
        checks[mode+'_all_101_thresholds_recomputed']=True
    assert all(a['pred_relevant_windows']==b['pred_relevant_windows'] for a,b in zip(all_predictions['baseline'],all_predictions['frozen']))
    frozen=read(OUT/'run_metadata/frozen/COMPLETE.json')['frozen_audit']
    assert frozen['native_tensors_bitwise_equal'] and frozen['tensors']==300
    checks['frozen_native_and_all_exported_candidates_unchanged']=True
    with (OUT/'full_frozen_comparison_sweep.csv').open() as f:
        sweep=list(csv.DictReader(f))
    assert len(sweep)==303
    for index,mode in enumerate(('baseline','full','frozen')):
        for i in range(101):
            row=sweep[index*101+i]
            for k in (1,3):
                expected=snapshot['data']['series'][mode]['grid'][i][f'gmiou_{k}']-snapshot['data']['series']['baseline']['grid'][i][f'gmiou_{k}']
                assert math.isclose(float(row[f'delta_gmiou_{k}_vs_GMR']),expected,abs_tol=1e-12)
    checks['all_303_curve_rows_use_same_GMR_reference']=True
    complete=read(OUT/'EVALUATION_COMPLETE.json')
    assert complete['test_split']=='SoccerGMR Standard Test' and complete['no_training']
    for mode,path in complete['checkpoints'].items():
        assert sha(Path(path))==complete['checkpoint_sha256'][mode]
    checks['three_selected_checkpoints_unchanged']=True
    method=read(OUT/'METHOD_EQUIVALENCE.json')
    assert all(method[k] for k in ('core_method_equivalent','shared_head_strict_state_load','shared_head_synthetic_forward_bitwise_equal','E1_EvidenceRead_forward_AST_equal'))
    checks['shared_GEQ_and_E1_readout_equivalence_verified']=True
    source=read(REPO/'backbones/sim_detr/SOURCE_MANIFEST.json')
    assert all(sha(REPO/'backbones/sim_detr/source'/path)==expected for path,expected in source['files'].items())
    checks['all_bundled_native_sources_match_original']=True
    for mode in ('full','frozen'):
        p=OUT/'run_metadata'/mode/'history.jsonl'
        history=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        marker=read(OUT/'run_metadata'/mode/'COMPLETE.json')
        selected=next(h for h in history if h['epoch']==marker['selected_epoch'])
        assert selected['selection']['mean']==max(h['selection']['mean'] for h in history)
    checks['both_GEQs_selected_at_Val_mean_maximum']=True
    figures={}
    for p in OUT.glob('*.png'):
        with Image.open(p) as im:
            im.verify()
        figures[p.name]=sha(p)
    for p in OUT.glob('*.pdf'):
        info=subprocess.check_output(['pdfinfo',str(p)],text=True)
        assert re.search(r'Pages:\s+1\b',info)
        figures[p.name]=sha(p)
    for p in OUT.glob('*.svg'):
        assert '<svg' in p.read_text()
        figures[p.name]=sha(p)
    offline=Path('/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/simdetr_standard_geq_extension_20261102/offline_replot/sim_detr')
    for stem in ('full_frozen_comparison','full_frozen_comparison_gmiou_1_3'):
        for ext in ('png','pdf','svg'):
            assert sha(OUT/(stem+'.'+ext))==sha(offline/(stem+'.'+ext)),(stem,ext)
    checks['six_offline_replots_byte_identical']=True
    for p in [REPO/'README.md',OUT/'README.md',REPO/'backbones/sim_detr/README.md']:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('http','#')):continue
            target=target.split('#')[0]
            if target in ('AUDIT.json','FIGURE_REVIEW.md'):continue
            assert (p.parent/target).exists(),(p,target)
    checks['README_relative_links_valid']=True
    result=dict(passed=True,dataset='SoccerGMR Standard Test',charades_evaluated=False,
        rows_each=1036,positive=544,negative=492,models=3,checks=checks,figure_sha256=figures)
    write=OUT/'AUDIT.json'
    write.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
