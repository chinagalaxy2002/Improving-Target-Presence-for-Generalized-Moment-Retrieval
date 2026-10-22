#!/usr/bin/env python3
"""Recompute archived epoch4 TEST metrics and figures, using saved predictions only."""
import argparse
import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'shared'))
from official_eval.eval_main import evaluate_gmr
import epoch_plots as plots
plots.NAMES['flashvtg']='FlashVTG'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone',choices=('all','moment_detr','qd_detr','cg_detr','eatr','flashvtg'),default='all')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--metrics-only',action='store_true')
    args=parser.parse_args()
    names=('moment_detr','qd_detr','cg_detr','eatr','flashvtg') if args.backbone=='all' else (args.backbone,)
    output=args.output or ROOT/'reproduced'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True,exist_ok=False)
    gt=[json.loads(line) for line in (ROOT/'shared/labels/test.jsonl').read_text().splitlines() if line.strip()]
    summary={}
    for name in names:
        index=json.loads((ROOT/name/'ARCHIVE_INDEX.json').read_text())
        prediction=ROOT/name/index['test_predictions']
        pred=[json.loads(line) for line in prediction.read_text().splitlines() if line.strip()]
        assert len(pred)==len(gt)==1036 and {str(p['qid']) for p in pred}=={str(g['qid']) for g in gt}
        result=evaluate_gmr(pred,gt,cls_thresholds=(.4,.5,.6,.8),map_num_workers=0,verbose=False)
        recorded=json.loads((ROOT/name/index['test_metrics']).read_text())
        assert result['brief']==recorded['brief'],f'{name}: metric mismatch'
        directory=output/name;directory.mkdir()
        (directory/'test_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        if not args.metrics_only:
            epochdir=ROOT/name/'epoch_004'
            current=json.loads((epochdir/'figures/S4_TEST_CURVES.json').read_text())
            variants=('gmr',) if name=='flashvtg' else ('original','gmr')
            baseline={v:json.loads((ROOT/'shared/plot_baselines'/f'{name}_{v}.json').read_text()) for v in variants}
            plots.render(name,{**baseline,'epoch_004':current},directory/'this_epoch')
            previous=epochdir/'supporting_curves/epoch_003.json'
            cumulative=dict(baseline)
            if previous.exists():cumulative['epoch_003']=json.loads(previous.read_text())
            cumulative['epoch_004']=current
            plots.render(name,cumulative,directory/'all_epochs_so_far')
        summary[name]={'passed':True,'epoch':4,'split':'test','queries':len(pred),'metrics_match_archive':True,
                       'figures_regenerated':not args.metrics_only}
        print(name,'TEST_METRICS_MATCH',flush=True)
    (output/'REPRODUCTION.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(output)


if __name__=='__main__':
    main()
