"""Rebuild documentation tables from labels and frozen predictions, without GPU.

Only docs/research/generated is written. No training or asset access occurs.
Requires NumPy; official metrics are loaded from the existing local evaluator.
"""
import argparse
from collections import Counter
import copy
import csv
import gzip
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get('GMR_DOCUMENTATION_SOURCE_ROOT', Path(__file__).resolve().parents[2])).resolve()
PHASE = ROOT / 'Phase2(addbackbone_with_S4_minusFusion)'


def rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_csv(path, data):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'generated')
    args = parser.parse_args()
    if not (ROOT / 'data/train.jsonl').exists():
        parser.error('Full source workspace required. Set GMR_DOCUMENTATION_SOURCE_ROOT to the workspace containing data/, results/, and baseline_source/. For the GitHub publication archive, use tools/reproduce_results.py instead.')
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    splits = {s: rows(ROOT / 'data' / f'{s}.jsonl') for s in ['train', 'val', 'test']}
    stats, flat, overlaps = {}, [], []
    for split, data in splits.items():
        ids = [r['qid'] for r in data]
        if len(ids) != len(set(ids)):
            errors.append(f'{split}: duplicate qids')
        invalid = []
        for row in data:
            for window in row['relevant_windows']:
                if not (len(window) == 2 and all(math.isfinite(v) for v in window) and 0 <= window[0] < window[1] <= row['duration']):
                    invalid.append(row['qid'])
        if invalid:
            errors.append(f'{split}: invalid windows at {invalid}')
        count = Counter(len(r['relevant_windows']) for r in data)
        info = dict(queries=len(data), videos=len({r['vid'] for r in data}),
                    no_target=count[0], single=count[1], multi=sum(v for k, v in count.items() if k >= 2),
                    total_moments=sum(k*v for k,v in count.items()), max_moments=max(count),
                    queries_above_five=sum(v for k,v in count.items() if k > 5),
                    window_count_histogram=dict(sorted(count.items())),
                    durations=dict(Counter(r['duration'] for r in data)),
                    sources=dict(Counter(r.get('dataset_source', 'unknown') for r in data)),
                    action_types=dict(Counter(r.get('action_type', 'unknown') for r in data)),
                    invalid_window_qids=invalid, sha256=sha(ROOT / 'data' / f'{split}.jsonl'))
        info['source_regimes'] = {}
        for source in info['sources']:
            subset = [r for r in data if r.get('dataset_source','unknown') == source]
            c = Counter(len(r['relevant_windows']) for r in subset)
            info['source_regimes'][source] = dict(queries=len(subset), no_target=c[0], single=c[1], multi=sum(v for k,v in c.items() if k>=2))
        stats[split] = info
        flat.append(dict(split=split, **{k:info[k] for k in ['queries','videos','no_target','single','multi','total_moments','max_moments','queries_above_five']}))
    for a, b in itertools.combinations(splits, 2):
        rec = {'splits': [a,b]}
        for field in ['qid', 'vid', 'query']:
            common = {r[field] for r in splits[a]} & {r[field] for r in splits[b]}
            rec[field + '_overlap_count'] = len(common)
            # Query text can legitimately repeat across videos; no automatic leakage claim.
            if field != 'query': rec[field + '_examples'] = sorted(common, key=str)[:12]
        ma = {json.dumps(r['match_info'],sort_keys=True) for r in splits[a] if r.get('match_info')}
        mb = {json.dumps(r['match_info'],sort_keys=True) for r in splits[b] if r.get('match_info')}
        rec['match_info_overlap_count'] = len(ma & mb)
        overlaps.append(rec)
    stats['cross_split_overlap'] = overlaps
    stats['match_overlap_scope'] = 'Exact match_info only where supplied; not inferred from filenames.'

    sys.path.insert(0, str(ROOT / 'baseline_source'))
    from eval.metrics import compute_G_mIoU, compute_gmr_cls, compute_mAP
    gt = splits['test']
    fixed = []
    verification = []
    for model in ['E0', 'S4Fusion', 'S4Fusion_QCD_A3_ep04']:
        path = ROOT / 'results' / model / 'test_predictions.jsonl.gz'
        prediction = rows(path)
        if len(prediction) != len(gt) or {r['qid'] for r in prediction} != {r['qid'] for r in gt}:
            raise ValueError(f'{model}: prediction qids mismatch')
        report = json.loads((path.parent / 'test_metrics.json').read_text())
        ap = compute_mAP(prediction, gt, num_workers=0)['mAP']
        for tau in [.4, .5, .6]:
            cls = compute_gmr_cls(prediction, gt, (tau,))
            c = cls['per_threshold'][str(tau)]
            gated = copy.deepcopy(prediction)
            for r in gated:
                if r['pred_exist_score'] <= tau:
                    r['pred_relevant_windows'] = []
            g = compute_G_mIoU(gated, gt)
            item = dict(model=model, threshold=tau, **g, mAP=ap,
                        AUROC=cls['AUROC'], Rej_F1=c['Rej-F1'],
                        N_acc=100*c['TN']/(c['TN']+c['FP']),
                        PosRec=100*c['TP']/(c['TP']+c['FN']),
                        TP=c['TP'], TN=c['TN'], FP=c['FP'], FN=c['FN'],
                        exact_threshold_ties=sum(r['pred_exist_score']==tau for r in prediction),
                        prediction_sha256=sha(path))
            key = {.4:'gate_04', .5:'fixed_05', .6:'gate_06'}[tau]
            expected = report[key]
            comparisons = {k: (v,expected[k]) for k,v in g.items()}
            comparisons['mAP']=(ap,expected['mAP'])
            comparisons['AUROC']=(cls['AUROC'],round(report['AUROC'],2))
            rej_expected = 200*expected['TN']/(2*expected['TN']+expected['FN']+expected['FP'])
            comparisons['Rej_F1']=(c['Rej-F1'],round(rej_expected,2))
            for k,(got,want) in comparisons.items():
                if abs(got-want)>0.011: errors.append(f'{model} tau={tau} {k}: {got} != {want}')
            fixed.append(item)
        verification.append({'model':model,'queries':len(prediction),'prediction_sha256':sha(path),'official_recomputed':True})

    cross = []
    for b in ['moment_detr','qd_detr','cg_detr','eatr','flashvtg']:
        p=PHASE/b/'epoch_004/test_results/test_predictions.jsonl'
        pred=rows(p)
        if len(pred)!=len(gt) or {r['qid'] for r in pred}!={r['qid'] for r in gt}: raise ValueError(f'{b}: qid mismatch')
        cls=compute_gmr_cls(pred,gt,(.4,)); c=cls['per_threshold']['0.4']
        gated=copy.deepcopy(pred)
        for r in gated:
            if r['pred_exist_score']<=.4:r['pred_relevant_windows']=[]
        g=compute_G_mIoU(gated,gt)
        ap=compute_mAP(pred,gt,num_workers=0)['mAP']
        baseline=json.loads((PHASE/'shared/plot_baselines'/f'{b}_gmr.json').read_text())
        bm=baseline['tau_0.4']
        bc=next(x for x in baseline['grid'] if abs(x['threshold']-.4)<1e-9)
        item=dict(backbone=b,epoch=4,threshold=.4,AUROC=cls['AUROC'],Rej_F1=c['Rej-F1'],mAP=ap,
                  G1=g['G-mIoU@1'],G3=g['G-mIoU@3'],N_acc=100*c['TN']/(c['TN']+c['FP']),
                  PosRec=100*c['TP']/(c['TP']+c['FN']),baseline_AUROC=bm['AUROC'],baseline_mAP=bm['mAP'],
                  baseline_G1=bm['G-mIoU@1'],baseline_G3=bm['G-mIoU@3'],baseline_N_acc=bc['N_acc'],baseline_PosRec=bc['positive_recall'],
                  delta_AUROC=round(cls['AUROC']-bm['AUROC'],2),delta_mAP=round(ap-bm['mAP'],2),
                  delta_G1=round(g['G-mIoU@1']-bm['G-mIoU@1'],2),
                  delta_PosRec=round(100*c['TP']/(c['TP']+c['FN'])-bc['positive_recall'],2),
                  baseline_evidence='archived curve summary; current S4 recomputed from predictions',
                  prediction_sha256=sha(p))
        for metric,value in [('AUROC',cls['AUROC']),('Rej-F1@0.4',c['Rej-F1']),('mAP',ap),('G-mIoU@1',g['G-mIoU@1']),('G-mIoU@3',g['G-mIoU@3'])]:
            table=list(csv.DictReader((PHASE/'EPOCH4_TEST_RESULTS.csv').open()))
            # Headers are discovered to keep the archive's naming intact.
            target=next(r for r in table if r['backbone'].lower().replace('-','')==b.replace('_',''))
            if metric in target and abs(value-float(target[metric]))>.011:errors.append(f'{b} {metric} mismatch')
        cross.append(item)
        verification.append({'model':b,'queries':len(pred),'official_recomputed':True,'prediction_sha256':sha(p)})

    for name,obj in [('dataset_statistics.json',stats),('verification.json',{'passed':not errors,'errors':errors,'models':verification,'scope':'Current evaluation from frozen predictions only; no training or checkpoint inference rerun.'})]:
        (args.output/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
    output_csv(args.output/'dataset_statistics.csv',flat)
    output_csv(args.output/'fixed_threshold_results.csv',fixed)
    output_csv(args.output/'cross_backbone_results.csv',cross)
    print(json.dumps({'passed':not errors,'errors':errors,'splits':flat,'cross_backbone':cross,'overlaps':overlaps},indent=2))
    if errors: raise SystemExit(1)


if __name__=='__main__':main()
