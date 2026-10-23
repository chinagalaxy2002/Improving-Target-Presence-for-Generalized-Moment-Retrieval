#!/usr/bin/env python3
"""Audit this publication repository using only its labels and saved predictions.

Writes docs/audit by default. Does not train, download, mutate result caches,
load checkpoints, or access a different source workspace.
"""
import argparse
from collections import Counter
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'Phase2(addbackbone_with_S4_minusFusion)'
MODELS = ('moment_detr', 'qd_detr', 'cg_detr', 'eatr', 'flashvtg')
NAMES = dict(zip(MODELS, ('Moment-DETR', 'QD-DETR', 'CG-DETR', 'EaTR', 'FlashVTG')))
sys.path.insert(0, str(ARCHIVE / 'shared'))
sys.dont_write_bytecode = True
from official_eval.eval_main import evaluate_gmr
from official_eval.metrics import prepare_submission_for_gmiou, compute_G_mIoU


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, values):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/audit')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = set()
    splits = {}
    data_stats = {}
    for split in ('train', 'val', 'test'):
        path = ARCHIVE/'shared/labels'/f'{split}.jsonl'
        inputs.add(path)
        records = rows(path)
        assert len(records) == len({r['qid'] for r in records}), f'{split}: duplicate qids'
        for r in records:
            for w in r['relevant_windows']:
                assert len(w) == 2 and all(math.isfinite(v) for v in w)
                assert 0 <= w[0] < w[1] <= r['duration'], (split, r['qid'])
        splits[split] = records
        counts = Counter(len(r['relevant_windows']) for r in records)
        info = dict(queries=len(records), videos=len({r['vid'] for r in records}),
                    no_target=counts[0], single=counts[1], multi=sum(n for k,n in counts.items() if k>=2),
                    moments=sum(k*n for k,n in counts.items()), max_moments=max(counts),
                    above_five=sum(n for k,n in counts.items() if k>5),
                    min_duration=min(r['duration'] for r in records), max_duration=max(r['duration'] for r in records),
                    sha256=sha(path), source_regimes={})
        for source in sorted({r.get('dataset_source','unknown') for r in records}):
            sub=[r for r in records if r.get('dataset_source','unknown')==source]
            c=Counter(len(r['relevant_windows']) for r in sub)
            info['source_regimes'][source]=dict(queries=len(sub),no_target=c[0],single=c[1],multi=sum(n for k,n in c.items() if k>=2))
        data_stats[split]=info
    data_stats['overlaps']=[]
    for a,b in itertools.combinations(splits,2):
        pair={'splits':[a,b]}
        for key in ('qid','vid','query'):
            pair[key]=len({r[key] for r in splits[a]} & {r[key] for r in splits[b]})
        matches=lambda records:{json.dumps(r['match_info'],sort_keys=True) for r in records if r.get('match_info')}
        pair['exact_match_info']=len(matches(splits[a]) & matches(splits[b]))
        data_stats['overlaps'].append(pair)
    write_json(args.output/'dataset.json', data_stats)

    expected={}
    for tau in (.4,.5,.6):
        suffix='' if tau==.4 else f'_tau_{tau}'
        path=ROOT/'backbonefolder'/f'TABLE2_COMPLETE_RESULTS{suffix}.csv'
        inputs.add(path)
        with path.open() as stream:
            table={r['Model']:r for r in csv.DictReader(stream) if r['Source']=='Our experiment'}
        assert len(table)==14
        expected[tau]=table
    summary=[]
    for backbone in MODELS:
        variants=('gmr','s4') if backbone=='flashvtg' else ('original','gmr','s4')
        for variant in variants:
            if variant=='s4':
                cache=ROOT/'backbonefolder/figures/complete_comparison/data'/f'{backbone}_s4_epoch4.json'
            else:
                cache=ARCHIVE/'shared/plot_baselines'/f'{backbone}_{variant}.json'
            inputs.add(cache)
            source=json.loads(cache.read_text())
            path=(ROOT/source['source_predictions']).resolve()
            assert path.is_relative_to(ROOT), 'Prediction input must be in this repository'
            inputs.add(path)
            assert sha(path)==source['source_sha256'], f'{backbone}/{variant}: hash mismatch'
            predictions=rows(path)
            by_id={str(r['qid']):r for r in predictions}
            gt=splits['test']
            assert len(predictions)==len(by_id)==len(gt)
            assert set(by_id)=={str(r['qid']) for r in gt}
            predictions=[{**by_id[str(r['qid'])], 'qid':r['qid']} for r in gt]
            official=evaluate_gmr(predictions,gt,cls_thresholds=(.4,.5,.6),map_num_workers=0,verbose=False)
            name=NAMES[backbone] if variant=='original' else NAMES[backbone]+'-GMR'
            if variant=='s4':name+=' + S4_minus_Fusion (epoch 4)'
            elif backbone=='flashvtg':name+=' (E0)'
            for tau in (.4,.5,.6):
                c=official['GMR-CLS']['per_threshold'][str(tau)]
                gated,_=prepare_submission_for_gmiou(predictions,cls_threshold=tau,max_pred_windows=10)
                g=compute_G_mIoU(gated,gt,k_list=(1,3))
                row=dict(backbone=backbone,variant=variant,model=name,threshold=tau,
                         AUROC=official['brief']['AUROC'],Rej_F1=c['Rej-F1'],
                         **{k:official['brief'][k] for k in ('mAP','mR@1','mR@5','mR+@5')},**g,
                         N_acc=100*c['TN']/(c['TN']+c['FP']),PosRec=100*c['TP']/(c['TP']+c['FN']),
                         **{k:c[k] for k in ('TP','TN','FP','FN')},prediction_file=str(path.relative_to(ROOT)),prediction_sha256=sha(path))
                published=expected[tau][name]
                mapping={'Rej_F1':f'Rej-F1@{tau}'}
                for key in ('AUROC','Rej_F1','mAP','mR@1','mR@5','mR+@5','G-mIoU@1','G-mIoU@3'):
                    assert abs(row[key]-float(published[mapping.get(key,key)]))<.011,(name,tau,key)
                summary.append(row)
            print(f'PASS {backbone}/{variant}: 1036 queries; three thresholds',flush=True)
    assert len(summary)==42
    write_csv(args.output/'metrics.csv',summary)
    gains=[]
    for b in MODELS:
        for tau in (.4,.5,.6):
            baseline=next(r for r in summary if r['backbone']==b and r['variant']=='gmr' and r['threshold']==tau)
            adapted=next(r for r in summary if r['backbone']==b and r['variant']=='s4' and r['threshold']==tau)
            gains.append(dict(backbone=b,threshold=tau,**{k:round(adapted[k]-baseline[k],2) for k in ('AUROC','Rej_F1','mAP','G-mIoU@1','G-mIoU@3','N_acc','PosRec')}))
    write_csv(args.output/'gains.csv',gains)
    for b in MODELS:
        inputs.add(ARCHIVE/b/'configuration/CONFIG.json')
        inputs.add(ARCHIVE/b/'model.py' if b!='flashvtg' else ARCHIVE/b/'code/model.py')
    inputs.add(ARCHIVE/'shared/config.json')
    inputs.add(ARCHIVE/'shared/train.py')
    inputs.add(ROOT/'PUBLICATION_SCOPE.json')
    write_json(args.output/'sources.json',dict(repository='chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval',
        inspected_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip(),
        sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(inputs)}))
    write_json(args.output/'verification.json',dict(passed=True,data_scope='Original Standard data; no data augmentation experiments',
        local_experiments=14,thresholds=[.4,.5,.6],metric_rows_verified=42,compared_metrics_per_row=8,
        local_table_metric_checks=336,all_predictions_hash_matched=True,all_test_qids_matched=True,
        external_paper_rows_excluded=6,training_or_checkpoint_inference=False,external_source_workspace_used=False))


if __name__=='__main__':main()
