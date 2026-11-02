#!/usr/bin/env python3
"""Evaluate existing Sim-DETR GMR/GEQ bests; reuse shared GEQ and E1 figures."""
import argparse
import copy
import csv
import fcntl
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

KIT = Path(__file__).resolve().parent
REPO = KIT.parents[1]
PROTOCOL = KIT.parent / 'geq_protocol'
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))

DEFAULT_SOURCE = Path('/home/guoxiangyu/simdetr_gmr_multiroute_geq')
DEFAULT_BASELINE = Path('/home/guoxiangyu/VLMbasedIter_momentretrival/Sim-DETR/results_soccer_gmr_csc/native_nullaware_maskloss6_bsz8_seed2023/model_best.ckpt')
DEFAULT_OUTPUT = Path('/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/simdetr_standard_geq_extension_20261102')
METHODS = ('baseline', 'full', 'frozen')
LABELS = dict(baseline='GMR baseline', full='GMR + GEQ Full', frozen='GMR + GEQ Frozen')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=str)+'\n')
    temporary.replace(path)


def write_predictions(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in data))
    temporary.replace(path)


def csv_write(path, data):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(data[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)


def method_audit(args):
    import torch
    import ast
    sys.path.insert(0, str(PROTOCOL))
    from geq_model import GlobalEventQueryHead
    spec = importlib.util.spec_from_file_location('original_sim_geq', args.source/'geq_simdetr.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.manual_seed(2023)
    before = original.GlobalEventQueryHead(depth=2).eval()
    shared = GlobalEventQueryHead(depth=2).eval()
    shared.load_state_dict(before.state_dict(), strict=True)
    text, video = torch.randn(2, 7, 256), [torch.randn(2, 9, 256) for _ in range(2)]
    tm, vm = torch.ones(2, 7, dtype=torch.bool), torch.ones(2, 9, dtype=torch.bool)
    tm[0, -2:] = False
    vm[1, -3:] = False
    with torch.no_grad():
        expected, actual = before(text, tm, video, vm), shared(text, tm, video, vm)
    assert torch.equal(expected, actual)
    def forward_ast(path):
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='EvidenceRead')
        return ast.dump(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='forward'), include_attributes=False)
    e1 = Path('/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/ablations/global_hard_roi_20261023/code/model.py')
    assert forward_ast(e1) == forward_ast(PROTOCOL/'geq_model.py')
    return dict(core_method_equivalent=True, shared_head_strict_state_load=True,
        shared_head_synthetic_forward_bitwise_equal=True, E1_EvidenceRead_forward_AST_equal=True,
        objective='full: original native GMR loss + GEQ BCE; frozen: GEQ BCE only',
        lr_native=1e-5, lr_geq=3e-5, weight_decay=1e-4, gradient_clip=0.1,
        validation_selection='Standard Val G-mIoU@1 mean at 0.4 / 0.5 / 0.6',
        source_kind='Sim-DETR native GMR initialization; not one of the four historical main E1 checkpoints',
        adaptation='Shared GEQ depth 2; first encoder pass before repeated HD2MR encoder call',
        differences=dict(native_backbone='Sim-DETR', native_batch=8, native_checkpoint_epoch_raw=136,
                         inference='Preserved Sim native foreground * IoU-quality scoring; canonical frozen candidate reuse'),
        files_sha256={str(p):sha(p) for p in (args.source/'geq_simdetr.py',args.source/'run_pipeline.py',PROTOCOL/'geq_model.py',e1)})


def evaluate(args):
    import numpy as np
    import torch
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    from backbones.sim_detr.adapter import build_native, standard_test_dataset, generate_submission, SimDETRGEQ
    from eval.eval_main import evaluate_gmr
    torch.set_num_threads(4)
    random.seed(2023)
    np.random.seed(2023)
    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dump(args.output/'METHOD_EQUIVALENCE.json', method_audit(args))
    paths = dict(baseline=args.baseline, full=args.source/'runs/geq_baseline_full/best.ckpt',
                 frozen=args.source/'runs/geq_baseline_frozen/best.ckpt')
    baseline_saved = torch.load(args.baseline, map_location='cpu', weights_only=False)
    gt_path = Path(baseline_saved['opt'].test_path)
    gt = rows(gt_path)
    canonical = Path('/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/test.jsonl')
    assert gt == rows(canonical)
    assert len(gt)==1036 and len({r['qid'] for r in gt})==1036
    before_hashes = {m:sha(p) for m,p in paths.items()}
    for method in ('baseline', 'full', 'frozen'):
        dest = args.output/method
        dest.mkdir(parents=True, exist_ok=True)
        identity = dict(checkpoint_sha256=before_hashes[method], ground_truth_sha256=sha(gt_path))
        if (dest/'COMPLETE.json').exists():
            old = read(dest/'COMPLETE.json')
            assert all(old[k]==v for k,v in identity.items())
            assert old['prediction_sha256']==sha(dest/'predictions.jsonl')
            assert old['metrics_sha256']==sha(dest/'metrics.json')
            print('REUSE_VERIFIED_COMPLETE',method,flush=True)
            continue
        native, criterion, opt = build_native(args.baseline, args.device)
        selected_epoch = baseline_saved['epoch']
        freeze_check = None
        model = native
        if method != 'baseline':
            model = SimDETRGEQ(native, method).to(args.device)
            saved = torch.load(paths[method], map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'], strict=True)
            selected_epoch = saved['epoch']
            history = rows(args.source/f'runs/geq_baseline_{method}/history.jsonl')
            assert math_close(saved['selection']['mean'],max(h['selection']['mean'] for h in history))
            if method == 'frozen':
                current = model.base.state_dict()
                assert current.keys()==baseline_saved['model'].keys()
                assert all(torch.equal(v.cpu(),baseline_saved['model'][k]) for k,v in current.items())
                freeze_check = dict(native_tensors_bitwise_equal=True, tensors=len(current))
        data = standard_test_dataset(opt)
        assert data.data==gt
        predictions = generate_submission(model.eval(),data,opt)
        assert [p['qid'] for p in predictions]==[g['qid'] for g in gt]
        assert len(predictions)==1036
        if method == 'frozen':
            parent = rows(args.output/'baseline/predictions.jsonl')
            assert all([w[:2] for w in p['pred_relevant_windows']]==[w[:2] for w in b['pred_relevant_windows']] for p,b in zip(predictions,parent))
            scores = [p['pred_exist_score'] for p in predictions]
            predictions = copy.deepcopy(parent)
            for p,score in zip(predictions,scores):
                p['pred_exist_score'] = score
            assert all(p['pred_relevant_windows']==b['pred_relevant_windows'] for p,b in zip(predictions,parent))
            freeze_check.update(candidate_boundaries_order_scores_identical=True,
                                parent_prediction_sha256=sha(args.output/'baseline/predictions.jsonl'))
        reports = {str(t):evaluate_gmr(predictions,gt,cls_thresholds=(.4,.5,.6),gmiou_cls_threshold=t,
                                      map_num_workers=1,verbose=False) for t in (.4,.5,.6)}
        original_name = 'gmr' if method=='baseline' else 'baseline_geq_'+method
        old_reports = read(args.source/'results'/original_name/'standard_reports.json')
        for t in ('.4','.5','.6'):
            tau = '0'+t
            for metric in ('mAP','mR@1','mR@5','mR+@5','AUROC','G-mIoU@1','G-mIoU@3'):
                assert reports[tau]['brief'][metric]==old_reports[tau]['brief'][metric],(method,tau,metric)
        if method == 'frozen':
            parent_metrics = read(args.output/'baseline/metrics.json')['0.4']['brief']
            for metric in ('mAP','mR@1','mR@5','mR+@5'):
                assert reports['0.4']['brief'][metric]==parent_metrics[metric]
        write_predictions(dest/'predictions.jsonl',predictions)
        dump(dest/'metrics.json',reports)
        dump(dest/'native_options.json',vars(opt))
        dump(dest/'COMPLETE.json',dict(**identity,method=method,backbone='sim_detr',rows=1036,
             positive=544,negative=492,selected_epoch=selected_epoch,selected_on='Standard Val',
             prediction_sha256=sha(dest/'predictions.jsonl'),metrics_sha256=sha(dest/'metrics.json'),
             original_report_fixed_metrics_reproduced=True,strict_model_load=True,frozen_audit=freeze_check,
             retraining=False,full_dataset_evaluated=False,
             source_files_sha256={str(p.relative_to(REPO)):sha(p) for p in (Path(__file__),REPO/'backbones/sim_detr/adapter.py',PROTOCOL/'geq_model.py')}))
        print('STANDARD_COMPLETE',method,reports['0.4']['brief']['mAP'],flush=True)
        del native,model,criterion
        torch.cuda.empty_cache()
    assert all(sha(p)==before_hashes[m] for m,p in paths.items())
    dump(args.output/'COMPLETE.json',dict(complete=True,test_split='SoccerGMR Standard Test',
         methods=list(METHODS),model_count=3,rows_each=1036,checkpoint_hashes_unchanged=True,
         method_equivalence=read(args.output/'METHOD_EQUIVALENCE.json'),checkpoints={m:str(p) for m,p in paths.items()},
         checkpoint_sha256=before_hashes,ground_truth_sha256=sha(gt_path),no_training=True))


def math_close(a,b):
    return abs(a-b)<1e-12


def plot(args):
    sys.path.insert(0,str(PROTOCOL))
    import plotting
    from report import measured
    destination = args.publication
    destination.mkdir(parents=True,exist_ok=True)
    assert read(args.output/'COMPLETE.json')['complete']
    series = {}
    for method in METHODS:
        d = args.output/method
        series[method] = measured(d/'predictions.jsonl',d/'metrics.json',destination/f'full_frozen_comparison_{method}_data.json')
    data = dict(series=series,protocol_note='GEQ changes rejection and localization trade-offs; all differences are relative to the same GMR baseline.',
        seed_note='Sim-DETR native GMR init | Val G1 mean at 0.4 / 0.5 / 0.6 | Standard Test 1036 queries',
        figure_name='full_frozen_comparison',subtitle='GMR baseline / GMR + GEQ Full / GMR + GEQ Frozen')
    plotting.configure_style()
    plotting.setup(METHODS,LABELS)
    plotting.render('sim_detr',data,destination.parent)
    dump(destination/'full_frozen_comparison_snapshot.json',dict(backbone='sim_detr',methods=list(METHODS),labels=LABELS,data=data))
    csv_write(destination/'full_frozen_comparison_sweep.csv',[dict(method=LABELS[m],**point,
              delta_gmiou_1_vs_GMR=point['gmiou_1']-series['baseline']['grid'][i]['gmiou_1'],
              delta_gmiou_3_vs_GMR=point['gmiou_3']-series['baseline']['grid'][i]['gmiou_3'])
              for m in METHODS for i,point in enumerate(series[m]['grid'])])
    for method in METHODS:
        d = args.output/method
        target = destination/'run_metadata'/method
        target.mkdir(parents=True,exist_ok=True)
        for name in ('metrics.json','COMPLETE.json','native_options.json'):
            (target/name).write_bytes((d/name).read_bytes())
        with (target/'predictions.jsonl.gz').open('wb') as f,gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as gz:
            gz.write((d/'predictions.jsonl').read_bytes())
    dump(destination/'METHOD_EQUIVALENCE.json',read(args.output/'METHOD_EQUIVALENCE.json'))
    dump(destination/'EVALUATION_COMPLETE.json',read(args.output/'COMPLETE.json'))
    print('PLOTTED',destination,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action',choices=['evaluate','plot','all'],default='all')
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--baseline',type=Path,default=DEFAULT_BASELINE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--publication',type=Path,default=REPO/'results/geq_epoch_study/sim_detr')
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'extension.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.action in ('evaluate','all'):
            evaluate(args)
        if args.action in ('plot','all'):
            plot(args)


if __name__=='__main__':
    main()
