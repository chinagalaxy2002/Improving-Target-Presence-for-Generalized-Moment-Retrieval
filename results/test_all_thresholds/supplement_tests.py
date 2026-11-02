#!/usr/bin/env python3
"""Supplement existing Val-selected experiments; never train or select on Test."""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROJECT = Path('/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval')
CHARADES = PROJECT / 'charades/multiroute_geq_seed2023_20261101'
CORE = Path('/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr')
EVALUATOR = Path('/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared')
TAUS = (0.4, 0.5, 0.6)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compress(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
        stream.write(source.read_bytes())
    assert hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest() == sha(source)


def charades_supplement(gpu):
    out = CHARADES / 'runs/eatr/multi_positive_only'
    trained = read(out / 'TRAINED.json')
    checkpoint_hash = sha(out / 'best.ckpt')
    assert checkpoint_hash == trained['checkpoint_sha256']
    history = rows(out / 'history.jsonl')
    validated = [h for h in history if h.get('validated', True) and h.get('mAP') is not None]
    assert max(h['mAP'] for h in validated) == trained['score']
    assert any(h['epoch'] == trained['best_epoch'] and h['mAP'] == trained['score'] for h in validated)
    # Retain a digest of every old formal export, so this supplement cannot
    # silently change the original 24/48 experiment.
    protected = {str(p.relative_to(CHARADES)): sha(p) for p in (CHARADES / 'runs').glob('*/*/test_*/COMPLETE.json')}
    for rel in list(protected):
        parent = (CHARADES / rel).parent
        for name in ('metrics.json', 'predictions.jsonl'):
            p = parent / name
            protected[str(p.relative_to(CHARADES))] = sha(p)
    result = []
    for split in ('test_id', 'test_ood'):
        dest = out / split
        existed = (dest / 'COMPLETE.json').exists()
        if not existed:
            occupancy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True).strip()
            assert not occupancy, 'GPU task appeared; check placement before starting supplemental Test'
            command = [sys.executable, str(CHARADES / 'code/runner.py'), '--backbone', 'eatr', '--method', 'multi',
                       '--output-suffix', '_positive_only', '--auxiliary-query-scope', 'positive_only_v1',
                       '--mode', 'test', '--split', split]
            print('SUPPLEMENT_FORMAL_TEST', split, 'Val-best epoch', trained['best_epoch'], flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            subprocess.run(command, env=env, check=True)
        marker = read(dest / 'COMPLETE.json')
        assert marker['checkpoint_sha256'] == checkpoint_hash
        assert marker['metrics_sha256'] == sha(dest / 'metrics.json')
        assert marker['prediction_sha256'] == sha(dest / 'predictions.jsonl')
        assert marker['rows'] == 7440
        exported = rows(dest / 'predictions.jsonl')
        gt = rows(CHARADES / 'data' / (split + '.jsonl'))
        assert [r['qid'] for r in exported] == [r['evaluation_id'] for r in gt]
        assert read(dest / 'metrics.json')['coverage'] == dict(rows=7440, unique_evaluation_ids=7440, positive=3720, negative=3720)
        target = HERE / 'supplemental/charades/eatr/multi_positive_only' / split
        for name in ('COMPLETE.json', 'metrics.json'):
            target.mkdir(parents=True, exist_ok=True)
            (target / name).write_bytes((dest / name).read_bytes())
        compress(dest / 'predictions.jsonl', target / 'predictions.jsonl.gz')
        result.append(dict(split=split, reused_existing=existed, **marker))
        print('FORMAL_TEST_COMPLETE', split, read(dest / 'metrics.json')['mAP'], flush=True)
    assert sha(out / 'best.ckpt') == checkpoint_hash
    assert all(sha(CHARADES / rel) == expected for rel, expected in protected.items())
    dump(HERE / 'supplemental/charades/eatr/multi_positive_only/selection.json',
         dict(trained=trained, checkpoint_unchanged=True, original_exports_unchanged=True, tests=result,
              authorization='User: 没有正式的test补test实验。然后报告。',
              no_retraining=True, no_test_selection=True))


def soccer_supplement():
    import numpy as np
    if not hasattr(np, 'trapz'):
        np.trapz = np.trapezoid
    sys.path.insert(0, str(EVALUATOR))
    from official_eval.eval_main import evaluate_gmr
    from official_eval.metrics import compute_G_mIoU, prepare_submission_for_gmiou

    gt_path = PROJECT / 'soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/test.jsonl'
    gt = rows(gt_path)
    assert len(gt) == 1036
    assert sum(bool(g['relevant_windows']) for g in gt) == 544
    prior = read(REPO / 'results/snapshot.json')
    for backbone in ('moment_detr', 'eatr', 'qd_detr', 'cg_detr'):
        for method in ('sepffn', 'instruction', 'adaptive'):
            run = PROJECT / 'mr_experiments' / method if backbone == 'moment_detr' else CORE / 'runs' / backbone / method
            metrics_path = run / ('test/metrics_gmiou_0.4.json' if backbone == 'moment_detr' else 'test_metrics.json')
            predictions = run / ('test/submission.jsonl' if backbone == 'moment_detr' else 'test_predictions.jsonl')
            assert metrics_path.is_file() and predictions.is_file(), (backbone, method, 'missing formal export')
            old = read(metrics_path)
            pred = rows(predictions)
            assert len(pred) == len(gt) == len({p['qid'] for p in pred})
            assert {p['qid'] for p in pred} == {g['qid'] for g in gt}
            selection = {}
            if backbone == 'eatr' and method == 'adaptive':
                frozen = read(run / 'frozen_checkpoint.json')
                status = read(run / 'status.json')
                checkpoint_hash = sha(run / 'best.ckpt')
                assert status['status'] == 'COMPLETED'
                assert checkpoint_hash == frozen['sha256'] == status['checkpoint_sha256']
                history = rows(run / 'history.jsonl')
                observed = [h for h in history if h.get('validation') is not None]
                best_val = max(h['validation']['mAP'] for h in observed)
                assert best_val == frozen['best_val_map'] == 20.46
                best_epoch = min(h['epoch'] for h in observed if h['validation']['mAP'] == best_val)
                assert best_epoch == 75
                selection = dict(best_epoch=best_epoch, best_val_map=best_val, checkpoint_sha256=checkpoint_hash,
                                 existing_formal_test_reused=True, status=status['status'])
            signature = dict(prediction_sha256=sha(predictions), metrics_sha256=sha(metrics_path),
                             ground_truth_sha256=sha(gt_path), evaluator_sha256=sha(EVALUATOR / 'official_eval/metrics.py'))
            target = HERE / 'supplemental/soccergmr' / backbone / method
            cached = read(target / 'data.json') if (target / 'data.json').exists() else {}
            if cached.get('signature') == signature:
                (target / 'formal_test_metrics.json').write_bytes(metrics_path.read_bytes())
                print('REUSE_VERIFIED_THRESHOLD_EXPORT', backbone, method, flush=True)
                continue
            # Complete the missing threshold reports from the exact existing
            # formal Test predictions. No GPU inference and no weight changes.
            report = evaluate_gmr(pred, gt, cls_thresholds=TAUS, gmiou_cls_threshold=0.4, map_num_workers=1, verbose=False)
            for metric in ('AUROC', 'mAP', 'mR@1', 'mR@5', 'mR+@5', 'G-mIoU@1', 'G-mIoU@3'):
                assert report['brief'][metric] == old['brief'][metric], (backbone, method, metric)
            previous = next(r for r in prior['rows'] if r['backbone'] == backbone and r['variant'] == method)
            if previous['test'] is not None:
                assert report['brief']['mAP'] == previous['test']['mAP']
                assert signature['metrics_sha256'] == previous['source_sha256']
            grid = []
            for tau in TAUS:
                gated, _ = prepare_submission_for_gmiou(pred, cls_threshold=tau, max_pred_windows=10)
                giou = compute_G_mIoU(gated, gt, k_list=(1, 3))
                cls = report['GMR-CLS']['per_threshold'][str(tau)]
                point = dict(threshold=tau, gmiou_1=giou['G-mIoU@1']/100,
                             gmiou_3=giou['G-mIoU@3']/100, negative_accuracy=cls['TN']/492,
                             rejection_f1=cls['Rej-F1']/100)
                saved = next((r for r in prior['sweeps'] if r['backbone'] == backbone and r['variant'] == method
                              and abs(r['threshold']-tau) < 1e-10), None)
                if saved:
                    assert abs(saved['G-mIoU@3_0_1']-point['gmiou_3']) < 1e-12
                grid.append(point)
            target.mkdir(parents=True, exist_ok=True)
            (target / 'formal_test_metrics.json').write_bytes(metrics_path.read_bytes())
            dump(target / 'threshold_metrics.json', report)
            dump(target / 'data.json', dict(signature=signature, metrics=report['brief'], grid=grid,
                 source_predictions=str(predictions), source_metrics=str(metrics_path), selection=selection,
                 protocol='Existing formal Test predictions; official threshold completion only; no new model inference',
                 queries=1036, positive=544, negative=492))
            if selection:
                compress(predictions, target / 'formal_test_predictions.jsonl.gz')
                dump(target / 'selection.json', selection)
            print('THRESHOLDS_COMPLETE', backbone, method, report['brief']['mAP'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip-charades', action='store_true')
    args = parser.parse_args()
    lock_root = PROJECT / 'test_report_20261102'
    lock_root.mkdir(parents=True, exist_ok=True)
    with (lock_root / 'supplement.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not args.skip_charades:
            charades_supplement(args.gpu)
        soccer_supplement()


if __name__ == '__main__':
    main()
