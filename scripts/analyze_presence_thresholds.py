#!/usr/bin/env python3
"""Historical checkpoint diagnostic: select thresholds on Val, evaluate fixed Tests.

Inference reuses the historical compatibility reader/exporter. No training or
checkpoint selection is performed. Selection is serialized before Test is read.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

NAMES = ('moment_detr', 'eatr', 'qd_detr', 'cg_detr', 'sim_detr',
         'bm_detr', 'bam_detr', 'uvcom', 'tr_detr', 'taskweave', 'ld_detr')
METHODS = ('gmr', 'geq_full')
THRESHOLDS = np.arange(101, dtype=float) / 100


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def zipped_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n').encode())


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def infer(args):
    # Native backbones share top-level package names and require fresh processes.
    if len(args.backbones) > 1:
        import os
        import subprocess
        import sys
        manifest = json.loads(args.manifest.read_text())
        for name in args.backbones:
            entry = manifest['backbones'][name]
            env = dict(os.environ, PYTHONPATH=entry.get('runtime_src', os.environ.get('PYTHONPATH', 'src')))
            subprocess.run([entry.get('runtime_python', sys.executable), str(Path(__file__).resolve()), 'infer',
                            '--manifest', str(args.manifest), '--output', str(args.output),
                            '--backbones', name, '--device', args.device], check=True, env=env)
        return
    import torch
    from torch.utils.data import DataLoader
    from evaluate_historical_geq import Inputs, collate, export
    from soccergmr.backbones.registry import build, get_spec, forward
    from soccergmr.checkpoint import seed_all, source_identity, run_lock
    from soccergmr.config import Options
    from soccergmr.data import move

    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    seed_all(2023)
    labels = read_rows(manifest['val']['labels'])
    assert len(labels) == 465 and len({r['qid'] for r in labels}) == 465
    label_hash = sha(manifest['val']['labels'])
    for name in args.backbones:
        entry = manifest['backbones'][name]
        if entry.get('profile') == 'formal':
            infer_formal(args, name, entry, labels, label_hash)
            continue
        baseline = torch.load(entry['gmr']['path'], map_location='cpu', weights_only=False)
        opt = baseline.get('opt', baseline.get('config'))
        opt = Options(opt if isinstance(opt, dict) else vars(opt))
        opt.binding_loss_coef = 0.
        opt.device = args.device
        spec = get_spec(name)
        config = dict(backbone=name, architecture=opt, data={'max_video_tokens': opt.max_v_l})
        data = Inputs(manifest['val']['labels'], manifest['val']['features'], spec, opt)
        feature_hashes = {str(p): sha(p) for row in labels for p in data.paths(row)}
        for method in METHODS:
            dest = args.output / name / method
            with run_lock(dest):
                path = Path(entry[method]['path'])
                checkpoint_hash = sha(path)
                assert checkpoint_hash == entry[method]['sha256']
                identity = dict(checkpoint_sha256=checkpoint_hash, label_sha256=label_hash,
                                feature_hashes=feature_hashes, source=source_identity(),
                                runner_sha256=sha(__file__), compatibility_sha256=sha(Path(__file__).with_name('evaluate_historical_geq.py')),
                                torch=torch.__version__, device=args.device)
                if (dest / 'complete.json').exists():
                    done = json.loads((dest / 'complete.json').read_text())
                    assert done['identity'] == identity
                    assert done['predictions_sha256'] == sha(dest / 'predictions.jsonl.gz')
                    print('REUSE', name, method, flush=True)
                    continue
                model, _ = build(config, args.device)
                if method == 'geq_full':
                    from soccergmr.geq import GEQFull
                    model = GEQFull(model, spec).to(args.device)
                saved = torch.load(path, map_location='cpu', weights_only=False)
                model.load_state_dict(saved['model'], strict=True)
                model.eval().requires_grad_(False)
                batch = opt.eval_bsz if name == 'sim_detr' else int(entry['batch_size'])
                loader = DataLoader(data, batch_size=batch, shuffle=False, num_workers=2, collate_fn=collate)
                predictions = []
                with torch.inference_mode():
                    for metas, inputs in loader:
                        outputs = forward(model, name, move(inputs, args.device), metas=metas)
                        predictions.extend(export(outputs, metas, name, opt))
                assert [p['qid'] for p in predictions] == [g['qid'] for g in labels]
                assert sha(path) == checkpoint_hash
                zipped_rows(dest / 'predictions.jsonl.gz', predictions)
                dump(dest / 'complete.json', dict(identity=identity, rows=len(predictions),
                     selected_epoch=saved['epoch'], checkpoint_sha256=checkpoint_hash,
                     predictions_sha256=sha(dest / 'predictions.jsonl.gz'),
                     strict_state_load=True, training_called=False))
                if method == 'geq_full':
                    model.close()
                del model, saved
                torch.cuda.empty_cache()
                print('COMPLETE', name, method, len(predictions), flush=True)


def infer_formal(args, name, entry, labels, label_hash):
    import torch
    from soccergmr.backbones.registry import build, get_spec
    from soccergmr.checkpoint import source_identity, run_lock, asset_identity
    from soccergmr.data import SoccerDataset
    from soccergmr.geq import GEQFull
    from soccergmr.predict import predict
    for method in METHODS:
        dest = args.output / name / method
        with run_lock(dest):
            path = Path(entry[method]['path'])
            checkpoint_hash = sha(path)
            assert checkpoint_hash == entry[method]['sha256']
            saved = torch.load(path, map_location='cpu', weights_only=False)
            stage = json.loads((path.parent / 'complete.json').read_text())
            assert stage['best_sha256'] == checkpoint_hash and stage['status'] == 'complete'
            assert saved['source'] == source_identity(), (name, 'Frozen runtime changed')
            config = saved['config']
            data = SoccerDataset(config, 'val', get_spec(name))
            assert data.data == labels and sha(data.path) == label_hash
            identity = dict(checkpoint_sha256=checkpoint_hash, label_sha256=label_hash,
                            assets=asset_identity([data]), source=saved['source'],
                            runner_sha256=sha(__file__), torch=torch.__version__, device=args.device)
            if (dest / 'complete.json').exists():
                done = json.loads((dest / 'complete.json').read_text())
                assert done['identity'] == identity
                assert done['predictions_sha256'] == sha(dest / 'predictions.jsonl.gz')
                print('REUSE', name, method, flush=True)
                continue
            model, _ = build(config, args.device)
            if method == 'geq_full':
                model = GEQFull(model, get_spec(name)).to(args.device)
            model.load_state_dict(saved['model'], strict=True)
            model.eval().requires_grad_(False)
            predictions = predict(model, data, config, args.device)
            assert [p['qid'] for p in predictions] == [g['qid'] for g in labels]
            assert sha(path) == checkpoint_hash
            zipped_rows(dest / 'predictions.jsonl.gz', predictions)
            dump(dest / 'complete.json', dict(identity=identity, rows=len(predictions),
                selected_epoch=stage['best_epoch'], checkpoint_sha256=checkpoint_hash,
                predictions_sha256=sha(dest / 'predictions.jsonl.gz'),
                strict_state_load=True, training_called=False))
            if method == 'geq_full':
                model.close()
            del model, saved
            torch.cuda.empty_cache()
            print('COMPLETE', name, method, len(predictions), flush=True)


def arrays(gt, predictions):
    from soccergmr.evaluation.official.metrics import _clean_pred_windows, _compute_set_iou_score
    indexed = {p['qid']: p for p in predictions}
    assert len(indexed) == len(predictions) == len(gt)
    assert set(indexed) == {g['qid'] for g in gt}
    ordered = [indexed[g['qid']] for g in gt]
    assert all(p['vid'] == g['vid'] and p.get('query', g['query']) == g['query'] for p, g in zip(ordered, gt))
    scores = np.array([p['pred_exist_score'] for p in ordered])
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    utilities = {}
    for k in (1, 3):
        utilities[k] = np.array([_compute_set_iou_score(
            [w[:2] for w in _clean_pred_windows(p['pred_relevant_windows'], max_pred_windows=10)[:k]],
            g['relevant_windows']) for p, g in zip(ordered, gt)])
    return scores, utilities


def values(scores, utility, y, threshold):
    return np.where(scores > threshold, utility, (~y).astype(float))


def select(scores, utility, y):
    curve = np.array([values(scores, utility, y, t).mean() for t in THRESHOLDS])
    tied = np.flatnonzero(np.isclose(curve, curve.max(), rtol=0, atol=1e-12))
    index = min(tied, key=lambda i: (abs(int(i) - 50), int(i)))
    return dict(threshold=float(THRESHOLDS[index]), val_G1=100 * float(curve[index]),
                tied_maxima=[float(THRESHOLDS[i]) for i in tied], curve=(100 * curve).tolist())


def rank_curve(scores, utility, y):
    """At tied scores, interpolation means random selection within the tie group."""
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True]) + 1
    increment = utility - (~y).astype(float)
    x = np.r_[0, ends / len(y)]
    g = (~y).mean() + np.r_[0, np.cumsum(increment[order])[ends - 1] / len(y)]
    return x, g


def bootstrap(delta, vids, repetitions=2000):
    _, inverse = np.unique(vids, return_inverse=True)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(20261103)
    weights = rng.multinomial(len(counts), np.ones(len(counts)) / len(counts), size=repetitions)
    estimates = (weights @ sums) / (weights @ counts)
    return (100 * np.quantile(estimates, [.025, .975])).tolist()


def report(args):
    from sklearn.metrics import roc_auc_score
    from soccergmr.evaluation.official.metrics import compute_G_mIoU, prepare_submission_for_gmiou
    manifest = json.loads(args.manifest.read_text())
    root = args.tests
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(selection_split='Standard Val', checkpoint_selection='unchanged finalized bests',
        objective='G-mIoU@1', thresholds=THRESHOLDS.tolist(),
        ties='within 1e-12: closest to 0.5, then smaller threshold',
        decision='pred_exist_score > threshold', secondary='G-mIoU@3 uses the same G1-selected threshold',
        primary='Select threshold on GMR Val G1; evaluate GMR and GEQ at this same threshold on both Tests',
        second_shared='Select threshold on GEQ best Val G1; evaluate GMR and GEQ at this same threshold on both Tests',
        independent_control='Additionally select a separate Val G1 threshold for every candidate/scorer cell',
        candidate_sources=list(METHODS), score_sources=list(METHODS),
        test_tuning=False, new_training=False, bootstrap='2000 paired video-cluster resamples, seed 20261103')
    dump(out / 'PROTOCOL.json', protocol)
    val_gt = read_rows(manifest['val']['labels'])
    assert len(val_gt) == 465
    zipped_rows(out / 'ground_truth/val.jsonl.gz', val_gt)
    yval = np.array([bool(g['relevant_windows']) for g in val_gt])
    selected, val_rows = {}, []
    audit = []
    # This complete selection loop finishes and is serialized before reading Tests.
    for name in NAMES:
        data = {}
        for method in METHODS:
            p = args.val_predictions / name / method
            done = json.loads((p / 'complete.json').read_text())
            assert done['checkpoint_sha256'] == manifest['backbones'][name][method]['sha256']
            assert done['identity']['label_sha256'] == sha(manifest['val']['labels'])
            assert done['predictions_sha256'] == sha(p / 'predictions.jsonl.gz')
            data[method] = arrays(val_gt, read_rows(p / 'predictions.jsonl.gz'))
            audit.append(dict(backbone=name, method=method, val_prediction_sha256=done['predictions_sha256'],
                              checkpoint_sha256=done['checkpoint_sha256']))
        selected[name] = {}
        for candidate in METHODS:
            selected[name][candidate] = {}
            for scorer in METHODS:
                result = select(data[scorer][0], data[candidate][1][1], yval)
                selected[name][candidate][scorer] = result
                for threshold, value in zip(THRESHOLDS, result['curve']):
                    val_rows.append(dict(backbone=name, candidates=candidate, scorer=scorer,
                                         threshold=float(threshold), G1=value))
    dump(out / 'VAL_SELECTION.json', dict(protocol=protocol, models=selected, audit=audit))
    selection_hash = sha(out / 'VAL_SELECTION.json')
    write_csv(out / 'VAL_SWEEPS.csv', val_rows)
    rows, primary, ranking, probabilities, rank_points, final_rows = [], [], [], [], [], []
    standard_pairs = None
    for split in ('standard', 'full'):
        gt = read_rows(root / 'ground_truth' / (split + '.jsonl.gz'))
        if split == 'standard':
            standard_pairs = {(g['vid'], g['query']) for g in gt}
        assert not ({(g['vid'], g['query']) for g in gt} & {(g['vid'], g['query']) for g in val_gt})
        y = np.array([bool(g['relevant_windows']) for g in gt])
        vids = np.array([g['vid'] for g in gt])
        for name in NAMES:
            data, predictions = {}, {}
            for method in METHODS:
                p = Path(manifest['backbones'][name].get('test_root', root)) / 'runs' / name / split / method
                done = json.loads((p / 'complete.json').read_text())
                metrics = json.loads((p / 'metrics.json').read_text())
                checkpoint_hash = done.get('checkpoint_sha256', metrics.get('provenance', {}).get('checkpoint_sha256'))
                assert checkpoint_hash == manifest['backbones'][name][method]['sha256']
                with gzip.open(p / 'predictions.jsonl.gz', 'rb') as stream:
                    assert hashlib.sha256(stream.read()).hexdigest() == done['predictions_sha256']
                predictions[method] = read_rows(p / 'predictions.jsonl.gz')
                data[method] = arrays(gt, predictions[method])
                score = data[method][0]
                bp, bn = float(np.mean((1-score[y])**2)), float(np.mean(score[~y]**2))
                probabilities.append(dict(split=split, backbone=name, method=method,
                    rows=len(gt), positive=int(y.sum()), videos=len(set(vids)),
                    mean_positive=float(score[y].mean()), mean_negative=float(score[~y].mean()),
                    Brier_positive=bp, Brier_negative=bn, Brier_balanced=(bp+bn)/2,
                    AUROC=100 * float(roc_auc_score(y, score))))
            cell_values = {}
            for candidate in METHODS:
                for scorer in METHODS:
                    threshold = selected[name][candidate][scorer]['threshold']
                    score, utility = data[scorer][0], data[candidate][1]
                    row = dict(split=split, backbone=name, candidates=candidate, scorer=scorer, threshold=threshold)
                    shared_threshold = selected[name]['gmr']['gmr']['threshold']
                    row['shared_GMR_threshold'] = shared_threshold
                    geq_threshold = selected[name]['geq_full']['geq_full']['threshold']
                    row['shared_GEQ_threshold'] = geq_threshold
                    for k in (1, 3):
                        per_query = values(score, utility[k], y, threshold)
                        cell_values[candidate, scorer, k] = per_query
                        row[f'G{k}'] = 100 * float(per_query.mean())
                        row[f'G{k}_at_05'] = 100 * float(values(score, utility[k], y, .5).mean())
                        shared = values(score, utility[k], y, shared_threshold)
                        row[f'G{k}_shared'] = 100 * float(shared.mean())
                        cell_values[candidate, scorer, k, 'shared'] = shared
                        shared_geq = values(score, utility[k], y, geq_threshold)
                        row[f'G{k}_shared_GEQ'] = 100 * float(shared_geq.mean())
                        cell_values[candidate, scorer, k, 'shared_GEQ'] = shared_geq
                    row['TPR'] = 100 * float((score[y] > threshold).mean())
                    row['TNR'] = 100 * float((score[~y] <= threshold).mean())
                    # Cross-check every selected-threshold cell using the official evaluator.
                    candidate_map = {p['qid']: p for p in predictions[candidate]}
                    score_map = {p['qid']: p['pred_exist_score'] for p in predictions[scorer]}
                    swapped = [dict(candidate_map[g['qid']], pred_exist_score=score_map[g['qid']]) for g in gt]
                    gated, _ = prepare_submission_for_gmiou(swapped, cls_threshold=threshold)
                    official = compute_G_mIoU(gated, gt, k_list=(1, 3))
                    for k in (1, 3):
                        assert abs(round(row[f'G{k}'], 2) - official[f'G-mIoU@{k}']) < 1e-9
                    gated, _ = prepare_submission_for_gmiou(swapped, cls_threshold=shared_threshold)
                    official_shared = compute_G_mIoU(gated, gt, k_list=(1, 3))
                    for k in (1, 3):
                        assert abs(round(row[f'G{k}_shared'], 2) - official_shared[f'G-mIoU@{k}']) < 1e-9
                    gated, _ = prepare_submission_for_gmiou(swapped, cls_threshold=geq_threshold)
                    official_geq = compute_G_mIoU(gated, gt, k_list=(1, 3))
                    for k in (1, 3):
                        assert abs(round(row[f'G{k}_shared_GEQ'], 2) - official_geq[f'G-mIoU@{k}']) < 1e-9
                    rows.append(row)
            base, geq = [next(r for r in rows if r['split']==split and r['backbone']==name and
                                r['candidates']==m and r['scorer']==m) for m in METHODS]
            record = dict(split=split, backbone=name, GMR_threshold=base['threshold'], GEQ_threshold=geq['threshold'])
            for k in (1, 3):
                delta = cell_values['geq_full', 'geq_full', k, 'shared'] - cell_values['gmr', 'gmr', k, 'shared']
                lo, hi = bootstrap(delta, vids)
                record.update({f'GMR_G{k}':base[f'G{k}_shared'], f'GEQ_G{k}':geq[f'G{k}_shared'],
                               f'delta_G{k}':100*float(delta.mean()), f'CI95_G{k}_low':lo, f'CI95_G{k}_high':hi,
                               f'GEQ_G{k}_independent':geq[f'G{k}'],
                               f'delta_G{k}_independent':geq[f'G{k}']-base[f'G{k}'],
                               f'GMR_G{k}_at_05':base[f'G{k}_at_05'], f'GEQ_G{k}_at_05':geq[f'G{k}_at_05']})
            primary.append(record)
            for scheme, suffix, cell_key, threshold in [
                ('GMR_Val', '_shared', 'shared', selected[name]['gmr']['gmr']['threshold']),
                ('GEQ_Val', '_shared_GEQ', 'shared_GEQ', selected[name]['geq_full']['geq_full']['threshold'])]:
                final_row = next((r for r in final_rows if r['backbone']==name and r['threshold_method']==scheme), None)
                if final_row is None:
                    final_row = dict(backbone=name, threshold_method=scheme, threshold=threshold)
                    final_rows.append(final_row)
                for k in (1,3):
                    final_row[f'{split}_GMR_G{k}'] = base[f'G{k}'+suffix]
                    final_row[f'{split}_GEQ_G{k}'] = geq[f'G{k}'+suffix]
                    delta = cell_values['geq_full','geq_full',k,cell_key]-cell_values['gmr','gmr',k,cell_key]
                    final_row[f'{split}_delta_G{k}'] = 100*float(delta.mean())
            subsets = [('all', np.ones(len(gt),dtype=bool))]
            if split == 'full':
                subsets.append(('additional_to_standard', np.array([(g['vid'],g['query']) not in standard_pairs for g in gt])))
            for subset, mask in subsets:
                yy = y[mask]
                for candidate in METHODS:
                    for k in (1, 3):
                        curves = {}
                        for scorer in METHODS:
                            x, g = rank_curve(data[scorer][0][mask], data[candidate][1][k][mask], yy)
                            area = float(np.sum(np.diff(x) * (g[:-1]+g[1:])/2))
                            curves[scorer] = (area, x, g)
                            ranking.append(dict(split=split, subset=subset, backbone=name, candidates=candidate,
                                scorer=scorer, k=k, area=100*area, diagnostic_max=100*float(g.max()),
                                AUROC=100*float(roc_auc_score(yy,data[scorer][0][mask])), rows=int(mask.sum())))
                            if subset=='all':
                                for fraction in np.arange(101)/100:
                                    rank_points.append(dict(split=split, backbone=name, candidates=candidate,
                                        scorer=scorer, k=k, accepted_fraction=float(fraction),
                                        G=100*float(np.interp(fraction,x,g))))
                        # A strictly increasing transform must preserve the full curve.
                        s=data['gmr'][0][mask]
                        x0,g0=rank_curve(s,data[candidate][1][k][mask],yy)
                        x1,g1=rank_curve(.1+.8*s,data[candidate][1][k][mask],yy)
                        assert np.array_equal(x0,x1) and np.array_equal(g0,g1)
            print('REPORTED',split,name,record['delta_G1'],flush=True)
    assert sha(out / 'VAL_SELECTION.json') == selection_hash
    for filename, records in [('PRIMARY.csv',primary),('CANDIDATE_SWAP.csv',rows),
                              ('PROBABILITIES.csv',probabilities),('RANKING.csv',ranking),('RANK_CURVES.csv',rank_points),
                              ('FINAL_TABLE.csv',final_rows)]:
        write_csv(out / filename, records)
    dump(out / 'AUDIT.json', dict(passed=True, selection_sha256=selection_hash,
        selected_threshold_cells=len(NAMES)*4, official_test_cell_checks=len(rows)*3,
        thresholds_selected_before_test_read=True, training_called=False,test_threshold_selection=False,
        source_test_provenance_sha256=sha(root/'PROVENANCE.json'),
        script_sha256=sha(__file__), validation_predictions=audit,
        caveats=['Historical five: six-patience/video75; new six: ten-patience/video128; single seed.',
                 'Full is 2970/2986 decodable queries and overlaps Standard.',
                 'Rank-curve area and maxima are retrospective diagnostics, not deployable Test scores.',
                 'Candidate swapping diagnoses these jointly fine-tuned models; it does not isolate architecture from training budget.']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('infer')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backbones', nargs='+', choices=NAMES, default=list(NAMES))
    p.add_argument('--device', default='cuda:0')
    p = sub.add_parser('report')
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--val-predictions', type=Path, required=True)
    p.add_argument('--tests', type=Path, default=Path('results/test_thresholds'))
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    (infer if args.command == 'infer' else report)(args)
