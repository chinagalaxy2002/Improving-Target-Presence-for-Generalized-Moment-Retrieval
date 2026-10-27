"""Official evaluation, frozen prediction rescoring and coverage diagnostics."""
import json
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import torch

# NumPy 2.0 removed ``trapz``; the repository's unchanged official evaluator
# still calls that name.  ``trapezoid`` implements the same composite
# trapezoidal integration and keeps the historical evaluator source intact.
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid

from gemr.common import REFERENCE, dump, rows
from gemr.models.temporal_relation import temporal_iou

# Import before native packages introduce their own eval/standalone_eval packages.
sys.path.insert(0, str(REFERENCE))
from official_eval.eval_main import evaluate_gmr

METRICS = ('mAP', 'mR@1', 'mR@5', 'mR+@5', 'AUROC', 'Rej-F1@0.4', 'G-mIoU@1', 'G-mIoU@3', 'N-acc')


def score(predictions, ground_truth, threshold=.4):
    assert len(predictions) == len(ground_truth)
    assert len({p['qid'] for p in predictions}) == len(predictions)
    assert {p['qid'] for p in predictions} == {g['qid'] for g in ground_truth}
    report = evaluate_gmr(predictions, ground_truth, cls_thresholds=(threshold,),
                          gmiou_cls_threshold=threshold, map_num_workers=0, verbose=False)
    negatives = {g['qid'] for g in ground_truth if not g['relevant_windows']}
    report['brief']['N-acc'] = 100*sum(p['pred_exist_score'] <= threshold for p in predictions if p['qid'] in negatives)/max(1, len(negatives))
    return report


def coverage(predictions, gt):
    by_qid = {p['qid']: p for p in predictions}
    result = {}
    for threshold in (.5, .75):
        counts = Counter()
        for meta in gt:
            if not meta['relevant_windows']:
                continue
            count = len(meta['relevant_windows'])
            p = torch.tensor(by_qid[meta['qid']]['pred_relevant_windows'][:10]).reshape(-1, 3)
            gt_xy = torch.tensor(meta['relevant_windows'], dtype=torch.float32)
            iou = temporal_iou(p[:, :2], gt_xy)
            covered = iou.max(0).values >= threshold if len(p) else torch.zeros(count, dtype=torch.bool)
            counts['positive_queries'] += 1
            counts['gt_instances'] += count
            counts['covered_gt_instances'] += int(covered.sum())
            counts['top10_covers_any_gt'] += int(covered.any())
            counts['top10_covers_all_gt'] += int(covered.all())
            if len(p):
                best, owner = iou.max(1)
                hits = owner[best >= threshold]
                counts['duplicate_hits_same_gt'] += len(hits)-len(hits.unique())
                counts['best_quality_candidate_below_rank1'] += int(iou.max(1).values.argmax() > 0)
                counts['top1_hits'] += int(best[0] >= threshold)
        result[str(threshold)] = dict(counts)
    return result


@torch.no_grad()
def predict(model, data, batch_size, max_batches=None):
    model.eval()
    adapter = model.adapter
    predictions, raw_predictions, details = [], [], []
    begun = time.monotonic()
    for number, start in enumerate(range(0, len(data), batch_size)):
        if max_batches is not None and number >= max_batches:
            break
        metas, inputs = adapter.collate([data[i] for i in range(start, min(start+batch_size, len(data)))])
        out = model(inputs)
        frozen = out['frozen']
        for b, meta in enumerate(metas):
            valid = frozen['proposal_valid'][b]
            raw = adapter.export_ranked_proposals(frozen['proposals'][b, valid], meta)
            refined = adapter.export_ranked_proposals(out['ranked_proposals'][b, valid], meta)
            prob, logit = float(out['presence_probability'][b]), float(out['presence_logit'][b])
            base = dict(qid=meta['qid'], vid=meta['vid'], query=meta['query'],
                        pred_exist_score=prob, pred_exist_logit_raw=logit)
            # Localization is always ungated. The official evaluator applies
            # the existence gate only to G-mIoU, retaining all positive queries.
            predictions.append(dict(base, pred_relevant_windows=refined, pred_relevant_windows_pre_exist=refined))
            native_logit = frozen['native_presence_logit']
            native_prob = float(native_logit[b].sigmoid().reshape(-1)[0]) if native_logit is not None else None
            assert native_prob is not None, 'Native GMR presence must be retained'
            raw_predictions.append(dict(base, pred_relevant_windows=raw, pred_relevant_windows_pre_exist=raw,
                                        pred_exist_score=native_prob))
            ids = out['candidate_ids'][b]
            details.append(dict(qid=meta['qid'], candidate_ids=ids.tolist(),
                                valid=out['selected_valid'][b].tolist(),
                                proposals=out['selected_proposals'][b].tolist(),
                                refined_scores=out['ranked_proposals'][b, ids, 2].tolist(),
                                quality=out['quality_logit'][b].sigmoid().tolist(),
                                residual=out['score_residual'][b].tolist(),
                                evidence=out['evidence_mean'][b].tolist(),
                                contrast=out['evidence_contrast'][b].tolist(),
                                competition_attention=out['competition_attention'][b].tolist(),
                                cooperation_attention=out['cooperation_attention'][b].tolist(),
                                relation_gates=out['relation_gates'][b].tolist()))
        if number % 50 == 0:
            print(json.dumps(dict(stage='inference', backbone=adapter.name, processed=len(predictions), total=len(data))), flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return predictions, raw_predictions, details, time.monotonic()-begun


def save_evaluation(model, data, output, batch_size, threshold=.4, checkpoint=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    pred, raw, details, seconds = predict(model, data, batch_size)
    for name, values in [('predictions', pred), ('native_predictions', raw), ('proposal_diagnostics', details)]:
        with (output/f'{name}.jsonl').open('x') as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
    report = score(pred, data.data, threshold)
    native_report = score(raw, data.data, threshold)
    # Separate unchanged-GEQ presence from original GMR presence.
    geq_raw = [dict(r, pred_exist_score=p['pred_exist_score']) for r, p in zip(raw, pred)]
    geq_report = score(geq_raw, data.data, threshold)
    delta = {k: report['brief'][k]-native_report['brief'][k] for k in report['brief']}
    coverage_report = dict(native=coverage(raw, data.data), gemr=coverage(pred, data.data))
    dump(output/'metrics.json', report, exclusive=True)
    dump(output/'native_metrics.json', native_report, exclusive=True)
    dump(output/'global_only_metrics.json', geq_report, exclusive=True)
    dump(output/'coverage.json', coverage_report, exclusive=True)
    dump(output/'evaluation.json', dict(checkpoint=str(checkpoint), threshold=threshold,
                                       seconds=seconds, queries=len(data), seconds_per_query=seconds/len(data),
                                       delta_vs_native=delta, localization_gate=False,
                                       selection_split='val', raw_predictions_retained=True), exclusive=True)
    lines = ['# GEMR evaluation', '', f'Checkpoint: `{checkpoint}`', '',
             f'Presence threshold: {threshold}; positive localization evaluated without a presence gate.', '',
             '| Metric | Native GMR | Frozen Global-only | GEMR | Δ vs GMR (pp) |', '|---|---:|---:|---:|---:|']
    for key in report['brief']:
        lines.append(f"| {key} | {native_report['brief'][key]:.4f} | {geq_report['brief'][key]:.4f} | {report['brief'][key]:.4f} | {delta[key]:+.4f} |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report, native_report


def standalone_directory(run, split, epoch, threshold):
    tag = time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    return Path(run)/f'{split}_results'/f'epoch_{epoch:03d}_tau_{threshold:g}_{tag}'
