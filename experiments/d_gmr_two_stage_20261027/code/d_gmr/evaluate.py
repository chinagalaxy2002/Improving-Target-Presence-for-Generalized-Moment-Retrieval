"""Official evaluation for D-GMR with plurality-aware metrics and diagnostic coverage."""
import json
import time
import uuid
from pathlib import Path

import torch

from gemr.engine.evaluate import coverage, score
from d_gmr.common import FROZEN_RUNS, RUNS, dump, rows


def original_predictions(backbone, split):
    published = Path(__file__).resolve().parents[2]/'results'/backbone/split/'original_gmr_predictions.jsonl'
    if published.exists():
        return rows(published), published
    if split == 'val':
        path = FROZEN_RUNS/backbone/'val_results/epoch_001/native_predictions.jsonl'
    else:
        completed = json.loads((FROZEN_RUNS/backbone/'TEST_COMPLETE.json').read_text())
        path = Path(completed['output'])/'native_predictions.jsonl'
    values = rows(path)
    return values, path


@torch.no_grad()
def predict(model, data, batch_size, max_batches=None):
    model.eval()
    adapter = model.adapter
    reranked, joint_native, joint_geq, details = [], [], [], []
    begun = time.monotonic()
    for number, start in enumerate(range(0, len(data), batch_size)):
        if max_batches is not None and number >= max_batches:
            break
        metas, inputs, _ = adapter.collate(
            [data[index] for index in range(start, min(start+batch_size, len(data)))])
        output = model(inputs)
        backbone = output['frozen']
        for batch, meta in enumerate(metas):
            valid = backbone['proposal_valid'][batch]
            raw = adapter.export_ranked_proposals(backbone['proposals'][batch, valid], meta, apply_pat=False)
            refined = adapter.export_ranked_proposals(output['ranked_proposals'][batch, valid], meta)
            geq_probability = float(output['presence_probability'][batch])
            geq_logit = float(output['presence_logit'][batch])
            native_logit = backbone['native_presence_logit'][batch].reshape(-1)[0]
            native_probability = float(native_logit.sigmoid())
            base = dict(qid=meta['qid'], vid=meta['vid'], query=meta['query'])
            reranked.append(dict(base, pred_relevant_windows=refined,
                                 pred_relevant_windows_pre_exist=refined,
                                 pred_exist_score=geq_probability,
                                 pred_exist_logit_raw=geq_logit))
            joint_native.append(dict(base, pred_relevant_windows=raw,
                                     pred_relevant_windows_pre_exist=raw,
                                     pred_exist_score=native_probability,
                                     pred_exist_logit_raw=float(native_logit)))
            joint_geq.append(dict(base, pred_relevant_windows=raw,
                                  pred_relevant_windows_pre_exist=raw,
                                  pred_exist_score=geq_probability,
                                  pred_exist_logit_raw=geq_logit))
            ids = output['candidate_ids'][batch]
            details.append(dict(
                qid=meta['qid'], candidate_ids=ids.tolist(),
                valid=output['selected_valid'][batch].tolist(),
                proposals=output['selected_proposals'][batch].tolist(),
                refined_scores=output['ranked_proposals'][batch, ids, 2].tolist(),
                quality=output['quality_logit'][batch].sigmoid().tolist(),
                residual=output['score_residual'][batch].tolist(),
                evidence=output['evidence_mean'][batch].tolist(),
                contrast=output['evidence_contrast'][batch].tolist(),
                competition_attention=output['competition_attention'][batch].tolist(),
                cooperation_attention=output['cooperation_attention'][batch].tolist(),
                relation_gates=output['relation_gates'][batch].tolist()))
        if number % 50 == 0:
            print(json.dumps(dict(stage='inference', backbone=adapter.name,
                                  processed=len(reranked), total=len(data))), flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return reranked, joint_native, joint_geq, details, time.monotonic()-begun


def save_evaluation(model, data, split, output, batch_size, threshold=.4, checkpoint=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    predictions, native, geq, details, seconds = predict(model, data, batch_size)
    original, original_path = original_predictions(model.adapter.name, split)
    wanted = {row['qid'] for row in data.data}
    original = [row for row in original if row['qid'] in wanted]
    assert len(original) == len(wanted) and {row['qid'] for row in original} == wanted
    artifacts = [('predictions', predictions), ('joint_native_predictions', native),
                 ('joint_geq_predictions', geq), ('proposal_diagnostics', details)]
    for name, values in artifacts:
        with (output/f'{name}.jsonl').open('x') as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
    report = score(predictions, data.data, threshold)
    native_report = score(native, data.data, threshold)
    geq_report = score(geq, data.data, threshold)
    original_report = score(original, data.data, threshold)
    delta = {key: report['brief'][key]-original_report['brief'][key]
             for key in report['brief']}
    coverage_report = dict(original_gmr=coverage(original, data.data),
                           joint_native=coverage(native, data.data),
                           d_gmr=coverage(predictions, data.data))
    for name, value in [('metrics', report), ('joint_native_metrics', native_report),
                        ('joint_geq_metrics', geq_report),
                        ('original_gmr_metrics', original_report),
                        ('coverage', coverage_report)]:
        dump(output/f'{name}.json', value, exclusive=True)
    dump(output/'evaluation.json', dict(
        checkpoint=str(checkpoint), split=split, threshold=threshold, seconds=seconds,
        queries=len(data), seconds_per_query=seconds/len(data), delta_vs_original_gmr=delta,
        localization_gate=False, original_gmr_predictions=str(original_path),
        selection_split='val', raw_predictions_retained=True), exclusive=True)
    keys = report['brief'].keys()
    lines = ['# D-GMR evaluation', '', f'Checkpoint: `{checkpoint}`', '',
             f'Split: {split}; presence threshold: {threshold}; positive localization has no presence gate.', '',
             '| Metric | Original GMR | Joint native | Joint GEQ | D-GMR | Δ vs original GMR (pp) |',
             '|---|---:|---:|---:|---:|---:|']
    for key in keys:
        lines.append(f"| {key} | {original_report['brief'][key]:.4f} | "
                     f"{native_report['brief'][key]:.4f} | {geq_report['brief'][key]:.4f} | "
                     f"{report['brief'][key]:.4f} | {delta[key]:+.4f} |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report, original_report, native_report, geq_report


def standalone_directory(run, split, epoch, threshold):
    tag = time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    return Path(run)/f'{split}_results'/f'epoch_{epoch:03d}_tau_{threshold:g}_{tag}'
