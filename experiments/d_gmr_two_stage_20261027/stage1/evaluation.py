"""Native export, official metrics, and independent ranking/recall diagnostics."""
import copy, json, sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from common import OLD, dump, rows
sys.path.insert(0, str(OLD))
from official_eval.eval_main import evaluate_gmr

def iou(pred, gt):
    p = np.asarray(pred, dtype=float).reshape(-1, 2); g = np.asarray(gt, dtype=float).reshape(-1, 2)
    inter = np.maximum(0, np.minimum(p[:, None, 1], g[None, :, 1])-np.maximum(p[:, None, 0], g[None, :, 0]))
    union = np.maximum(0, p[:, 1]-p[:, 0])[:, None]+np.maximum(0, g[:, 1]-g[:, 0])[None]-inter
    return inter/np.maximum(union, 1e-8)

def diagnostics(out, metas):
    from pv import quality_targets
    targets = quality_targets(out['candidate_spans'], out['candidate_valid'], metas)
    result = []
    for b, m in enumerate(metas):
        valid = out['candidate_valid'][b]
        row = dict(qid=m['qid'], candidate_indices=out['candidate_indices'][b].tolist(),
                   candidate_valid=valid.tolist(), candidate_spans=out['candidate_spans'][b].detach().cpu().tolist(),
                   target_iou=targets[b].tolist())
        if 'quality_score' in out: row['quality_score'] = out['quality_score'][b].tolist()
        result.append(row)
    return result

@torch.no_grad()
def predict(adapter, model, data, limit=None):
    model.eval(); extra = []; pred = []
    if adapter.name == 'flashvtg':
        from torch.utils.data import DataLoader, Subset
        from training.flash_vtg_gmr.inference import compute_mr_results, post_processing_mr_nms
        class Bridge(torch.nn.Module):
            def forward(self, **inputs):
                meta = inputs['targets']['label']
                d = inputs['src_vid'].new_tensor([m['duration'] for m in meta])
                out = model(durations=d, **inputs)
                extra.extend(diagnostics(out, meta))
                logits.append(float(out['pred_exist_logits'][0]))
                return out
        logits = []
        selected = data if limit is None else Subset(data, range(limit))
        loader = DataLoader(selected, batch_size=1, collate_fn=adapter.data_module.start_end_collate)
        bridge = Bridge(); bridge.eval()
        pred, _ = compute_mr_results(bridge, loader, adapter.opt, None, None, None)
        for row, logit in zip(pred, logits):
            row['pred_exist_logit_raw'] = logit
            row['pred_exist_score'] = float(torch.tensor(logit).sigmoid())
        if adapter.opt.nms_thd != -1:
            pred = post_processing_mr_nms(pred, adapter.opt.nms_thd, adapter.opt.max_before_nms, adapter.opt.max_after_nms, adapter.opt.nms_type)
    else:
        count = len(data) if limit is None else min(limit, len(data))
        for start in range(0, count, adapter.bs):
            metas, inputs, _ = adapter.collate([data[i] for i in range(start, min(start+adapter.bs, count))])
            out = model(**inputs); extra.extend(diagnostics(out, metas))
            for b, m in enumerate(metas):
                # Preserve native clipping/rounding, including degenerate spans:
                # official evaluator applies its existing invalid-window rule.
                spans = out['candidate_seconds'][b].detach().cpu().clone()
                spans[:, :2] = (spans[:, :2]/2).round()*2
                spans[:, :2].clamp_(0, m['duration'])
                spans = spans[spans[:, 2].argsort(descending=True)]
                precision = 4 if adapter.name in ('moment_detr','qd_detr','eatr') else None
                win = spans.tolist()
                if precision is not None: win = [[round(float(x), precision) for x in w] for w in win]
                pred.append(dict(qid=m['qid'], vid=m['vid'], query=m['query'], pred_relevant_windows=win,
                                 pred_relevant_windows_pre_exist=win, pred_exist_logit_raw=float(out['pred_exist_logits'][b]),
                                 pred_exist_score=float(out['pred_exist_score'][b])))
    return pred, extra

def evaluate(pred, gt, directory, split, diag=None, baseline=None):
    assert len(pred) == len(gt) == len({p['qid'] for p in pred})
    mapping = {p['qid']:p for p in pred}; assert set(mapping) == {g['qid'] for g in gt}
    pred = [mapping[g['qid']] for g in gt]
    y = np.array([bool(g['relevant_windows']) for g in gt]); s = np.array([p['pred_exist_score'] for p in pred])
    reports = {}
    for tau in (.4, .5, .6):
        official = evaluate_gmr(pred, gt, cls_thresholds=(tau,), gmiou_cls_threshold=tau, map_num_workers=0, verbose=False)
        gate = s > tau; tp = int((gate & y).sum()); tn = int((~gate & ~y).sum()); fp = int((gate & ~y).sum()); fn = int((~gate & y).sum())
        b = official['brief']
        assert abs(b['AUROC'] - 100*roc_auc_score(y, s)) <= .011
        f1 = 100*2*tn/max(1, 2*tn+fp+fn)
        assert abs(b[f'Rej-F1@{tau}'] - f1) <= .011
        b.update(AUPRC=100*float(average_precision_score(y, s)), **{'Pos-Recall':100*tp/int(y.sum()), 'TNR':100*tn/int((~y).sum()), 'N-acc':100*tn/int((~y).sum())})
        reports[str(tau)] = official
    recall = {}; raw = copy.deepcopy(pred)
    for r in raw: r['pred_relevant_windows'] = r['pred_relevant_windows_pre_exist']
    raw_official = evaluate_gmr(raw, gt, cls_thresholds=(.4,), gmiou_cls_threshold=.4, map_num_workers=0, verbose=False)['brief']
    # Independent positive query hit recall: any GT at tIoU >= .5, no gate.
    for k in (1, 5, 8):
        hits = {}
        for p,g in zip(pred,gt):
            if not g['relevant_windows']: continue
            windows = [w[:2] for w in p['pred_relevant_windows'][:k] if w[1]>w[0]]
            hits[str(g['qid'])] = bool(len(windows) and iou(windows, g['relevant_windows']).max() >= .5)
        recall[str(k)] = dict(recall=100*float(np.mean(list(hits.values()))), hits=hits)
    score_d = {name:dict(mean=float(s[mask].mean()), std=float(s[mask].std()), quantiles=np.quantile(s[mask], [0,.1,.25,.5,.75,.9,1]).tolist(), scores=s[mask].tolist()) for name,mask in [('positive',y),('negative',~y)]}
    details = dict(positive_only_localization=recall, score_distribution=score_d,
                   mean_score_separation=float(s[y].mean()-s[~y].mean()), raw_localization=raw_official,
                   metric_verification='Independent sklearn AUROC and rejection confusion/F1 match official; ranking uses GEQ only')
    if diag:
        q, t, oracle_query, oracle_moment = [], [], [], []
        dm = {d['qid']:d for d in diag}
        for g in gt:
            d = dm[g['qid']]; valid = np.array(d['candidate_valid'], bool)
            if 'quality_score' in d:
                q.extend(np.array(d['quality_score'])[valid].tolist()); t.extend(np.array(d['target_iou'])[valid].tolist())
            if g['relevant_windows']:
                spans = np.array(d['candidate_spans'])[valid, :2]
                ov = iou(spans,g['relevant_windows'])
                best = ov.max(0) if len(spans) else np.zeros(len(g['relevant_windows']))
                oracle_query.append(bool((best>=.5).any())); oracle_moment.append(float((best>=.5).mean()))
        details['top8_oracle'] = dict(query_recall_iou05=100*float(np.mean(oracle_query)), mean_gt_coverage_iou05=100*float(np.mean(oracle_moment)))
        details['pv_quality_pearson'] = float(np.corrcoef(q,t)[0,1]) if len(q)>1 and np.std(q)>0 and np.std(t)>0 else None
    if baseline:
        old = {str(p['qid']):p for p in baseline}
        comparisons = {}
        for k in (1,5):
            lost, restored = [], []
            for g in gt:
                if not g['relevant_windows']: continue
                key = str(g['qid']); windows = [w[:2] for w in old[key]['pred_relevant_windows'][:k] if w[1]>w[0]]
                before = bool(windows and iou(windows,g['relevant_windows']).max()>=.5); after = recall[str(k)]['hits'][key]
                if before and not after: lost.append(g['qid'])
                if after and not before: restored.append(g['qid'])
            comparisons[str(k)] = dict(lost_count=len(lost), restored_count=len(restored), lost_qids=lost, restored_qids=restored)
        details['vs_gmr_localization_hits'] = comparisons
        details['vs_gmr_presence'] = {str(tau):dict(lost_qids=[g['qid'] for p,g in zip(pred,gt) if g['relevant_windows'] and old[str(g['qid'])]['pred_exist_score']>tau and p['pred_exist_score']<=tau], restored_qids=[g['qid'] for p,g in zip(pred,gt) if g['relevant_windows'] and old[str(g['qid'])]['pred_exist_score']<=tau and p['pred_exist_score']>tau]) for tau in (.4,.5,.6)}
    for report in reports.values():
        report['brief']['Positive-only Localization Recall@5/IoU0.5'] = recall['5']['recall']
    from independent_metrics import verify
    verification = verify(pred, gt, reports)
    details['independent_metric_verification'] = verification
    directory.mkdir(parents=True, exist_ok=True)
    (directory/f'{split}_predictions.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pred))
    dump(directory/f'{split}_metrics.json', reports); dump(directory/f'{split}_diagnostics.json', details)
    if diag: dump(directory/f'{split}_candidates.json', diag)
    return reports
