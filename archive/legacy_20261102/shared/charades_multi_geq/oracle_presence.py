"""Post-hoc perfect binary presence ceilings with fixed exported candidates."""
import math,json,fcntl,hashlib
from paths import KIT,OUT,dump,rows,sha
from metrics import _compute_set_iou_score


def compute_oracle_presence(predictions,ground_truth):
 assert len(predictions)==len(ground_truth)>0
 ids=[g['evaluation_id'] for g in ground_truth]
 assert len(set(ids))==len(ids) and [p['qid'] for p in predictions]==ids
 positive=[bool(g['relevant_windows']) for g in ground_truth]
 n_positive=sum(positive);n_negative=len(positive)-n_positive
 result=dict(definition='Perfect binary presence decisions; retain every positive query and reject every negative query; localization candidates and ranking stay fixed.',
  units='0–1',rows=len(positive),positive=n_positive,negative=n_negative,
  all_reject_gmiou=n_negative/len(positive),
  formula='(N_negative + sum_positive fixed-candidate set-IoU@k) / N_total',
  threshold_interpretation='Horizontal analytical ceiling; not a thresholded 0/1 score curve. At threshold=1 a bounded scored classifier still rejects all queries.',
  test_used_for_selection=False)
 for k in (1,3):
  values=[_compute_set_iou_score([w[:2] for w in p['pred_relevant_windows'][:k]],g['relevant_windows']) for p,g,yes in zip(predictions,ground_truth,positive) if yes]
  assert all(math.isfinite(x) and 0<=x<=1 for x in values)
  result[f'gmiou_{k}']=(n_negative+sum(values))/len(positive)
  result[f'positive_localization_mean_{k}']=sum(values)/n_positive if n_positive else None
 return result


def cached_oracle_presence(prediction_path,ground_truth_path,cache_path,expected_prediction_sha256):
 cache_path.parent.mkdir(parents=True,exist_ok=True)
 with cache_path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _cached_oracle_presence(prediction_path,ground_truth_path,cache_path,expected_prediction_sha256)


def _cached_oracle_presence(prediction_path,ground_truth_path,cache_path,expected_prediction_sha256):
 prediction_sha=sha(prediction_path);assert prediction_sha==expected_prediction_sha256
 source=dict(version=1,prediction_sha256=prediction_sha,ground_truth_sha256=sha(ground_truth_path),
  evaluator_sha256=sha(KIT/'vendor/official_eval/metrics.py'))
 if cache_path.exists():
  existing=json.loads(cache_path.read_text())
  if existing.get('source')==source:return existing
 pred_payload=prediction_path.read_bytes();gt_payload=ground_truth_path.read_bytes()
 assert hashlib.sha256(pred_payload).hexdigest()==prediction_sha
 assert hashlib.sha256(gt_payload).hexdigest()==source['ground_truth_sha256']
 result=compute_oracle_presence([json.loads(x) for x in pred_payload.splitlines() if x.strip()],[json.loads(x) for x in gt_payload.splitlines() if x.strip()]);result['source']=source
 dump(cache_path,result)
 return result


def selected_val_reference(directory,cache_path,history=None):
 """Only the exported current Val-best is available; never invent epoch oracles."""
 history=history if history is not None else rows(directory/'history.jsonl') if (directory/'history.jsonl').exists() else []
 if not history:return None
 epoch=history[-1].get('best_epoch')
 selected=next((r for r in history if r['epoch']==epoch),None)
 if not selected:return None
 pred=directory/'best_val_predictions.jsonl';metrics=directory/'best_val_metrics.json'
 if not pred.exists() or not metrics.exists():return None
 for _ in range(2):
  payload=metrics.read_bytes();metrics_sha=hashlib.sha256(payload).hexdigest();values=json.loads(payload)
  if selected.get('mAP') is None or abs(values['mAP']-selected['mAP'])>1e-11:return None
  if '_geq_' in directory.name and abs(values['selection_mean']-selected['selection'])>1e-11:return None
  prediction_sha=sha(pred)
  try:oracle=cached_oracle_presence(pred,OUT/'data/val.jsonl',cache_path,prediction_sha)
  except AssertionError:
   if sha(pred)!=prediction_sha or sha(metrics)!=metrics_sha:continue
   raise
  if sha(pred)!=prediction_sha or sha(metrics)!=metrics_sha:continue
  return dict(epoch=epoch,reference_kind='Current Val-best only; other epochs have no retained prediction exports.',
   oracle_presence=oracle,mAP_percent=values['mAP'],gmiou_1_mean=values['selection_mean'],
   gmiou_3_mean=sum(values['grid'][i]['gmiou_3'] for i in [40,50,60])/3,
   source=dict(metrics_sha256=metrics_sha,prediction_sha256=prediction_sha,source_split='val'))
 return None
