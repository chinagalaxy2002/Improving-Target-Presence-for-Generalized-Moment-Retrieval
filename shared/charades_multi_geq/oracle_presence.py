"""Post-hoc perfect binary presence ceilings with fixed exported candidates."""
import math
from paths import KIT,dump,rows,sha
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
 import json
 prediction_sha=sha(prediction_path);assert prediction_sha==expected_prediction_sha256
 source=dict(version=1,prediction_sha256=prediction_sha,ground_truth_sha256=sha(ground_truth_path),
  evaluator_sha256=sha(KIT/'vendor/official_eval/metrics.py'))
 if cache_path.exists():
  existing=json.loads(cache_path.read_text())
  if existing.get('source')==source:return existing
 result=compute_oracle_presence(rows(prediction_path),rows(ground_truth_path));result['source']=source
 dump(cache_path,result)
 return result
