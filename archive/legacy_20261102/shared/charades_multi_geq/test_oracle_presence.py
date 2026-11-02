"""Analytic examples, arbitrary gate bounds and immutable-source cache checks."""
import copy,itertools,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import oracle_presence
from oracle_presence import compute_oracle_presence,cached_oracle_presence,selected_val_reference
from metrics import _compute_set_iou_score
from paths import jsonl,sha,dump


def example():
 gt=[dict(qid=str(i),evaluation_id=str(i),relevant_windows=[[0.,10.]] if i<2 else []) for i in range(4)]
 pred=[dict(qid=str(i),pred_exist_score=0.1*i,pred_relevant_windows=([[0.,10.,.9],[10.,20.,.5],[20.,30.,.2]] if i==0 else [[0.,5.,.7]])) for i in range(4)]
 return pred,gt


class OraclePresenceTests(unittest.TestCase):
 def test_hand_computed_single_gt_ceilings_and_score_invariance(self):
  pred,gt=example();oracle=compute_oracle_presence(pred,gt)
  self.assertEqual(oracle['gmiou_1'],.875)
  self.assertAlmostEqual(oracle['gmiou_3'],(2+1/3+.5)/4)
  changed=copy.deepcopy(pred)
  for p in changed:p['pred_exist_score']=1-p['pred_exist_score']
  self.assertEqual(oracle,compute_oracle_presence(changed,gt))

 def test_every_possible_binary_gate_is_bounded(self):
  pred,gt=example();oracle=compute_oracle_presence(pred,gt)
  for k in (1,3):
   for gate in itertools.product([False,True],repeat=4):
    score=sum(_compute_set_iou_score([w[:2] for w in p['pred_relevant_windows'][:k]] if keep else [],g['relevant_windows']) for p,g,keep in zip(pred,gt,gate))/4
    self.assertLessEqual(score,oracle[f'gmiou_{k}']+1e-12)

 def test_unbalanced_and_empty_localization_cases(self):
  pred,gt=example();value=compute_oracle_presence(pred[:3],gt[:3]);self.assertAlmostEqual(value['gmiou_1'],(1+1+.5)/3)
  self.assertEqual(compute_oracle_presence(pred[2:],gt[2:])['gmiou_1'],1.)
  for p in pred:p['pred_relevant_windows']=[]
  self.assertEqual(compute_oracle_presence(pred,gt)['gmiou_1'],.5)

 def test_identity_and_cache_source_integrity(self):
  pred,gt=example()
  with self.assertRaises(AssertionError):compute_oracle_presence(list(reversed(pred)),gt)
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);pp=d/'pred.jsonl';gp=d/'gt.jsonl';cache=d/'oracle.json';jsonl(pp,pred);jsonl(gp,gt)
   first=cached_oracle_presence(pp,gp,cache,sha(pp));self.assertEqual(first,cached_oracle_presence(pp,gp,cache,sha(pp)))
   pred[0]['pred_relevant_windows']=[];jsonl(pp,pred)
   with self.assertRaises(AssertionError):cached_oracle_presence(pp,gp,cache,first['source']['prediction_sha256'])
   second=cached_oracle_presence(pp,gp,cache,sha(pp));self.assertLess(second['gmiou_1'],first['gmiou_1'])
   self.assertNotEqual(second['source'],first['source'])

 def test_current_best_reference_and_incomplete_export_are_distinct(self):
  pred,gt=example()
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);d=root/'gmr';d.mkdir();jsonl(root/'data/val.jsonl',gt);jsonl(d/'best_val_predictions.jsonl',pred)
   history=[dict(epoch=1,best_epoch=1,mAP=1.,selection=1.),dict(epoch=2,best_epoch=2,mAP=2.,selection=2.)]
   metrics=dict(mAP=2.,selection_mean=.6,grid=[dict(gmiou_3=.3) for _ in range(101)])
   dump(d/'best_val_metrics.json',metrics)
   with patch.object(oracle_presence,'OUT',root):
    reference=selected_val_reference(d,root/'cache/oracle.json',history)
    self.assertEqual(reference['epoch'],2);self.assertEqual(reference['oracle_presence']['gmiou_1'],.875)
    self.assertAlmostEqual(reference['gmiou_3_mean'],.3)
    self.assertIn('other epochs',reference['reference_kind'])
    dump(d/'best_val_metrics.json',dict(metrics,mAP=3.))
    self.assertIsNone(selected_val_reference(d,root/'cache/oracle.json',history))

 def test_presence_oracle_leaves_ungated_metrics_unchanged(self):
  from plot_report import metric_oracle
  record=dict(mAP_percent=31.,mR1_percent=42.,mR5_percent=55.,AUROC_percent=81.,gmiou_1_mean=.6,gmiou_3_mean=.5,oracle_gmiou_1=.75,oracle_gmiou_3=.58)
  for metric in ['mAP_percent','mR1_percent','mR5_percent']:self.assertEqual(metric_oracle(record,metric),record[metric])
  self.assertEqual(metric_oracle(record,'AUROC_percent'),100.)
  self.assertEqual(metric_oracle(record,'gmiou_1_mean'),.75)
  self.assertEqual(metric_oracle(record,'gmiou_3_mean'),.58)

 def test_relative_oracle_uses_observed_gmr_and_preserves_negative_gains(self):
  from plot_report import relative_records
  baseline=[dict(threshold=i/100,gmiou_1=.5+.1*i/100) for i in range(101)]
  multi=[dict(threshold=i/100,gmiou_1=.55+.03*i/100) for i in range(101)]
  snap=dict(methods=['gmr','multi'],series=dict(gmr=dict(grid=baseline,oracle_presence=dict(gmiou_1=.75)),multi=dict(grid=multi,oracle_presence=dict(gmiou_1=.8))))
  records=relative_records(snap);self.assertEqual(len(records),202)
  for row in records:
   self.assertLessEqual(row['actual_relative_to_gmr'],row['oracle_relative_to_gmr'])
   if row['method']=='gmr':self.assertEqual(row['actual_relative_to_gmr'],0.)
  chosen=next(r for r in records if r['method']=='multi' and r['threshold']==.5)
  self.assertAlmostEqual(chosen['actual_relative_to_gmr'],.015)
  self.assertAlmostEqual(chosen['oracle_relative_to_gmr'],.25)
  self.assertLess(records[-1]['actual_relative_to_gmr'],0.)


if __name__=='__main__':unittest.main()
