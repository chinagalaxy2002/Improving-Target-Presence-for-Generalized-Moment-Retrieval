"""Analytic examples, arbitrary gate bounds and immutable-source cache checks."""
import copy,itertools,json,tempfile,unittest
from pathlib import Path
from oracle_presence import compute_oracle_presence,cached_oracle_presence
from metrics import _compute_set_iou_score
from paths import jsonl,sha


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


if __name__=='__main__':unittest.main()
