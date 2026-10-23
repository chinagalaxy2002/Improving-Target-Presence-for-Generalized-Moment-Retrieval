import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from qcd import QCD,masks,replace_count


class QCDTests(unittest.TestCase):
    def test_example_and_union(self):
        gt,p=masks([[20,28],[60,68],[110,118]],75)
        self.assertEqual((gt.sum(),p.sum(),replace_count(75,gt,p,.3)),(12,18,19))
        gt,p=masks([[0,4],[2,6],[148,150]],75)
        self.assertEqual((gt.sum(),p.sum()),(4,6))
        gt,p=masks([[150,152]],75)
        self.assertEqual(p.sum(),0)

    def test_all_six_gt_padding_labels_tef_and_sync(self):
        windows=[[2,4],[12,14],[22,24],[32,34],[42,44],[52,54]]
        rows=[dict(qid=1,vid='a',query='query',dataset_source='worldcup2022',relevant_windows=windows),
              dict(qid=2,vid='b',query='other',dataset_source='worldcup2022',relevant_windows=[]),
              dict(qid=3,vid='c',query='third',dataset_source='other',relevant_windows=[[0,2]])]
        a=torch.zeros(35,2818);b=torch.arange(35)[:,None].expand(35,2818).float()+10
        dataset=SimpleNamespace(clip_len=2,preloaded_data=[(r,dict(video_feat=v)) for r,v in zip(rows,[a,b,a])])
        cfg=dict(enabled=True,query_filter=False,scope='all',context_size=1,replace_ratio=.3,p_aug=1,seed=20)
        aug=QCD(dataset,cfg)
        tensor=torch.stack([a,b,a]);tensor[0,30:]=999
        features=dict(src_vid=tensor,src_vid_mask=torch.tensor([[1]*30+[0]*5,[1]*35,[1]*35]))
        before=tensor.clone();metadata=copy.deepcopy(rows)
        rng_before=torch.get_rng_state().clone()
        records=aug.apply(features,rows)
        self.assertEqual(rows,metadata)
        self.assertTrue(torch.equal(rng_before,torch.get_rng_state()))
        self.assertEqual(len(records),1)
        r=records[0];gt,p=masks(windows,30)
        self.assertTrue(torch.equal(tensor[0,:30][p],before[0,:30][p]))
        self.assertTrue(torch.equal(tensor[:,:,2816:],before[:,:,2816:]))
        self.assertTrue(torch.equal(tensor[0,30:],before[0,30:]))
        self.assertTrue(torch.equal(tensor[1:],before[1:]))
        self.assertTrue(torch.equal(tensor[0,r['dst'],:2816],b[r['src'],:2816]))
        self.assertNotIn(26,r['dst']) # sixth GT, beyond max_windows=5
        self.assertEqual(len(r['dst']),replace_count(30,gt,p,.3))

    def test_full_protection(self):
        gt,p=masks([[0,150]],75)
        self.assertEqual(replace_count(75,gt,p,.3),0)


if __name__=='__main__':unittest.main()
