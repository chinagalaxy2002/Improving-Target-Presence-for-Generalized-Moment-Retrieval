"""Executable edge-case checks: IoU multi-GT, invalids, padding and GRU axes."""
import unittest
import torch
from pv import PostVerification, quality_targets, representation_loss, losses

class PVTests(unittest.TestCase):
    def test_max_iou_multiple_gt_and_negative(self):
        spans=torch.tensor([[[0.,4.,.9],[0.,4.,.8],[8.,12.,.7],[5.,5.,.5]],[[0.,4.,.9],[8.,12.,.7],[3.,2.,.5],[0.,0.,0.]]])
        valid=spans[...,1]>spans[...,0]
        target=quality_targets(spans,valid,[{'relevant_windows':[[0.,4.],[8.,12.]]},{'relevant_windows':[]}])
        torch.testing.assert_close(target,torch.tensor([[1.,1.,1.,0.],[0.,0.,0.,0.]]))
        self.assertFalse(target.requires_grad)
    def test_gru_temporal_axes_padding_and_candidate_permutation(self):
        torch.manual_seed(3); pv=PostVerification().eval()
        video=torch.randn(2,8,256,requires_grad=True); mask=torch.arange(8)[None]<torch.tensor([8,4])[:,None]
        spans=torch.tensor([[[1.,5.,.8],[10.,16.,.9]],[[0.,4.,.8],[4.,8.,.9]]]); valid=torch.ones(2,2,dtype=torch.bool)
        before=pv(video,mask,spans,valid)
        flipped=pv(video,mask,spans.flip(1),valid)
        torch.testing.assert_close(before['quality_score'],flipped['quality_score'].flip(1))
        after_video=video.detach().clone();after_video[1,4:]=10000
        after=pv(after_video,mask,spans,valid)
        torch.testing.assert_close(before['quality_score'],after['quality_score'])
        self.assertEqual(tuple(before['moment_sequence'].shape[:2]),(2,2))
        before['quality_score'].sum().backward();self.assertGreater(float(video.grad.abs().sum()),0)
        self.assertEqual(float(video.grad[1,4:].abs().sum()),0)
    def test_no_background_positives_and_padding(self):
        video=torch.randn(1,8,256,requires_grad=True); mask=torch.tensor([[1,1,1,1,0,0,0,0]],dtype=torch.bool)
        zero=representation_loss(video,mask,[{'relevant_windows':[]}]);self.assertEqual(float(zero.detach()),0);zero.backward()
        gt=[{'relevant_windows':[[0.,4.],[4.,8.]]}]
        a=representation_loss(video,mask,gt)
        changed=video.detach().clone();changed[:,4:]=1000
        torch.testing.assert_close(a,representation_loss(changed,mask,gt))
        a.backward();self.assertGreater(float(video.grad.abs().sum()),0)
    def test_invalid_candidates_do_not_contribute(self):
        video=torch.randn(1,5,256,requires_grad=True); pv=PostVerification()
        spans=torch.tensor([[[0.,4.,.9],[5.,5.,.8],[6.,2.,.7]]]);valid=torch.tensor([[1,0,0]],dtype=torch.bool)
        mask=torch.ones(1,5,dtype=torch.bool)
        out=dict(pv(video,mask,spans,valid),candidate_spans=spans,candidate_valid=valid,fusion_video=video,fusion_valid=mask)
        q,r=losses(out,[{'relevant_windows':[[0.,4.]]}]);expected=(out['quality_score'][0,0]-1).square()
        torch.testing.assert_close(q,expected)
        (q+.2*r).backward();self.assertTrue(torch.isfinite(video.grad).all())

if __name__=='__main__':unittest.main()
