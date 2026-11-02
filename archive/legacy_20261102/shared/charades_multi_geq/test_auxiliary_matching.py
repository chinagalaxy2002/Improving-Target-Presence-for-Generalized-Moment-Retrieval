"""CPU equivalence checks against unchanged per-example SciPy matching/loss."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from adapter import Adapter
from auxiliary_matching import hungarian_batch
from test_eatr_positive_auxiliary import example,make_adapter


class BatchedMatchingTests(unittest.TestCase):
    def test_assignments_keep_ties_and_variable_target_counts(self):
        generator=torch.Generator().manual_seed(2023)
        costs=[torch.rand(24,2,generator=generator),torch.zeros(24,2),
               torch.ones(24,2),torch.rand(24,6,generator=generator),torch.empty(24,0)]
        column=torch.rand(24,1,generator=generator)
        costs += [column.repeat(1,2),column.repeat(1,2)+torch.tensor([0.,1e-7])]
        actual=hungarian_batch(costs)
        for cost,(ii,jj) in zip(costs,actual):
            expected=linear_sum_assignment(cost.numpy())
            np.testing.assert_array_equal(ii,expected[0])
            np.testing.assert_array_equal(jj,expected[1])
        self.assertEqual(hungarian_batch([]),[])

    def test_one_host_transfer_per_layer(self):
        costs=[torch.zeros(24,2) for _ in range(13)]
        cpu=torch.Tensor.cpu
        calls=[]
        def counted(tensor,*args,**kwargs):
            calls.append(tensor.numel())
            return cpu(tensor,*args,**kwargs)
        with patch.object(torch.Tensor,'cpu',counted):
            hungarian_batch(costs)
        self.assertEqual(calls,[13*24*2])

    def test_full_physical_batch_loss_and_gradients_match_legacy(self):
        for tied in (False,True):
            with self.subTest(tied=tied):
                output,targets=example([True]*32)
                params=[]
                for prediction in output['o2m_outputs'].values():
                    for layer in prediction['aux_outputs']+[prediction]:
                        params += [layer['pred_logits'],layer['pred_spans']]
                        if tied:
                            with torch.no_grad():
                                layer['pred_logits'].zero_()
                                layer['pred_spans'].fill_(.4)
                actual=Adapter.loss(make_adapter(True),output,targets,[],1)
                # All-positive legacy path remains unchanged in Adapter.loss.
                expected=Adapter.loss(make_adapter(False),output,targets,[],1)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                ga=torch.autograd.grad(actual,params,retain_graph=True)
                ge=torch.autograd.grad(expected,params)
                for a,e in zip(ga,ge):
                    torch.testing.assert_close(a,e,rtol=0,atol=0)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
