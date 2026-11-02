"""Regression checks for the user-authorized EaTR positive-query auxiliary run."""
import types
import unittest

import torch
from torch.nn import functional as F

from adapter import Adapter


def make_adapter(positive_only):
    cfg = {'multi_auxiliary_query_scope': 'positive_only_v1'} if positive_only else {}

    def native_loss(output, targets, *_):
        value = F.binary_cross_entropy_with_logits(output['pred_exist_logits'], targets['exist_label'])
        return value, {'loss_exist': value}

    return types.SimpleNamespace(device='cpu', native=types.SimpleNamespace(cfg=cfg, native_loss=native_loss))


def example(positive):
    torch.manual_seed(2023)
    batch = len(positive)
    routes = {}
    for name in ('sepffn', 'instruction'):
        layers = [dict(pred_logits=torch.randn(batch, 24, 2, requires_grad=True),
                       pred_spans=torch.rand(batch, 24, 2, requires_grad=True)) for _ in range(3)]
        routes[name] = dict(layers[-1], aux_outputs=layers[:-1])
    output = dict(pred_exist_logits=torch.zeros(batch, requires_grad=True), o2m_outputs=routes)
    targets = dict(exist_label=torch.tensor(positive, dtype=torch.float32),
                   span_labels=[{'spans': torch.tensor([[0.5, 0.2]]) if value else torch.empty(0, 2)}
                                for value in positive])
    return output, targets


class PositiveAuxiliaryTests(unittest.TestCase):
    def test_mixed_batch_matches_positive_subset_and_preserves_negative_bce(self):
        output, targets = example([True, False, True, False])
        value = Adapter.loss(make_adapter(True), output, targets, [], 1)
        keep = targets['exist_label'].bool()
        subset = dict(pred_exist_logits=output['pred_exist_logits'][keep], o2m_outputs={})
        for name, prediction in output['o2m_outputs'].items():
            subset['o2m_outputs'][name] = {
                key: ([{k: v[keep] for k, v in layer.items()} for layer in item]
                      if key == 'aux_outputs' else item[keep])
                for key, item in prediction.items()
            }
        selected = dict(exist_label=targets['exist_label'][keep],
                        span_labels=[t for t, yes in zip(targets['span_labels'], keep.tolist()) if yes])
        reference = Adapter.loss(make_adapter(False), subset, selected, [], 1)
        torch.testing.assert_close(value, reference)
        value.backward()
        for prediction in output['o2m_outputs'].values():
            for layer in prediction['aux_outputs'] + [prediction]:
                for key in ('pred_logits', 'pred_spans'):
                    gradient = layer[key].grad
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertEqual(float(gradient[~keep].abs().sum()), 0.0)
                    self.assertGreater(float(gradient[keep].abs().sum()), 0.0)
        self.assertGreater(float(output['pred_exist_logits'].grad[~keep].abs().sum()), 0.0)

    def test_all_negative_has_existence_loss_without_auxiliary_gradient(self):
        output, targets = example([False, False])
        value = Adapter.loss(make_adapter(True), output, targets, [], 1)
        torch.testing.assert_close(value, F.binary_cross_entropy_with_logits(
            output['pred_exist_logits'], targets['exist_label']))
        value.backward()
        self.assertGreater(float(output['pred_exist_logits'].grad.abs().sum()), 0.0)
        for prediction in output['o2m_outputs'].values():
            for layer in prediction['aux_outputs'] + [prediction]:
                self.assertIsNone(layer['pred_logits'].grad)
                self.assertIsNone(layer['pred_spans'].grad)

    def test_all_positive_matches_previous_loss_and_gradients(self):
        output, targets = example([True, True])
        params = [layer[key] for prediction in output['o2m_outputs'].values()
                  for layer in prediction['aux_outputs'] + [prediction]
                  for key in ('pred_logits', 'pred_spans')]
        value = Adapter.loss(make_adapter(True), output, targets, [], 1)
        reference = Adapter.loss(make_adapter(False), output, targets, [], 1)
        torch.testing.assert_close(value, reference)
        actual = torch.autograd.grad(value, params, retain_graph=True)
        expected = torch.autograd.grad(reference, params)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
