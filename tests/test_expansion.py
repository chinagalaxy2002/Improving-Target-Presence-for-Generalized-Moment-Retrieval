import torch
from torch import nn

from soccergmr.backbones.bam_adapter import BAMLoss
from soccergmr.predict import export


def test_bam_candidate_ranking_uses_quality_without_multiplying_class_probability():
    outputs = {
        'pred_logits': torch.tensor([[[9., -9.], [-9., 9.]]]),
        'pred_quality': torch.tensor([[[.2], [.8]]]),
        'pred_spans': torch.tensor([[[.25, .1], [.75, .1]]]),
        'pred_exist_logits': torch.tensor([0.]),
    }
    rows = export(outputs, [{'qid': 1, 'vid': 'v', 'duration': 100.}])
    assert rows[0]['pred_relevant_windows'][0][0] == 70.
    assert abs(rows[0]['pred_relevant_windows'][0][2] - .8) < 1e-6


def test_bam_same_video_and_singleton_are_not_synthetic_negative_pairs():
    native = nn.Module()
    native.weight_dict = {}
    native.saliency_margin = .2
    criterion = BAMLoss(native)
    outputs = {'video_mask': torch.tensor([[1., 1.]]),
               'saliency_scores': torch.tensor([[0., 0.]]),
               'saliency_scores_neg': torch.tensor([[100., 100.]])}
    targets = {'video_group': torch.tensor([0]), 'exist_label': torch.tensor([1.]),
               'saliency_all_labels': torch.tensor([[1., 0.]]),
               'saliency_pair_mask': torch.tensor([1.]),
               'saliency_pos_labels': torch.tensor([[0, 0]]),
               'saliency_neg_labels': torch.tensor([[1, 1]])}
    first = criterion.saliency(outputs, targets)
    outputs['saliency_scores_neg'] = torch.tensor([[-100., -100.]])
    second = criterion.saliency(outputs, targets)
    assert torch.isfinite(first) and torch.equal(first, second)
