import torch
from gemr.models.temporal_relation import temporal_iou
from .quality_loss import quality_loss
from .ranking_loss import ranking_loss
from .set_loss import set_loss
from .evidence_loss import evidence_loss, evidence_targets


def objective(output, metas, config):
    proposals, valid = output['selected_proposals'], output['selected_valid']
    labels = torch.zeros_like(output['quality_logit'])
    for b, meta in enumerate(metas):
        gt = proposals.new_tensor(meta['relevant_windows']).reshape(-1, 2)
        if len(gt):
            labels[b] = temporal_iou(proposals[b, :, :2], gt).max(-1).values
    frozen = output['frozen']
    evidence_y = evidence_targets(frozen['times'], frozen['video_valid'], metas)
    set_value, diagnostics = set_loss(output, metas, config['coverage_iou_threshold'])
    parts = dict(quality=quality_loss(output['quality_logit'], labels, valid),
                 ranking=ranking_loss(output['ranking_logits'], labels, valid, config['ranking_epsilon']),
                 set=set_value,
                 evidence=evidence_loss(output['evidence_logit'], evidence_y, frozen['video_valid'],
                                        config['evidence_positive_weight']))
    contributions = {k: config['loss_weights'][k]*v for k, v in parts.items()}
    total = sum(contributions.values())
    assert all(torch.isfinite(x) for x in [total, *parts.values()]), parts
    return total, parts, contributions, diagnostics
