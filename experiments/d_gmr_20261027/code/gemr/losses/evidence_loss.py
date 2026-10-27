import torch
from torch.nn import functional as F
from .quality_loss import masked_mean


def evidence_targets(times, valid, metas):
    labels = torch.zeros_like(times)
    for b, meta in enumerate(metas):
        for start, end in meta['relevant_windows']:
            labels[b] = torch.maximum(labels[b], ((times[b] >= start) & (times[b] <= end)).to(times.dtype))
    return labels.masked_fill(~valid, 0)


def evidence_loss(logits, labels, valid, positive_weight=1.):
    return masked_mean(F.binary_cross_entropy_with_logits(logits, labels, reduction='none',
                       pos_weight=logits.new_tensor(positive_weight)), valid)
