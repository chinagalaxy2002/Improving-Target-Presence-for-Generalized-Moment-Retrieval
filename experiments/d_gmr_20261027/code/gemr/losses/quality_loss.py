from torch.nn import functional as F


def masked_mean(values, mask):
    return (values*mask).sum()/mask.sum().clamp_min(1)


def quality_loss(logits, target, valid):
    return masked_mean(F.smooth_l1_loss(logits.sigmoid(), target, reduction='none'), valid)
