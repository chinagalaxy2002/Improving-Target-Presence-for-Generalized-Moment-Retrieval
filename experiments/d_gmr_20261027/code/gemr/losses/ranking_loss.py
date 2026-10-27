from torch.nn import functional as F
from .quality_loss import masked_mean


def ranking_loss(logits, quality, valid, epsilon=.1):
    pairs = valid[:, :, None] & valid[:, None, :]
    pairs &= quality[:, :, None] > quality[:, None, :]+epsilon
    # At least the better candidate must overlap real GT. Negative queries have
    # no valid pair, and many zero-IoU background pairs do not dominate training.
    pairs &= quality[:, :, None] > epsilon
    return masked_mean(F.softplus(-(logits[:, :, None]-logits[:, None, :])), pairs)
