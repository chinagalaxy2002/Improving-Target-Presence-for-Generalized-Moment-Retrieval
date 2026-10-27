import torch
from torch.nn import functional as F


def temporal_iou(a, b):
    inter = (torch.minimum(a[..., :, None, 1], b[..., None, :, 1]) -
             torch.maximum(a[..., :, None, 0], b[..., None, :, 0])).clamp_min(0)
    union = (a[..., 1]-a[..., 0])[..., :, None] + (b[..., 1]-b[..., 0])[..., None, :] - inter
    return inter / union.clamp_min(1e-6)


def temporal_relations(proposals, z, durations, valid):
    xy = proposals[..., :2]
    centers = xy.mean(-1)
    lengths = (xy[..., 1]-xy[..., 0]).clamp_min(1e-6)
    relations = torch.stack([temporal_iou(xy, xy),
                            (centers[:, :, None]-centers[:, None, :]).abs()/durations[:, None, None],
                            (lengths[:, :, None]/lengths[:, None, :]).log(),
                            F.normalize(z, dim=-1) @ F.normalize(z, dim=-1).transpose(1, 2)], -1)
    pair_valid = valid[:, :, None] & valid[:, None, :]
    pair_valid &= ~torch.eye(z.shape[1], dtype=torch.bool, device=z.device)[None]
    return relations.masked_fill(~pair_valid[..., None], 0), pair_valid
