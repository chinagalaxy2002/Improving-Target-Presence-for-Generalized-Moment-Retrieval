"""Query-conditioned evidence with pooling on adapter-provided real timestamps."""
import math
import torch
from torch import nn


class TemporalEvidence(nn.Module):
    def __init__(self, video_dim=256, global_dim=256, hidden_dim=256, extension=.5):
        super().__init__()
        self.query_projection = nn.Linear(global_dim, hidden_dim)
        self.video_projection = nn.Linear(video_dim, hidden_dim)
        self.extension = extension

    def forward(self, video, global_event, valid):
        query = self.query_projection(global_event.detach())
        keys = self.video_projection(video.detach())
        logits = torch.einsum('bd,btd->bt', query, keys) / math.sqrt(keys.shape[-1])
        return logits.masked_fill(~valid, 0), logits.sigmoid().masked_fill(~valid, 0)

    def pool(self, video, evidence, times, valid, proposals, proposal_valid, durations):
        start, end = proposals[..., 0], proposals[..., 1]
        inside = (times[:, None] >= start[..., None]) & (times[:, None] <= end[..., None]) & valid[:, None]
        # Sub-grid proposals read the nearest valid clip. No invented temporal grid.
        nearest = (times[:, None] - ((start + end) / 2)[..., None]).abs().masked_fill(~valid[:, None], torch.inf).argmin(-1)
        fallback = torch.zeros_like(inside).scatter_(-1, nearest[..., None], True)
        inside = torch.where(inside.any(-1, keepdim=True), inside, fallback) & proposal_valid[..., None]
        width = (end - start).clamp_min(1e-6)
        left = (start - self.extension * width).clamp_min(0)
        right = torch.minimum(end + self.extension * width, durations[:, None])
        neighbor = (times[:, None] >= left[..., None]) & (times[:, None] <= right[..., None]) & valid[:, None] & ~inside
        weights = inside.float() / inside.sum(-1, keepdim=True).clamp_min(1)
        bg_weights = neighbor.float() / neighbor.sum(-1, keepdim=True).clamp_min(1)
        mean = torch.einsum('bkt,bt->bk', weights, evidence)
        background = torch.einsum('bkt,bt->bk', bg_weights, evidence)
        # Empty side neighborhoods have neutral contrast, including full-video proposals.
        background = torch.where(neighbor.any(-1), background, mean)
        return dict(temporal_feature=torch.einsum('bkt,btd->bkd', weights, video.detach()),
                    evidence_mean=mean, background_evidence=background, evidence_contrast=mean-background)
