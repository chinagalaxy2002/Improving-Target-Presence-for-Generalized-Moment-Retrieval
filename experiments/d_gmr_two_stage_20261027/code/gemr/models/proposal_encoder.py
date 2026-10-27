import torch
from torch import nn


class ProposalEncoder(nn.Module):
    def __init__(self, video_dim=256, text_dim=256, native_dim=256, width=256, dropout=.1):
        super().__init__()
        self.video_projection = nn.Linear(video_dim, width)
        self.text_projection = nn.Linear(text_dim, width)
        self.native_projection = nn.Linear(native_dim, width)
        self.encoder = nn.Sequential(nn.Linear(3*width+6, width), nn.LayerNorm(width), nn.ReLU(),
                                     nn.Dropout(dropout), nn.Linear(width, width), nn.LayerNorm(width))

    def forward(self, pooled, text, native, proposals, durations, valid):
        start, end, score = proposals.unbind(-1)
        geom = torch.stack([start/durations[:, None], end/durations[:, None],
                            (end-start)/durations[:, None], score], -1)
        z = self.encoder(torch.cat([self.video_projection(pooled['temporal_feature']),
                                   self.text_projection(text)[:, None].expand(-1, proposals.shape[1], -1),
                                   self.native_projection(native), geom,
                                   pooled['evidence_mean'][..., None], pooled['evidence_contrast'][..., None]], -1))
        return z.masked_fill(~valid[..., None], 0)
