from torch import nn


class QualityRanker(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.quality = nn.Sequential(nn.Linear(width, width//2), nn.ReLU(), nn.Linear(width//2, 1))
        self.residual = nn.Sequential(nn.Linear(width, width//2), nn.ReLU(), nn.Linear(width//2, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, h, valid):
        return dict(quality_logit=self.quality(h).squeeze(-1).masked_fill(~valid, 0),
                    score_residual=self.residual(h).squeeze(-1).masked_fill(~valid, 0))
