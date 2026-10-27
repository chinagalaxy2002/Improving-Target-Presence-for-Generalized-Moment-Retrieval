"""Distinct learned temporal relation branches with supervised instance semantics."""
import torch
from torch import nn
from .temporal_relation import temporal_relations


class RelationBranch(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(width*2+4, width//2), nn.ReLU(), nn.Linear(width//2, 1))
        self.value = nn.Linear(width, width)

    def forward(self, z, relation, valid):
        k = z.shape[1]
        inputs = torch.cat([z[:, :, None].expand(-1, -1, k, -1),
                            z[:, None].expand(-1, k, -1, -1), relation], -1)
        logits = self.pair(inputs).squeeze(-1)
        # Softmax plus a null message: rows with no neighbors produce exactly zero.
        augmented = torch.cat([logits.masked_fill(~valid, -torch.inf), torch.zeros_like(logits[..., :1])], -1)
        attention = augmented.softmax(-1)[..., :-1]
        return attention @ self.value(z), logits, attention


class CompetitionCooperation(nn.Module):
    def __init__(self, width=256, dropout=.1):
        super().__init__()
        self.competition = RelationBranch(width)
        self.cooperation = RelationBranch(width)
        self.gate = nn.Linear(width, 2)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, width*2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width*2, width))
        self.final_norm = nn.LayerNorm(width)

    def forward(self, z, proposals, durations, valid):
        relation, pair_valid = temporal_relations(proposals, z, durations, valid)
        comp, comp_logits, comp_attention = self.competition(z, relation, pair_valid)
        coop, coop_logits, coop_attention = self.cooperation(z, relation, pair_valid)
        gates = self.gate(z).sigmoid()
        h = self.norm(z + gates[..., :1]*comp + gates[..., 1:]*coop)
        h = self.final_norm(h + self.ffn(h)).masked_fill(~valid[..., None], 0)
        return h, dict(competition_logits=comp_logits, cooperation_logits=coop_logits,
                       competition_attention=comp_attention, cooperation_attention=coop_attention,
                       relation_mask=pair_valid, relation_features=relation, relation_gates=gates)
