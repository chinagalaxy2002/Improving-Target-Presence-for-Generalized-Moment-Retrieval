import torch
from torch import nn
from gemr.adapters.native import ProbabilityScoreAdapter
from .temporal_evidence import TemporalEvidence
from .proposal_encoder import ProposalEncoder
from .competition_cooperation import CompetitionCooperation
from .quality_ranker import QualityRanker


class GEMR(nn.Module):
    def __init__(self, adapter, config):
        super().__init__()
        self.adapter, self.config = adapter, dict(config)
        width = config['hidden_dim']
        self.temporal_evidence = TemporalEvidence(hidden_dim=width, extension=config['neighborhood_extension'])
        self.proposal_encoder = ProposalEncoder(width=width, dropout=config['dropout'])
        self.reasoner = CompetitionCooperation(width, config['dropout'])
        self.ranker = QualityRanker(width)

    def train(self, mode=True):
        super().train(mode)
        self.adapter.eval()
        return self

    def forward(self, inputs):
        frozen = self.adapter(inputs)
        return self.rank(frozen)

    def rank(self, frozen):
        proposals, valid = frozen['proposals'], frozen['proposal_valid']
        scores = proposals[..., 2].masked_fill(~valid, -torch.inf)
        ids = scores.argsort(dim=1, descending=True, stable=True)[:, :self.config['top_k']]
        selected = proposals.gather(1, ids[..., None].expand(-1, -1, 3))
        selected_valid = valid.gather(1, ids)
        evidence_logit, evidence = self.temporal_evidence(frozen['video'], frozen['global_event_feature'], frozen['video_valid'])
        pooled = self.temporal_evidence.pool(frozen['video'], evidence, frozen['times'], frozen['video_valid'],
                                            selected, selected_valid, frozen['durations'])
        native_features = frozen['native_features'].gather(1, ids[..., None].expand(-1, -1, frozen['native_features'].shape[-1]))
        if self.adapter.name == 'flashvtg':
            native_features = pooled['temporal_feature']
        z = self.proposal_encoder(pooled, frozen['text'], native_features, selected, frozen['durations'], selected_valid)
        h, relations = self.reasoner(z, selected, frozen['durations'], selected_valid)
        heads = self.ranker(h, selected_valid)
        delta = self.config['residual_scale']*heads['score_residual']
        rank_logits = ProbabilityScoreAdapter.to_logit(selected[..., 2]) + delta
        refined_scores = ProbabilityScoreAdapter.restore(selected[..., 2], delta)
        all_scores = proposals[..., 2].scatter(1, ids, refined_scores)
        refined = torch.cat([proposals[..., :2], all_scores[..., None]], -1)
        assert torch.equal(refined[..., :2], proposals[..., :2]), 'Ranking must never update coordinates'
        return dict(**heads, **relations, **pooled, frozen=frozen, evidence_logit=evidence_logit,
                    evidence_probability=evidence, selected_proposals=selected, selected_valid=selected_valid,
                    candidate_ids=ids, ranking_logits=rank_logits, ranked_proposals=refined,
                    global_event_feature=frozen['global_event_feature'], presence_logit=frozen['presence_logit'],
                    presence_probability=frozen['presence_probability'])
