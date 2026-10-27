"""DGMR Model: Decoupled Generalized Moment Retrieval with Plurality-Aware Reranker."""
import sys
import torch
from torch import nn

from gemr.adapters.native import ProbabilityScoreAdapter
from gemr.models.competition_cooperation import CompetitionCooperation
from gemr.models.proposal_encoder import ProposalEncoder
from gemr.models.quality_ranker import QualityRanker
from d_gmr.common import REFERENCE, module


class TemporalEvidenceDGMR(nn.Module):
    def __init__(self, width=256, extension=.5):
        super().__init__()
        self.query_projection = nn.Linear(width, width)
        self.video_projection = nn.Linear(width, width)
        self.extension = extension

    def forward(self, video, global_event, valid):
        query = self.query_projection(global_event)
        keys = self.video_projection(video)
        logits = torch.einsum('bd,btd->bt', query, keys)/(keys.shape[-1]**.5)
        return logits.masked_fill(~valid, 0), logits.sigmoid().masked_fill(~valid, 0)

    def pool(self, video, evidence, times, valid, proposals, proposal_valid, durations):
        start, end = proposals[..., 0], proposals[..., 1]
        inside = ((times[:, None] >= start[..., None]) & (times[:, None] <= end[..., None]) &
                  valid[:, None])
        nearest = (times[:, None]-((start+end)/2)[..., None]).abs().masked_fill(
            ~valid[:, None], torch.inf).argmin(-1)
        fallback = torch.zeros_like(inside).scatter_(-1, nearest[..., None], True)
        inside = torch.where(inside.any(-1, keepdim=True), inside, fallback) & proposal_valid[..., None]
        width = (end-start).clamp_min(1e-6)
        left = (start-self.extension*width).clamp_min(0)
        right = torch.minimum(end+self.extension*width, durations[:, None])
        neighbor = ((times[:, None] >= left[..., None]) & (times[:, None] <= right[..., None]) &
                    valid[:, None] & ~inside)
        weights = inside.float()/inside.sum(-1, keepdim=True).clamp_min(1)
        background_weights = neighbor.float()/neighbor.sum(-1, keepdim=True).clamp_min(1)
        mean = torch.einsum('bkt,bt->bk', weights, evidence)
        background = torch.einsum('bkt,bt->bk', background_weights, evidence)
        background = torch.where(neighbor.any(-1), background, mean)
        return dict(temporal_feature=torch.einsum('bkt,btd->bkd', weights, video),
                    evidence_mean=mean, background_evidence=background,
                    evidence_contrast=mean-background)


class DGMRModel(nn.Module):
    def __init__(self, adapter, config):
        super().__init__()
        self.adapter = adapter
        self.config = dict(config)
        self.base = adapter.base
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config['seed'])
            sys.path.insert(0, str(REFERENCE/'backends/moment_detr'))
            geq_class = module('dgmr_random_geq',
                               REFERENCE/'backends/moment_detr/model.py').GlobalEventQueryHead
            self.geq = geq_class(len(self.base.transformer.encoder.layers))
            self.temporal_evidence = TemporalEvidenceDGMR(
                config['hidden_dim'], config['neighborhood_extension'])
            self.proposal_encoder = ProposalEncoder(width=config['hidden_dim'], dropout=config['dropout'])
            self.reasoner = CompetitionCooperation(config['hidden_dim'], config['dropout'])
            self.ranker = QualityRanker(config['hidden_dim'])
        self.context = None
        self.handles = [self.base.input_txt_proj.register_forward_hook(self._text),
                        self.base.transformer.register_forward_hook(self._transformer),
                        self.geq.global_head.register_forward_pre_hook(self._global)]
        self.handles += [layer.register_forward_hook(self._video)
                         for layer in self.base.transformer.encoder.layers]
        self.requires_grad_(True)

    def _text(self, module_value, args, output):
        if self.context is not None:
            self.context['text'] = output

    def _video(self, module_value, args, output):
        if self.context is not None:
            value = output.transpose(0, 1)[:, self.adapter.offset:self.adapter.offset+self.length]
            self.context['videos'].append(value.clone() if self.adapter.name == 'cg_detr' else value)

    def _transformer(self, module_value, args, output):
        if self.context is not None:
            final = output[self.adapter.final_index]
            if self.adapter.name == 'flashvtg':
                final = final.transpose(0, 1)
            self.context['final'] = final[:, :self.length]
            if self.adapter.name != 'flashvtg':
                self.context['proposal_features'] = output[0][-1]

    def _global(self, module_value, args):
        if self.context is not None:
            self.context['global_event_feature'] = args[0]

    def _native_proposals(self, native, durations):
        if self.adapter.name == 'flashvtg':
            if self.training:
                coordinate, point = native['out_coord'], native['point']
                xy = coordinate*coordinate.new_tensor([-1., 1.])*point[None, :, 3:4]+point[None, :, 0:1]
                xy = xy*self.adapter.opt.clip_length
                proposals = torch.cat([xy, native['out_class'].sigmoid()], -1)
                valid = torch.cat(native['pymid_msk'], 1).bool()
            else:
                assert len(durations) == 1, 'FlashVTG native inference decoder requires batch size 1'
                proposals = native['_out']['boundary'][None].clone()
                valid = torch.ones(proposals.shape[:2], dtype=torch.bool, device=proposals.device)
        else:
            center, width = native['pred_spans'].unbind(-1)
            xy = torch.stack([center-width/2, center+width/2], -1)*durations[:, None, None]
            proposals = torch.cat([xy, native['pred_logits'].softmax(-1)[..., :1]], -1)
            valid = torch.ones(proposals.shape[:2], dtype=torch.bool, device=proposals.device)
        xy = torch.minimum(proposals[..., :2].clamp_min(0), durations[:, None, None])
        proposals = torch.cat([xy, proposals[..., 2:3]], -1)
        valid &= torch.isfinite(proposals).all(-1) & (xy[..., 1] > xy[..., 0])
        return proposals, valid

    def _native_forward(self, inputs):
        assert self.context is None
        self.length = inputs['src_vid_mask'].shape[1]
        durations = inputs['durations']
        features = {key: value for key, value in inputs.items() if key != 'durations'}
        if self.adapter.name == 'cg_detr' and not self.training:
            features.pop('targets', None)
        context = {'videos': []}
        self.context = context
        try:
            native = self.base(**features)
            text_valid = inputs['src_txt_mask'].bool()
            text = (context['text']*text_valid[..., None]).sum(1)/text_valid.sum(1, keepdim=True).clamp_min(1)
            native_presence = native.get('pred_exist_logits', native.get('pred_exist_logit'))
            assert native_presence is not None
            logit = self.geq(context['text'], text_valid,
                             context['videos'], inputs['src_vid_mask'].bool())
            proposals, proposal_valid = self._native_proposals(native, durations)
            times, video_valid = self.adapter.timestamps(inputs, durations)
            if self.adapter.name == 'flashvtg':
                proposal_features = proposals.new_zeros((*proposals.shape[:2], 256))
            else:
                proposal_features = context['proposal_features']
                assert proposal_features.shape[:2] == proposals.shape[:2]
            return dict(proposals=proposals, proposal_valid=proposal_valid,
                        native_features=proposal_features, video=context['final'],
                        video_valid=video_valid, times=times, text=text, durations=durations,
                        global_event_feature=context['global_event_feature'], presence_logit=logit,
                        presence_probability=logit.sigmoid(), native_presence_logit=native_presence,
                        native_output=native)
        finally:
            self.context = None

    def _rank(self, backbone):
        proposals, valid = backbone['proposals'], backbone['proposal_valid']
        scores = proposals[..., 2].masked_fill(~valid, -torch.inf)
        count = min(self.config['top_k'], proposals.shape[1])
        ids = scores.argsort(dim=1, descending=True, stable=True)[:, :count]
        selected = proposals.gather(1, ids[..., None].expand(-1, -1, 3))
        selected_valid = valid.gather(1, ids)
        evidence_logit, evidence = self.temporal_evidence(
            backbone['video'], backbone['global_event_feature'], backbone['video_valid'])
        pooled = self.temporal_evidence.pool(
            backbone['video'], evidence, backbone['times'], backbone['video_valid'],
            selected, selected_valid, backbone['durations'])
        native = backbone['native_features'].gather(
            1, ids[..., None].expand(-1, -1, backbone['native_features'].shape[-1]))
        if self.adapter.name == 'flashvtg':
            native = pooled['temporal_feature']
        z = self.proposal_encoder(pooled, backbone['text'], native, selected,
                                  backbone['durations'], selected_valid)
        hidden, relations = self.reasoner(z, selected, backbone['durations'], selected_valid)
        heads = self.ranker(hidden, selected_valid)
        residual = self.config['residual_scale']*heads['score_residual']
        ranking_logits = ProbabilityScoreAdapter.to_logit(selected[..., 2])+residual
        refined_score = ProbabilityScoreAdapter.restore(selected[..., 2], residual)
        all_scores = proposals[..., 2].scatter(1, ids, refined_score)
        ranked = torch.cat([proposals[..., :2], all_scores[..., None]], -1)
        assert torch.equal(ranked[..., :2], proposals[..., :2])
        return dict(**heads, **relations, **pooled, frozen=backbone,
                    evidence_logit=evidence_logit, evidence_probability=evidence,
                    selected_proposals=selected, selected_valid=selected_valid,
                    candidate_ids=ids, ranking_logits=ranking_logits,
                    ranked_proposals=ranked,
                    global_event_feature=backbone['global_event_feature'],
                    presence_logit=backbone['presence_logit'],
                    presence_probability=backbone['presence_probability'])

    def forward(self, inputs):
        backbone = self._native_forward(inputs)
        return self._rank(backbone)
