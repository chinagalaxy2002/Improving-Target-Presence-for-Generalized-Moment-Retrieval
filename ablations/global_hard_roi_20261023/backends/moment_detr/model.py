"""Moment-DETR interface around the unchanged S4 minus Fusion components."""
import torch
from torch import nn
from s4_reference import EvidenceRead, CandidateMomentExtractor, TemporalBoundaryRefiner, LoRALinear


class GlobalEventQueryHead(nn.Module):
    """Exact reference global readout, with no local-presence or fusion modules."""
    def __init__(self, depth):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, 256) * .02)
        self.text_read = EvidenceRead(256)
        self.video_reads = nn.ModuleList([EvidenceRead(256) for _ in range(depth)])
        self.global_head = nn.Linear(256, 1)

    def forward(self, text, text_valid, video, video_valid):
        state, _ = self.text_read(self.token.expand(text.shape[0], -1, -1), text, text_valid)
        for reader, memory in zip(self.video_reads, video):
            state, _ = reader(state, memory, video_valid)
        return self.global_head(state[:, 0]).squeeze(-1)


class S4MomentDETR(nn.Module):
    def __init__(self, base, job):
        super().__init__()
        assert job['presence_source'] == 'global_refine' and job['use_lora']
        assert not any(job.get(k, False) for k in ('augmentation','qcd','query_swap','background_replacement','synthetic_samples','hard_negative'))
        self.base, self.job = base, dict(job)
        layers = base.transformer.encoder.layers
        self.event_query = GlobalEventQueryHead(len(layers))
        self.extractor = CandidateMomentExtractor(256, top_k=job['top_k'], pooling_type='gaussian')
        self.refiner = TemporalBoundaryRefiner(256, max_delta=job['refine_max_delta'])
        for layer in layers:
            layer.linear1 = LoRALinear(layer.linear1, rank=job['lora_rank'])
            layer.linear2 = LoRALinear(layer.linear2, rank=job['lora_rank'])
        self.context = None
        self._handles = [base.input_txt_proj.register_forward_hook(self._text),
                         base.transformer.register_forward_hook(self._final)]
        self._handles += [layer.register_forward_hook(self._video) for layer in layers]
        for name, param in self.named_parameters():
            param.requires_grad_(not name.startswith('base.') or '.lora_' in name)

    def train(self, mode=True):
        super().train(mode)
        # Original backbone dropout is frozen; attached encoder activations retain LoRA gradients.
        self.base.eval()
        return self

    def _text(self, module, args, value):
        if self.context is not None: self.context['text'] = value

    def _video(self, module, args, value):
        if self.context is not None: self.context['video'].append(value.transpose(0, 1)[:, :self.video_length])

    def _final(self, module, args, value):
        if self.context is not None: self.context['final'] = value[1][:, :self.video_length]

    def forward(self, *, durations, **features):
        if self.context is not None: raise RuntimeError('Reentrant forward')
        assert torch.all((durations > 0) & (durations <= 150))
        video_valid, text_valid = features['src_vid_mask'].bool(), features['src_txt_mask'].bool()
        assert video_valid.any(1).all() and text_valid.any(1).all()
        self.video_length = features['src_vid'].shape[1]
        self.context = {'video': []}
        try:
            out = self.base(**features)
            ctx = self.context
            assert len(ctx['video']) == len(self.event_query.video_reads)
            logit = self.event_query(ctx['text'], text_valid, ctx['video'], video_valid)
            center, width = out['pred_spans'].unbind(-1)
            xy = torch.stack([center-width/2, center+width/2], -1) * durations[:, None, None]
            xy = torch.minimum(xy.clamp_min(0), durations[:, None, None])
            candidates = torch.cat([xy, out['pred_logits'].softmax(-1)[..., :1]], -1)
            ids = candidates[..., 2].topk(min(self.job['top_k'], candidates.shape[1]), dim=1).indices
            selected = candidates.gather(1, ids[..., None].expand(-1, -1, 3))
            tokens, _, _ = self.extractor(ctx['final'], video_valid, selected)
            refined, _ = self.refiner(tokens, ctx['text'], text_valid, selected,
                                     out['saliency_scores'].masked_fill(~video_valid, 0))
            end = torch.minimum(refined[..., 1], durations[:, None])
            start = torch.minimum(refined[..., 0], (end-1).clamp_min(0))
            refined = torch.stack([start, end, refined[..., 2]], -1)
            out.update(pred_exist_logits=logit, pred_exist_score=logit.sigmoid(), global_logit=logit,
                       refined_spans=refined, original_candidate_seconds=candidates,
                       candidate_seconds=candidates.scatter(1, ids[..., None].expand(-1, -1, 3), refined))
            return out
        finally:
            self.context = None
