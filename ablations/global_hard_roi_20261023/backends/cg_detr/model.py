"""CG-DETR S4-minus-Fusion: independent global event query + Gaussian RoI refiner."""
import json
import sys
from pathlib import Path
import torch
from torch import nn
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from cg_detr.config import BaseOptions
from cg_detr.model import build_model
from cg_detr.span_utils import span_cxw_to_xx
from reference_modules import EvidenceRead, CandidateMomentExtractor, TemporalBoundaryRefiner, LoRALinear


def valid_seconds(times, durations, min_width=1.0):
    if not bool((durations > 0).all()):
        raise ValueError("Every video must have a positive duration")
    duration = durations[:, None]
    width = torch.minimum(duration, torch.full_like(duration, min_width))
    start = torch.minimum(times[..., 0].clamp(min=0), duration - width)
    end = torch.minimum(torch.maximum(times[..., 1], start + width), duration)
    return torch.stack([start, end], dim=-1)


class GlobalEventQueryHead(nn.Module):
    """Exact active global readout computations from reference MomentConditionedVerifier."""
    def __init__(self, depth, width=256):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, width) * .02)
        self.text_read = EvidenceRead(width)
        self.video_reads = nn.ModuleList(EvidenceRead(width) for _ in range(depth))
        self.global_head = nn.Linear(width, 1)

    def forward(self, text, text_mask, videos, video_mask):
        state, _ = self.text_read(self.token.expand(text.shape[0], -1, -1), text, text_mask)
        for reader, memory in zip(self.video_reads, videos):
            state, _ = reader(state, memory, video_mask)
        return self.global_head(state[:, 0]).squeeze(-1)


class S4CGDETR(nn.Module):
    def __init__(self, base, config):
        super().__init__()
        self.base = base
        self.config = dict(config)
        self.context = None
        depth = len(base.transformer.encoder.layers)
        self.global_event_query = GlobalEventQueryHead(depth)
        self.extractor = CandidateMomentExtractor(top_k=config['top_k'], pooling_type='gaussian')
        self.refiner = TemporalBoundaryRefiner(max_delta=config['refine_max_delta'])
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        for layer in base.transformer.encoder.layers:
            layer.linear1 = LoRALinear(layer.linear1, rank=config['lora_rank'])
            layer.linear2 = LoRALinear(layer.linear2, rank=config['lora_rank'])
        self.handles = [base.input_txt_proj.register_forward_hook(self.capture_text),
                        base.transformer.register_forward_hook(self.capture_transformer)]
        self.handles += [layer.register_forward_hook(self.capture_video) for layer in base.transformer.encoder.layers]

    def train(self, mode=True):
        super().train(mode)
        # Frozen pretrained dropout uses evaluation mode, as reference training_mode does.
        self.base.eval()
        return self

    def capture_text(self, module, inputs, output):
        self.context['text'] = output

    def capture_video(self, module, inputs, output):
        # CG encoder includes a prepended saliency token. Retain video frames only.
        self.context['videos'].append(output[1:].transpose(0, 1).clone())

    def capture_transformer(self, module, inputs, output):
        self.context['final_video'] = output[2]

    def forward(self, durations, candidate_override=None, **inputs):
        assert self.context is None
        if inputs.get('vid') is not None:
            raise ValueError('vid must be None: CG synthetic pairing must stay disabled')
        self.context = {'videos': []}
        try:
            output = self.base(**inputs)
            ctx = self.context
            presence = self.global_event_query(ctx['text'], inputs['src_txt_mask'], ctx['videos'], inputs['src_vid_mask'])
            spans = valid_seconds(span_cxw_to_xx(output['pred_spans']).clamp(0, 1) * durations[:, None, None], durations, min_width=1e-3)
            scores = output['pred_logits'].softmax(-1)[..., 0]
            all_candidates = torch.cat([spans, scores[..., None]], dim=-1)
            indices = scores.topk(min(self.config['top_k'], scores.shape[1]), dim=1).indices
            candidates = all_candidates.gather(1, indices[..., None].expand(-1, -1, 3))
            if candidate_override is not None:
                candidates = candidate_override
            moments, _, _ = self.extractor(ctx['final_video'], inputs['src_vid_mask'], candidates)
            refined, deltas = self.refiner(moments, ctx['text'], inputs['src_txt_mask'], candidates, output['saliency_scores'])
            # The reference assumes 150-second videos; clip to actual per-video duration.
            times = valid_seconds(refined[..., :2], durations)
            refined = torch.cat([times, refined[..., 2:]], dim=-1)
            candidate_seconds = all_candidates.scatter(1, indices[..., None].expand(-1, -1, 3), refined)
            assert torch.isfinite(presence).all() and torch.isfinite(candidate_seconds).all()
            assert (candidate_seconds[..., 0] >= 0).all()
            assert (candidate_seconds[..., 1] > candidate_seconds[..., 0]).all()
            assert (candidate_seconds[..., 1] <= durations[:, None]).all()
            output.update(pred_exist_logits=presence, refined_spans=refined, candidate_spans=candidates,
                          candidate_seconds=candidate_seconds, native_candidate_seconds=all_candidates,
                          refine_deltas=deltas)
            return output
        finally:
            self.context = None

