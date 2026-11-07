"""Query-level existence adapter and D1 attention capture for Soccer-GMR."""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn

from sim_detr.semantic_calibration.semantic_calibrator import pool_video_evidence
from sim_detr.semantic_calibration.transformer_capture import TransformerOutputCapture

from .attention_capture import NativeD1AttentionCapture


class GMRAdapter(nn.Module):
    def __init__(self, hidden_dim=256, pool="max"):
        super().__init__()
        self.pool = pool
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        ])

    def forward(self, decoder_queries):
        if self.pool == "mean":
            value = decoder_queries.mean(dim=1)
        else:
            value = decoder_queries.max(dim=1).values
        return self.layers[1](F.relu(self.layers[0](value))).squeeze(-1)


class SoccerGMRModel(nn.Module):
    """Wrap Sim-DETR CSC variant with GMR existence head and D1 attention binding capture."""

    def __init__(
        self,
        semantic_model,
        hidden_dim: int = 256,
        exist_pool: str = "max",
        evidence_source: str = "native_mask_logits",
        binding_loss_coef: float = 0.0,
        options = None,
    ):
        super().__init__()
        self.semantic_model = semantic_model
        self.evidence_source = evidence_source
        self.binding_loss_coef = float(binding_loss_coef)
        self.exist_head = GMRAdapter(hidden_dim=hidden_dim, pool=exist_pool)
        self.exist_capture = TransformerOutputCapture(semantic_model.base_model.transformer)
        self.d1_capture = NativeD1AttentionCapture(semantic_model.base_model.transformer)

        from soccergmr.modules.interlayer_bind import build_interlayer_adapter
        num_queries = getattr(options, "num_queries", 10) if options is not None else 10
        self.query_cgp = build_interlayer_adapter(options, hidden_dim=hidden_dim, num_queries=num_queries)
        self.semantic_model.base_model.transformer.decoder.interlayer_adapter = self.query_cgp

    def forward(self, *args, **kwargs):
        src_vid = kwargs.get("src_vid")
        src_txt = kwargs.get("src_txt")
        src_txt_mask = kwargs.get("src_txt_mask")
        src_vid_mask = kwargs.get("src_vid_mask")
        if self.query_cgp is not None and src_txt is not None:
            self.query_cgp.clear_diagnostics()
            txt_proj = self.semantic_model.base_model.input_txt_proj(src_txt)
            semantic_mask = src_txt_mask.bool()
            semantic_count = semantic_mask.sum(dim=1, keepdim=True).clamp_min(1)
            query_semantic = (txt_proj * semantic_mask.unsqueeze(-1).to(txt_proj.dtype)).sum(dim=1) / semantic_count.to(txt_proj.dtype)
            self.semantic_model.base_model.transformer.decoder.interlayer_adapter_kwargs = {
                "query_semantic": query_semantic,
                "video_length": src_vid.shape[1],
            }

        outputs = self.semantic_model(*args, **kwargs)
        decoder_states, video_memory = self.exist_capture.consume()
        outputs["pred_exist_logits"] = self.exist_head(decoder_states[-1])

        # Capture and attach normalized D1 cross-attention for binding loss / pooling
        video_mask = outputs.get("video_mask")
        d1_attn = self.d1_capture.normalized_attention(video_mask)
        outputs["d1_attention"] = d1_attn

        # If evidence_source is d1_attention and variant is full, re-pool evidence using bound D1 attention
        if self.semantic_model.semantic_variant == "full" and self.evidence_source == "d1_attention":
            video_context = pool_video_evidence(d1_attn, video_memory)
            semantic_output = self.semantic_model.semantic_calibrator(
                query_states=decoder_states[-1],
                static_semantic=outputs["src_txt_cls_ed"],
                video_context=video_context,
                native_logits=outputs["pred_logits_native"],
                variant="full",
                semantic_scale_override=self.semantic_model.semantic_scale_override,
            )
            outputs["pred_logits"] = semantic_output.pred_logits
            outputs["semantic_scores"] = semantic_output.semantic_scores

        if self.query_cgp is not None and self.query_cgp.last_output is not None:
            outputs["query_cgp_temporal_attention"] = self.query_cgp.last_output.temporal_attention
            outputs["query_cgp_basis_weights"] = self.query_cgp.last_output.basis_weights
            outputs["query_cgp_video_mask"] = video_mask.bool() if video_mask is not None else src_vid_mask.bool()

        return outputs

    def set_counterfactual(self, semantic_variant=None, context_variant=None):
        self.semantic_model.set_counterfactual(semantic_variant, context_variant)

    @property
    def semantic_calibrator(self):
        return self.semantic_model.semantic_calibrator

    def close(self):
        self.d1_capture.close()
        self.exist_capture.close()
        self.semantic_model.close()
