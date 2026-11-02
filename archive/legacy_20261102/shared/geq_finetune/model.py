"""Global-only fine tuning, based on main commit 39ab0eb74427709851476b2bb86b698c8a3f4186.

EvidenceRead and GlobalEventQueryHead follow the main component ablation's
global path. No LoRA, proposal reader, boundary refiner or score fusion exists.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn


class EvidenceRead(nn.Module):
    def __init__(self, width=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, n_heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(width, width * 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width * 2, width)
        )
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feedforward = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, state, memory, valid=None):
        key_padding_mask = ~valid.bool() if valid is not None else None
        evidence, attention = self.attention(
            state, memory, memory, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=False,
        )
        state = self.norm_attention(state + self.dropout(evidence))
        state = self.norm_feedforward(state + self.dropout(self.feedforward(state)))
        attn_out = attention.mean(dim=1).squeeze(1) if attention.ndim == 4 else attention.mean(dim=1)
        return state, attn_out


class GlobalEventQueryHead(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        self.text_read = EvidenceRead(256)
        self.video_reads = nn.ModuleList([EvidenceRead(256) for _ in range(depth)])
        self.global_head = nn.Linear(256, 1)

    def forward(self, text, text_valid, video, video_valid):
        state, _ = self.text_read(self.token.expand(text.shape[0], -1, -1), text, text_valid)
        if len(video) != len(self.video_reads):
            raise RuntimeError("Missing encoder feature levels")
        for reader, memory in zip(self.video_reads, video):
            state, _ = reader(state, memory, video_valid)
        return self.global_head(state[:, 0]).squeeze(-1)


def state_digest(module):
    """Include every parameter and buffer, including frozen auxiliary routes."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class FrozenGEQ(nn.Module):
    def __init__(self, frozen, backbone):
        super().__init__()
        self.frozen = frozen.requires_grad_(False).eval()
        self.backbone = backbone
        layers = frozen.base.transformer.encoder.layers
        self.geq = GlobalEventQueryHead(len(layers))
        self.context = None
        self.video_length = 0
        self.handles = [frozen.base.input_txt_proj.register_forward_hook(self.capture_text)]
        self.handles.extend(layer.register_forward_hook(self.capture_video) for layer in layers)

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def capture_text(self, module, args, output):
        if self.context is not None:
            self.context["text"] = output.detach()

    def capture_video(self, module, args, output):
        if self.context is not None:
            offset = int(self.backbone in ("qd_detr", "cg_detr"))
            memory = output.transpose(0, 1)[:, offset:offset + self.video_length].detach()
            self.context["video"].append(memory)

    def forward(self, **inputs):
        if self.context is not None:
            raise RuntimeError("Reentrant forward")
        video_valid = inputs["src_vid_mask"].bool()
        text_valid = inputs["src_txt_mask"].bool()
        if not bool(video_valid.any(1).all() and text_valid.any(1).all()):
            raise ValueError("Empty feature sequence")
        self.video_length = video_valid.shape[1]
        context = {"video": []}
        self.context = context
        try:
            with torch.no_grad():
                output = self.frozen(**inputs)
        finally:
            self.context = None
        for memory in context["video"]:
            if memory.shape[:2] != video_valid.shape or memory.shape[-1] != 256:
                raise RuntimeError(f"Incorrect video-token selection: {memory.shape}, {video_valid.shape}")
        if context["text"].shape[:2] != text_valid.shape:
            raise RuntimeError("Text mask and projected text do not align")
        logit = self.geq(context["text"], text_valid, context["video"], video_valid)
        output = dict(output)
        output.update(pred_exist_logits=logit, pred_exist_logit=logit, pred_exist_score=logit.sigmoid())
        return output
