"""Shared differentiable GEQ Full evidence head; native localization stays intact."""
from .clip import GMRWithClipHead
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



class GEQFull(GMRWithClipHead):
    def __init__(self, base, spec, clip_head=False):
        super().__init__(base.requires_grad_(True), spec, clip_head=clip_head)
        self.geq = GlobalEventQueryHead(self.depth)

    def forward(self, **inputs):
        native, text, video = self.evidence(inputs)
        logit = self.geq(text, inputs['src_txt_mask'], video, inputs['src_vid_mask'])
        return {**native, 'native_output': native, 'geq_logit': logit,
                'pred_exist_logits': logit, 'pred_exist_logit': logit,
                **self.clip_output(text, video, inputs)}
