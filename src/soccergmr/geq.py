"""Shared differentiable GEQ Full evidence head; native localization stays intact."""
from contextlib import nullcontext
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



class GEQFull(nn.Module):
    def __init__(self, base, spec):
        super().__init__()
        from .backbones.registry import encoder_root
        self.base=base.requires_grad_(True)
        self.spec=spec
        root=encoder_root(base,spec.name)
        layers=root.transformer.encoder.layers
        self.geq=GlobalEventQueryHead(len(layers))
        self.context=None
        self.handles=[root.input_txt_proj.register_forward_hook(self.capture_text)]
        self.handles.extend(layer.register_forward_hook(self.capture_video) for layer in layers)
    def capture_text(self,module,args,output):
        if self.context is not None and 'text' not in self.context:
            self.context['text']=output
    def capture_video(self,module,args,output):
        if self.context is not None:
            start=self.spec.encoder_offset
            self.context['video'].append(output.transpose(0,1)[:,start:start+self.video_length])
    def forward(self, **inputs):
        if self.context is not None:raise RuntimeError('Reentrant GEQ forward')
        self.video_length=inputs['src_vid_mask'].shape[1]
        context={'video':[]};self.context=context
        try:native=self.base(**inputs)
        finally:self.context=None
        depth=len(self.geq.video_reads)
        if len(context['video'])!=depth*self.spec.encoder_passes:
            raise RuntimeError(f"Unexpected encoder passes: {len(context['video'])}/{depth}")
        video=context['video'][:depth]
        if any(x.shape[:2]!=inputs['src_vid_mask'].shape for x in video):
            raise RuntimeError('Incorrect encoder evidence dimensions')
        logit=self.geq(context['text'],inputs['src_txt_mask'],video,inputs['src_vid_mask'])
        return {**native,'native_output':native,'geq_logit':logit,
                'pred_exist_logits':logit,'pred_exist_logit':logit}
    def close(self):
        for handle in self.handles:handle.remove()
