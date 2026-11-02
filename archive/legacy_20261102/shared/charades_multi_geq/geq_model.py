"""A GEQ head with a differentiable full mode and a strictly frozen control."""
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



class RouteGEQ(nn.Module):
    def __init__(self, base, backbone, mode):
        super().__init__()
        self.base = base.requires_grad_(mode == "full")
        self.backbone = backbone
        self.mode = mode
        layers = base.base.transformer.encoder.layers
        self.geq = GlobalEventQueryHead(len(layers))
        self.context = None
        self.video_length = 0
        self.handles = [base.base.input_txt_proj.register_forward_hook(self.capture_text)]
        self.handles.extend(layer.register_forward_hook(self.capture_video) for layer in layers)
        if mode == "frozen":
            self.base.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.mode == "frozen":
            self.base.eval()
        return self

    def capture_text(self, module, args, output):
        if self.context is not None and "text" not in self.context:
            self.context["text"] = output if self.mode == "full" else output.detach()

    def capture_video(self, module, args, output):
        if self.context is not None:
            offset = int(self.backbone in ("qd_detr", "cg_detr"))
            memory = output.transpose(0, 1)[:, offset:offset + self.video_length]
            self.context["video"].append(memory if self.mode == "full" else memory.detach())

    def forward(self, **inputs):
        # Native FlashVTG eval decodes one video at a time. The frozen trunk
        # remains in eval; GEQ gradients still accumulate over the full batch.
        batch = inputs["src_vid"].shape[0]
        if self.backbone == "flashvtg" and self.mode == "frozen" and batch > 1:
            def select(value, index):
                if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch:
                    return value[index:index+1]
                if isinstance(value, dict):
                    return {k: select(v, index) for k, v in value.items()}
                if isinstance(value, list) and len(value) == batch:
                    return value[index:index+1]
                return value
            outputs = [self.forward(**{k: select(v, i) for k, v in inputs.items()}) for i in range(batch)]
            logit = torch.cat([o["pred_exist_logits"].reshape(-1) for o in outputs])
            return dict(native_output={"per_sample_outputs": [o["native_output"] for o in outputs]},
                        pred_exist_logits=logit, pred_exist_logit=logit,
                        pred_exist_score=logit.sigmoid())
        if self.context is not None:
            raise RuntimeError("Reentrant GEQ forward")
        video_valid = inputs["src_vid_mask"].bool()
        text_valid = inputs["src_txt_mask"].bool()
        self.video_length = video_valid.shape[1]
        context = {"video": []}
        self.context = context
        try:
            with torch.no_grad() if self.mode == "frozen" else nullcontext():
                native = self.base(**inputs)
        finally:
            self.context = None
        depth = len(self.geq.video_reads)
        video = context["video"]
        if len(video) != depth or any(x.shape[:2] != video_valid.shape for x in video):
            raise RuntimeError("Incorrect encoder video evidence")
        logit = self.geq(context["text"], text_valid, video, video_valid)
        output = dict(native)
        output.update(native_output=native, pred_exist_logits=logit,
                      pred_exist_logit=logit, pred_exist_score=logit.sigmoid())
        return output
