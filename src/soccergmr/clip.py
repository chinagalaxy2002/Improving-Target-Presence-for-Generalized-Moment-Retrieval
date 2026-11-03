"""Query-conditioned dense clip supervision, independent of inference gates."""
import torch
from torch import nn
from torch.nn import functional as F


class ClipHead(nn.Module):
    def __init__(self, video_width=256, text_width=256, width=256):
        super().__init__()
        self.video = nn.Linear(video_width, width)
        self.text = nn.Linear(text_width, width)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, video, text, video_valid, text_valid):
        if video.shape[:2] != video_valid.shape or text.shape[:2] != text_valid.shape:
            raise ValueError('Clip evidence/mask shape mismatch')
        mask = text_valid.to(text.dtype).unsqueeze(-1)
        query = (text * mask).sum(1) / mask.sum(1).clamp_min(1)
        v, q = self.video(video), self.text(query).unsqueeze(1)
        return self.output(self.norm(v + q + v * q)).squeeze(-1)


def masked_clip_loss(logits, labels, valid):
    if logits.shape != labels.shape or logits.shape != valid.shape:
        raise ValueError('Clip logits/labels/mask shape mismatch')
    mask = valid.to(logits.dtype)
    losses = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype), reduction='none')
    return ((losses * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()


class GMRWithClipHead(nn.Module):
    """Capture first native encoder pass; never include global/padded tokens."""
    def __init__(self, base, spec, clip_head=True):
        super().__init__()
        from .backbones.registry import encoder_root, encoder_layers
        self.base, self.spec = base, spec
        root, layers = encoder_root(base, spec.name), encoder_layers(base, spec.name)
        # Auxiliary initialization must not change the native/GEQ RNG sequence.
        with torch.random.fork_rng(devices=[]):
            self.clip_head = ClipHead(width=layers[-1].self_attn.embed_dim,
                                      video_width=layers[-1].self_attn.embed_dim,
                                      text_width=layers[-1].self_attn.embed_dim) if clip_head else None
        self.depth = len(layers)
        self.context = None
        self.handles = [root.input_txt_proj.register_forward_hook(self.capture_text)]
        self.handles.extend(layer.register_forward_hook(self.capture_video) for layer in layers)

    def capture_text(self, module, args, output):
        if self.context is not None and 'text' not in self.context:
            self.context['text'] = output

    def capture_video(self, module, args, output):
        if self.context is not None:
            start = self.spec.encoder_offset
            self.context['video'].append(output.transpose(0, 1)[:, start:start+self.video_length])

    def evidence(self, inputs):
        if self.context is not None:
            raise RuntimeError('Reentrant evidence forward')
        self.video_length = inputs['src_vid_mask'].shape[1]
        context = {'video': []}
        self.context = context
        try:
            native = self.base(**inputs)
        finally:
            self.context = None
        passes = self.spec.encoder_passes + int(bool(self.spec.extra_encoder_input) and self.spec.extra_encoder_input in inputs)
        if len(context['video']) != self.depth * passes:
            raise RuntimeError(f"Unexpected encoder passes: {len(context['video'])}/{self.depth}")
        video = context['video'][:self.depth]
        if any(x.shape[:2] != inputs['src_vid_mask'].shape for x in video):
            raise RuntimeError('Incorrect encoder evidence dimensions')
        return native, context['text'], video

    def clip_output(self, text, video, inputs):
        if self.clip_head is None:
            return {}
        return {'clip_logits': self.clip_head(video[-1], text, inputs['src_vid_mask'], inputs['src_txt_mask'])}

    def forward(self, **inputs):
        native, text, video = self.evidence(inputs)
        return {**native, 'native_output': native, **self.clip_output(text, video, inputs)}

    def close(self):
        for handle in self.handles:
            handle.remove()


def wrap_gmr(base, spec, config):
    return GMRWithClipHead(base, spec) if config.get('clip', {}).get('gmr_weight', 0) > 0 else base


def clip_enabled(config):
    return any(config.get('clip', {}).get(k, 0) > 0 for k in ('gmr_weight', 'geq_weight'))
