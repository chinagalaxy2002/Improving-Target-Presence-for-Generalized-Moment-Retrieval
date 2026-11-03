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
            self.clip_head = ClipHead(width=getattr(getattr(layers[-1],'self_attn',None),'embed_dim',256),
                                      video_width=getattr(getattr(layers[-1],'self_attn',None),'embed_dim',256),
                                      text_width=getattr(getattr(layers[-1],'self_attn',None),'embed_dim',256)) if clip_head else None
        self.depth = len(layers)
        self.context = None
        self.handles = [(root.text_encoder if spec.name=='ld_detr' else root.input_txt_proj).register_forward_hook(self.capture_text)]
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
        if self.spec.name=='taskweave' and (inputs.get('epoch') is None or inputs['epoch']>=100):passes=6
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
    c=config.get('clip', {})
    enabled=(c.get('loss_weight',0)>0 and c.get('method')!='native') if 'method' in c else c.get('gmr_weight',0)>0
    return GMRWithClipHead(base, spec) if enabled else base


def clip_enabled(config):
    c=config.get('clip',{})
    if 'method' in c:return c['method']!='native' and c.get('geq_weight',0)>0
    return any(c.get(k,0)>0 for k in ('gmr_weight','geq_weight'))


def balanced_reduce(losses, positive, valid):
    """Each video has unit weight; average only nonempty region groups."""
    valid=valid.bool(); positive=positive.bool() & valid; negative=~positive & valid
    np_=positive.sum(1); nn_=negative.sum(1)
    pos=(losses*positive).sum(1)/np_.clamp_min(1)
    neg=(losses*negative).sum(1)/nn_.clamp_min(1)
    groups=(np_>0).to(losses.dtype)+(nn_>0).to(losses.dtype)
    return ((pos+neg)/groups.clamp_min(1)).mean()


def clip_auxiliary_loss(logits, targets, valid, settings):
    method=settings['method']
    hard=targets['saliency_all_labels']
    labels=targets['clip_soft_labels'] if method=='soft' else hard
    if logits.shape!=labels.shape or valid.shape!=labels.shape:raise ValueError('Clip loss dimensions')
    positive=labels>0
    losses=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
    prob=logits.sigmoid()
    if method=='focal':
        pt=torch.where(positive,prob,1-prob)
        losses=losses*(1-pt).pow(settings['gamma'])
    balanced=balanced_reduce(losses,positive,valid)
    total=balanced; dice=logits.sum()*0; rank=logits.sum()*0
    if method=='dice':
        mask=valid.to(prob.dtype); active=(hard.bool() & valid.bool()).any(1)
        intersection=(prob*hard*mask).sum(1)
        d=1-(2*intersection+1e-6)/((prob*mask).sum(1)+(hard*mask).sum(1)+1e-6)
        dice=d[active].mean() if active.any() else logits.sum()*0
        total=total+settings['dice_weight']*dice
    if method=='ranking':
        terms=[]
        for z,y,v in zip(logits,hard,valid.bool()):
            pos=z[(y>0)&v];neg=z[(y==0)&v]
            if pos.numel() and neg.numel():
                hardest=neg.topk(min(settings['topk'],neg.numel())).values
                terms.append(F.softplus(settings['margin']-pos.mean()+hardest.mean()))
        rank=torch.stack(terms).mean() if terms else logits.sum()*0
        total=total+settings['rank_weight']*rank
    return total, dict(balanced=balanced.detach(),dice=dice.detach(),ranking=rank.detach())


def load_native_gmr(base, spec, config, state):
    """Strictly validate the entire Stage-1 state, then discard auxiliary head.

    No identity bypass and no strict=False. Temporary auxiliary construction
    preserves RNG; no auxiliary module survives into GEQ.
    """
    wrapped=wrap_gmr(base,spec,config)
    wrapped.load_state_dict(state,strict=True)
    if isinstance(wrapped,GMRWithClipHead):
        wrapped.close()
        return wrapped.base
    return wrapped
