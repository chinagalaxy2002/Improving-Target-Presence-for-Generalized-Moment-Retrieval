"""Attached GEQ readouts over native GMR activations; native head is retained."""
import torch
from torch import nn
from common import OLD, module
EvidenceRead = module('geq_reference', OLD/'code/model.py').EvidenceRead

class GlobalEventQuery(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, 256)*.02)
        self.text_read = EvidenceRead(256)
        self.video_reads = nn.ModuleList(EvidenceRead(256) for _ in range(depth))
        self.global_head = nn.Linear(256, 1)
    def forward(self, text, tm, videos, vm):
        assert len(videos) == len(self.video_reads)
        state, _ = self.text_read(self.token.expand(len(text), -1, -1), text, tm)
        for read, video in zip(self.video_reads, videos): state, _ = read(state, video, vm)
        return self.global_head(state[:, 0]).squeeze(-1)

class JointGMR(nn.Module):
    def __init__(self, base, backbone, pv=False):
        super().__init__()
        self.base, self.backbone = base, backbone
        self.geq = GlobalEventQuery(len(base.transformer.encoder.layers))
        if pv:
            from pv import PostVerification
            self.pv = PostVerification()
        self.context = None
        self.handles = [base.input_txt_proj.register_forward_hook(self._text),
                        base.transformer.register_forward_hook(self._transformer)]
        self.handles += [layer.register_forward_hook(self._video) for layer in base.transformer.encoder.layers]
        self.requires_grad_(True)
    def _text(self, mod, args, out):
        if self.context is not None: self.context['text'] = out
    def _video(self, mod, args, out):
        if self.context is not None:
            offset = int(self.backbone in ('qd_detr', 'cg_detr'))
            self.context['videos'].append(out.transpose(0, 1)[:, offset:offset+self.length])
    def _transformer(self, mod, args, out):
        if self.context is not None:
            if self.backbone == 'flashvtg': final = out[0].transpose(0, 1)
            elif self.backbone in ('qd_detr', 'cg_detr'): final = out[2]
            else: final = out[1]
            self.context['final'] = final[:, :self.length]
    def candidates(self, out, durations):
        if self.backbone == 'flashvtg':
            if self.training:
                # Decode in original pyramid point order, before native score sort.
                coord, point = out['out_coord'], out['point']
                xy = coord * coord.new_tensor([-1., 1.]) * point[None, :, 3:4] + point[None, :, 0:1]
                xy = xy * 2.
                all_spans = torch.cat([xy, out['out_class'].sigmoid()], -1)
                valid = torch.cat(out['pymid_msk'], 1).bool()
            else:
                all_spans = out['_out']['boundary'].unsqueeze(0)
                valid = torch.ones(all_spans.shape[:2], dtype=torch.bool, device=all_spans.device)
        else:
            c, w = out['pred_spans'].unbind(-1)
            xy = torch.stack([c-w/2, c+w/2], -1) * durations[:, None, None]
            all_spans = torch.cat([xy, out['pred_logits'].softmax(-1)[..., :1]], -1)
            valid = torch.ones(all_spans.shape[:2], dtype=torch.bool, device=xy.device)
        xy = torch.minimum(all_spans[..., :2].clamp_min(0), durations[:, None, None])
        all_spans = torch.cat([xy, all_spans[..., 2:]], -1)
        valid = valid & torch.isfinite(all_spans).all(-1) & (xy[..., 1] > xy[..., 0])
        scores = all_spans[..., 2].masked_fill(~valid, -torch.inf)
        ids = scores.topk(min(8, scores.shape[1]), dim=1).indices
        return all_spans.gather(1, ids[..., None].expand(-1, -1, 3)), ids, valid.gather(1, ids), all_spans
    def forward(self, *, durations, **inputs):
        assert self.context is None
        self.length = inputs['src_vid_mask'].shape[1]
        ctx = {'videos': []}; self.context = ctx
        try:
            if self.backbone == 'cg_detr' and not self.training:
                inputs = dict(inputs); inputs.pop('targets', None)
            native = self.base(**inputs)
            # Preserve the entire original dict for its own criterion.
            logits = self.geq(ctx['text'], inputs['src_txt_mask'].bool(), ctx['videos'], inputs['src_vid_mask'].bool())
            spans, ids, valid, all_spans = self.candidates(native, durations)
            out = dict(native)
            out.update(native_output=native, pred_exist_logits=logits, pred_exist_score=logits.sigmoid(),
                       candidate_spans=spans, candidate_indices=ids, candidate_valid=valid,
                       candidate_seconds=all_spans, fusion_video=ctx['final'], fusion_valid=inputs['src_vid_mask'].bool())
            if '_out' in native:
                out['_out'] = dict(native['_out'], pred_exist_score=logits.sigmoid().detach().cpu())
            if hasattr(self, 'pv'):
                devices = [logits.device.index] if logits.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    out.update(self.pv(ctx['final'], inputs['src_vid_mask'].bool(), spans, valid))
            return out
        finally: self.context = None
