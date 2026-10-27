"""FlashMMR-style temporal GRU quality and GT agreement auxiliary supervision."""
import torch
from torch import nn
from torch.nn import functional as F

@torch.no_grad()
def quality_targets(spans, valid, metadata):
    target = spans.new_zeros(spans.shape[:2])
    for b, meta in enumerate(metadata):
        gt = spans.new_tensor(meta['relevant_windows']).reshape(-1, 2)
        gt = gt[torch.isfinite(gt).all(1) & (gt[:, 1] > gt[:, 0])]
        if not len(gt): continue
        xy = spans[b, :, :2]
        inter = (torch.minimum(xy[:, None, 1], gt[None, :, 1])-torch.maximum(xy[:, None, 0], gt[None, :, 0])).clamp_min(0)
        union = (xy[:, 1]-xy[:, 0])[:, None] + (gt[:, 1]-gt[:, 0])[None] - inter
        target[b] = (inter/union.clamp_min(1e-8)).max(1).values
    return target.masked_fill(~valid, 0)

class PostVerification(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(256, 128, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(.3), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(.3), nn.Linear(32, 1))
    def forward(self, video, mask, spans, valid):
        # Constrained adjustment is clipping plus outward 2s grid alignment for
        # feature selection only. No learned boundary correction or exported edit.
        seq, lengths = [], []
        for b in range(len(video)):
            n = int(mask[b].sum())
            for k in range(spans.shape[1]):
                if not bool(valid[b, k]) or n == 0:
                    seq.append(video[b, :1]*0); lengths.append(1); continue
                s, e = spans[b, k, :2].detach()
                a = max(0, min(n-1, int(torch.floor(s/2))))
                z = max(a+1, min(n, int(torch.ceil(e/2))))
                frames = video[b, a:z]
                seq.append(frames); lengths.append(len(frames))
        padded = nn.utils.rnn.pad_sequence(seq, batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        logit = self.classifier(hidden[-1]).reshape(spans.shape[:2])
        return dict(quality_logit=logit, quality_score=logit.sigmoid(),
                    moment_sequence=padded.reshape(len(video), spans.shape[1], padded.shape[1], -1),
                    moment_lengths=torch.tensor(lengths, device=video.device).reshape(spans.shape[:2]))

def representation_loss(video, mask, metadata):
    """Row CE over valid frames; positives share an actual GT interval.

    Background is only a contrastive denominator. No background-background
    positives and no self pairs. Multiple overlapping GT memberships are allowed.
    """
    pieces = []
    for b, meta in enumerate(metadata):
        n = int(mask[b].sum()); feat = F.normalize(video[b, :n], dim=-1)
        gt = video.new_tensor(meta['relevant_windows']).reshape(-1, 2)
        if not len(gt) or n < 2: continue
        centers = (torch.arange(n, device=video.device)+.5)*2.
        member = (centers[:, None] >= gt[None, :, 0]) & (centers[:, None] < gt[None, :, 1])
        agreement = (member.float() @ member.float().T) > 0
        agreement.fill_diagonal_(False)
        anchors = agreement.any(1)
        if not anchors.any(): continue
        similarity = feat @ feat.T
        similarity = similarity.masked_fill(torch.eye(n, device=video.device, dtype=torch.bool), -1e4)
        target = agreement.float()/agreement.sum(1, keepdim=True).clamp_min(1)
        pieces.append(-(target[anchors]*F.log_softmax(similarity[anchors], dim=-1)).sum(-1).mean())
    return torch.stack(pieces).mean() if pieces else video.sum()*0

def losses(out, metadata):
    target = quality_targets(out['candidate_spans'], out['candidate_valid'], metadata)
    valid = out['candidate_valid']
    quality = ((out['quality_score']-target).square()*valid).sum()/valid.sum().clamp_min(1)
    representation = representation_loss(out['fusion_video'], out['fusion_valid'], metadata)
    return quality, representation
