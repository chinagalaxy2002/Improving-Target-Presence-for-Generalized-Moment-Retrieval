"""BAM-DETR SoccerGMR adaptation."""
import torch
from torch import nn


class BAMModel(nn.Module):
    def __init__(self, native, dimension, options=None):
        super().__init__()
        self.native = native
        self.exist_head = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, 1))
        self._decoder_features = None
        self.native.class_embed[-1].register_forward_pre_hook(self._capture)

        from soccergmr.modules.interlayer_bind import build_interlayer_adapter
        hidden_dim = getattr(native, "hidden_dim", 256)
        num_queries = getattr(options, "num_queries", 10) if options is not None else 10
        self.query_cgp = build_interlayer_adapter(options, hidden_dim=hidden_dim, num_queries=num_queries)
        self.native.transformer.decoder.interlayer_adapter = self.query_cgp

    def _capture(self, module, args):
        self._decoder_features = args[0]

    def forward(self, **inputs):
        src_vid = inputs.get("src_vid")
        src_txt = inputs.get("src_txt")
        src_txt_mask = inputs.get("src_txt_mask")
        src_vid_mask = inputs.get("src_vid_mask")
        if self.query_cgp is not None and src_txt is not None:
            self.query_cgp.clear_diagnostics()
            txt_proj = self.native.input_txt_proj(src_txt)
            semantic_mask = src_txt_mask.bool()
            semantic_count = semantic_mask.sum(dim=1, keepdim=True).clamp_min(1)
            query_semantic = (txt_proj * semantic_mask.unsqueeze(-1).to(txt_proj.dtype)).sum(dim=1) / semantic_count.to(txt_proj.dtype)
            self.native.transformer.decoder.interlayer_adapter_kwargs = {
                "query_semantic": query_semantic,
                "video_length": src_vid.shape[1],
            }
        self._decoder_features = None
        try:
            out = self.native(**inputs)
            if self._decoder_features is None:
                raise RuntimeError('Missing BAM final decoder representation')
            out['pred_exist_logits'] = self.exist_head(self._decoder_features.amax(dim=1)).squeeze(-1)
            if self.query_cgp is not None and self.query_cgp.last_output is not None:
                out["query_cgp_temporal_attention"] = self.query_cgp.last_output.temporal_attention
                out["query_cgp_basis_weights"] = self.query_cgp.last_output.basis_weights
                out["query_cgp_video_mask"] = src_vid_mask.bool()
            return out
        finally:
            self._decoder_features = None


class BAMLoss(nn.Module):
    """Preserve BAM localization, quality, boundary and saliency supervision."""
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.weight_dict = {
            **native.weight_dict,
            'loss_exist': 1.0,
            'loss_query_cgp_bind': 0.2,
            'loss_query_cgp_route': 0.01,
        }

    def quality(self, outputs, targets):
        from bam_detr.span_utils import temporal_iou, span_cxw_to_xx
        predicted = outputs['pred_spans']
        quality = outputs['pred_quality'].squeeze(-1)
        reference = []
        for spans, target in zip(predicted.detach(), targets['span_labels']):
            gt = target['spans']
            if len(gt):
                iou, _ = temporal_iou(span_cxw_to_xx(spans), span_cxw_to_xx(gt))
                reference.append(iou.max(dim=1).values)
            else:
                reference.append(torch.zeros_like(quality[0]))
        reference = torch.stack(reference).flatten()
        delta = (quality.flatten() - reference).abs()
        groups = (reference > .7, (reference <= .7) & (reference > .3), reference <= .3)
        return sum(delta[mask].mean() if mask.any() else delta.sum() * 0 for mask in groups) / 3

    def saliency(self, outputs, targets):
        from torch.nn import functional as F
        valid = outputs['video_mask'].bool()
        scores = outputs['saliency_scores']
        neg_scores = outputs['saliency_scores_neg']
        groups = targets['video_group']
        # Native rotates queries by one row; equal-video rotations are not
        # assumed negative. A singleton also has no valid synthetic negative.
        sim = outputs.get('sim', {})
        negative_index = sim.get('negative_idx') if isinstance(sim, dict) else None
        paired_groups = groups[negative_index] if negative_index is not None else groups.roll(-1)
        neg_valid = groups.ne(paired_groups)[:, None] & valid
        loss_negative = (F.softplus(neg_scores) * neg_valid).sum(dim=1).mean()
        # A real null query independently supervises its own valid frames.
        real_null = targets['exist_label'].eq(0)[:, None] & valid
        loss_null = (F.softplus(scores) * real_null).sum(dim=1).mean()
        positive = targets['saliency_all_labels'].bool() & valid
        combined = torch.cat((scores, neg_scores), 1) / .5
        combined_valid = torch.cat((valid, neg_valid), 1)
        combined = combined.masked_fill(~combined_valid, float('-inf'))
        log_probs = combined - torch.logsumexp(combined, dim=1, keepdim=True)
        # Mask before multiplication: 0 * -inf is NaN.
        positive_log_probs = log_probs[:, :scores.shape[1]].masked_fill(~positive, 0)
        ranking = -(positive_log_probs.sum(1) / positive.sum(1).clamp_min(1)).mean() / 12
        pair_valid = targets['saliency_pair_mask'].bool()
        if pair_valid.any():
            rows = torch.arange(len(scores), device=scores.device)[:, None]
            pos = scores[rows, targets['saliency_pos_labels']]
            neg = scores[rows, targets['saliency_neg_labels']]
            pair = (self.native.saliency_margin + neg - pos).clamp_min(0)[pair_valid].mean() * 2
        else:
            pair = scores.sum() * 0
        return loss_negative + loss_null + ranking + pair

    def boundary(self, outputs, targets):
        from .loss import select
        pieces = []
        batch = len(targets['exist_label'])
        # The native boundary grid must use each video's real, unpadded length.
        for index in range(batch):
            ix = torch.tensor([index], device=outputs['pred_spans'].device)
            out = select(outputs, ix, batch)
            target = select(targets, ix, batch)
            length = int(out['video_mask'].sum())
            out['video_mask'] = out['video_mask'][:, :length]
            out['memory_boundary'] = out['memory_boundary'][:, :length]
            pieces.append(self.native.loss_boundary(out, target, None)['loss_boundary'])
        return torch.stack(pieces).mean()

    def forward(self, outputs, targets, epoch=0):
        from torch.nn import functional as F
        from .loss import select
        logit = outputs['pred_exist_logits'].reshape(-1)
        batch = len(logit)
        positive = torch.where(targets['exist_label'] > 0)[0]
        losses = {}
        primary_matches = None
        levels = [outputs] + list(outputs.get('aux_outputs', []))
        for level, out in enumerate(levels):
            suffix = '' if level == 0 else f'_{level - 1}'
            matches = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)) for _ in range(batch)]
            if len(positive):
                pos_out = select({k:v for k,v in out.items() if k != 'aux_outputs'}, positive, batch)
                pos_targets = select(targets, positive, batch)
                pos_matches = self.native.matcher(pos_out, pos_targets)
                for index, match in zip(positive.tolist(), pos_matches):
                    matches[index] = match
                for key, value in self.native.loss_spans(pos_out, pos_targets, pos_matches).items():
                    losses[key + suffix] = value
            else:
                losses['loss_span'+suffix] = logit.sum() * 0
                losses['loss_giou'+suffix] = logit.sum() * 0
            losses['loss_label'+suffix] = self.native.loss_labels(out, targets, matches, log=False)['loss_label']
            losses['loss_quality'+suffix] = self.quality(out, targets)
            if level == 0:
                primary_matches = matches
        losses['loss_boundary'] = self.boundary(outputs, targets)
        losses['loss_saliency'] = self.saliency(outputs, targets)
        losses['loss_exist'] = F.binary_cross_entropy_with_logits(logit, targets['exist_label'])
        if 'loss_query_cgp_bind' not in losses and 'query_cgp_temporal_attention' in outputs:
            from soccergmr.modules.dq_cgp import compute_query_cgp_loss
            cgp_losses = compute_query_cgp_loss(outputs, targets, indices=primary_matches)
            losses.update(cgp_losses)
        total = sum(value * self.weight_dict[key] for key, value in losses.items() if key in self.weight_dict)
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite BAM-DETR loss')
        return total, losses
