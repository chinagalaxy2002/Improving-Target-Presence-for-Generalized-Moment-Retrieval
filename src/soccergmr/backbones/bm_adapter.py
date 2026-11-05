"""SoccerGMR adaptation of BM-DETR; its frame probabilities are not existence logits."""
import torch
from torch import nn
from torch.nn import functional as F
from .loss import select


class BMModel(nn.Module):
    def __init__(self, native, dimension):
        super().__init__()
        self.native = native
        self.exist_head = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, 1))

        from soccergmr.modules.dq_cgp import DETRQueryCGP
        self.query_cgp = DETRQueryCGP(
            hidden_dim=dimension,
            num_basis=16,
            prompt_length=6,
            router_hidden_dim=256,
            frf_hidden_dim=512,
            temperature=1.0,
            beta=0.05,
        )
        self.native.transformer.decoder.interlayer_adapter = self.query_cgp

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
        out = self.native(**inputs)
        out['pred_exist_logits'] = self.exist_head(out['hs'][-1].amax(dim=1)).squeeze(-1)
        if self.query_cgp is not None and self.query_cgp.last_output is not None:
            out["query_cgp_temporal_attention"] = self.query_cgp.last_output.temporal_attention
            out["query_cgp_basis_weights"] = self.query_cgp.last_output.basis_weights
            out["query_cgp_video_mask"] = src_vid_mask.bool()
        return out


class BMLoss(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.weight_dict = {
            **native.weight_dict,
            'loss_exist': 1.0,
            'loss_query_cgp_bind': 0.2,
            'loss_query_cgp_route': 0.01,
        }

    def forward(self, outputs, targets, epoch=0):
        logit = outputs['pred_exist_logits'].reshape(-1)
        batch = len(logit)
        positives = torch.where(targets['exist_label'] > 0)[0]
        zero = logit.sum() * 0
        losses = {key: zero for key in self.native.weight_dict}
        # Match only real positives; every unmatched prediction, including every
        # prediction for a true null query, receives native background CE.
        indices = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)) for _ in range(batch)]
        if len(positives):
            selected = select({k: v for k, v in outputs.items() if k != 'hs'}, positives, batch)
            pos_targets = select(targets, positives, batch)
            matches = self.native.matcher(selected, pos_targets)
            for index, match in zip(positives.tolist(), matches):
                indices[index] = match
            losses.update(self.native.loss_spans(selected, pos_targets, matches))
            # Upstream assumes each batch contains unique videos. The shared
            # sampler has no such constraint, so same-video off-diagonal
            # pairs are excluded rather than labelled as negative examples.
            video, text = selected['proj_vid_mem'], selected['proj_txt_mem']
            similarity = video @ text.T / self.native.temperature
            group = pos_targets['video_group']
            duplicate = group[:, None].eq(group[None, :])
            duplicate.fill_diagonal_(False)
            similarity = similarity.masked_fill(duplicate, float('-inf'))
            labels = torch.arange(len(similarity), device=similarity.device)
            losses['loss_contrastive_align'] = F.cross_entropy(similarity, labels)
            valid_pairs = torch.where(pos_targets['saliency_pair_mask'] > 0)[0]
            if len(valid_pairs):
                pair_out = select(selected, valid_pairs, len(positives))
                pair_targets = select(pos_targets, valid_pairs, len(positives))
                losses.update(self.native.loss_saliency(pair_out, pair_targets, None))
        losses.update(self.native.loss_labels(outputs, targets, indices, log=False))
        # Probability guidance uses all valid frames. True null rows contribute
        # only the background term, avoiding division by zero for no foreground.
        mask = targets['logit_mask']
        foreground = targets['saliency_all_labels'] * mask
        background = mask - foreground
        probabilities = outputs['vid_logits']
        fg_count = foreground.sum(-1)
        bg_count = background.sum(-1)
        fg = (foreground * probabilities).sum(-1) / fg_count.clamp_min(1)
        bg = (background * probabilities).sum(-1) / bg_count.clamp_min(1)
        losses['loss_prob_guidance'] = ((self.native.prob_thd - fg) * (fg_count > 0) + bg * (bg_count > 0)).mean()
        losses['loss_exist'] = F.binary_cross_entropy_with_logits(logit, targets['exist_label'])
        if 'loss_query_cgp_bind' not in losses and 'query_cgp_temporal_attention' in outputs:
            from soccergmr.modules.dq_cgp import compute_query_cgp_loss
            cgp_losses = compute_query_cgp_loss(outputs, targets, matcher=self.native.matcher)
            losses.update(cgp_losses)
        total = sum(value * self.weight_dict[key] for key, value in losses.items() if key in self.weight_dict)
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite BM-DETR loss')
        return total, losses
