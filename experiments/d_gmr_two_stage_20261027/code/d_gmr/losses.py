"""Decoupled Two-Tier Objective for D-GMR.

Implements:
1. Native localization loss (preserved from native backbone)
2. GEQ existence BCE loss
3. Quality IoU regression loss
4. Two-Tier Decoupled Ranking Loss:
   - Intra-GT redundancy suppression (penalizes high-scoring duplicates of the same GT)
   - Inter-GT plurality parity (ensures every distinct GT's best proposal beats background)
5. Set diversity and evidence BCE loss
"""
import torch
from torch.nn import functional as F

from gemr.losses.evidence_loss import evidence_loss, evidence_targets
from gemr.losses.quality_loss import quality_loss
from gemr.losses.set_loss import set_loss
from gemr.models.temporal_relation import temporal_iou


def two_tier_decoupled_ranking_loss(logits, proposals, valid, metas,
                                    intra_margin=0.15, inter_margin=0.15,
                                    pos_iou_thresh=0.5, dup_iou_thresh=0.3, bg_iou_thresh=0.2):
    """
    logits: (B, K)
    proposals: (B, K, 2) or (B, K, 3)
    valid: (B, K) bool
    metas: list of meta dicts with 'relevant_windows'
    """
    device = logits.device
    zero = logits.sum() * 0.0
    intra_losses = []
    inter_losses = []

    for batch, meta in enumerate(metas):
        gts = meta.get('relevant_windows', [])
        if not gts:
            continue
        gts_tensor = proposals.new_tensor(gts).reshape(-1, 2) # (G, 2)
        v_mask = valid[batch]
        if not v_mask.any():
            continue

        b_proposals = proposals[batch, v_mask, :2] # (N, 2)
        b_logits = logits[batch, v_mask] # (N,)

        # Compute IoU matrix (N, G)
        ious = temporal_iou(b_proposals, gts_tensor) # (N, G)

        best_prop_per_gt = {}
        for g_idx in range(len(gts)):
            gt_ious = ious[:, g_idx]
            max_val, max_idx = gt_ious.max(0)
            if float(max_val.detach()) >= pos_iou_thresh:
                best_prop_per_gt[g_idx] = (int(max_idx.detach()), float(max_val.detach()))

        # 1. Intra-GT redundancy suppression
        for g_idx, (best_idx, _) in best_prop_per_gt.items():
            gt_ious = ious[:, g_idx]
            dup_indices = torch.where((gt_ious >= dup_iou_thresh) & (torch.arange(len(gt_ious), device=device) != best_idx))[0]
            if len(dup_indices) > 0:
                best_logit = b_logits[best_idx]
                dup_logits = b_logits[dup_indices]
                # best_logit should be higher than dup_logits + intra_margin
                diff = best_logit - dup_logits - intra_margin
                loss = F.softplus(-diff).mean()
                intra_losses.append(loss)

        # 2. Inter-GT plurality parity
        max_iou_per_prop = ious.max(dim=1).values # (N,)
        bg_indices = torch.where(max_iou_per_prop < bg_iou_thresh)[0]

        if len(best_prop_per_gt) > 0 and len(bg_indices) > 0:
            bg_logits = b_logits[bg_indices]
            for g_idx, (best_idx, _) in best_prop_per_gt.items():
                best_logit = b_logits[best_idx]
                diff = best_logit - bg_logits - inter_margin
                loss = F.softplus(-diff).mean()
                inter_losses.append(loss)

    intra_val = torch.stack(intra_losses).mean() if intra_losses else zero
    inter_val = torch.stack(inter_losses).mean() if inter_losses else zero
    return intra_val, inter_val


def objective(adapter, output, metas, targets, config, epoch=0):
    native, native_details = adapter.native_loss(output['frozen']['native_output'], targets, metas, epoch)
    presence_target = output['presence_logit'].new_tensor(
        [bool(meta['relevant_windows']) for meta in metas])
    presence = F.binary_cross_entropy_with_logits(output['presence_logit'].reshape(-1), presence_target)
    proposals, valid = output['selected_proposals'], output['selected_valid']
    quality_target = torch.zeros_like(output['quality_logit'])
    for batch, meta in enumerate(metas):
        ground_truth = proposals.new_tensor(meta['relevant_windows']).reshape(-1, 2)
        if len(ground_truth):
            quality_target[batch] = temporal_iou(
                proposals[batch, :, :2], ground_truth).max(-1).values.detach()
    backbone = output['frozen']
    evidence_target = evidence_targets(backbone['times'], backbone['video_valid'], metas)
    set_value, diagnostics = set_loss(output, metas, config['coverage_iou_threshold'])

    intra_rank, inter_rank = two_tier_decoupled_ranking_loss(
        output['ranking_logits'], proposals, valid, metas,
        intra_margin=config.get('intra_margin', 0.15),
        inter_margin=config.get('inter_margin', 0.15),
        pos_iou_thresh=config.get('coverage_iou_threshold', 0.5),
        dup_iou_thresh=0.3, bg_iou_thresh=0.2)

    parts = dict(
        native=native,
        presence=presence,
        quality=quality_loss(output['quality_logit'], quality_target, valid),
        intra_ranking=intra_rank,
        inter_ranking=inter_rank,
        set=set_value,
        evidence=evidence_loss(output['evidence_logit'], evidence_target,
                               backbone['video_valid'], config['evidence_positive_weight']))
    contributions = {name: config['loss_weights'][name]*value for name, value in parts.items()}
    total = sum(contributions.values())
    assert torch.isfinite(total) and all(torch.isfinite(value) for value in parts.values()), parts
    return total, parts, contributions, diagnostics, native_details
