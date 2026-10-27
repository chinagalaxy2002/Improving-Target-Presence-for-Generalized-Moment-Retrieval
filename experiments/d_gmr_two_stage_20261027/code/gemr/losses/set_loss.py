"""One-to-one instance coverage and supervised competition/cooperation edges.

Hungarian cost prioritizes the number of matches above the fixed IoU threshold,
then their summed IoU. Unsupported GTs create no positive labels. A matched
candidate keeps its assigned instance even when two GT intervals overlap.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from gemr.models.temporal_relation import temporal_iou


def set_loss(output, metas, threshold=.5):
    qlogit, rank = output['quality_logit'], output['ranking_logits']
    zero = qlogit.sum()*0 + output['competition_logits'].sum()*0 + output['cooperation_logits'].sum()*0
    terms = []
    stats = dict(gt_total=0, gt_without_any_candidate=0, gt_without_distinct_candidate=0,
                 matched_instances=0, duplicate_candidates=0)
    for b, meta in enumerate(metas):
        gt = qlogit.new_tensor(meta['relevant_windows']).reshape(-1, 2)
        ids = output['selected_valid'][b].nonzero().flatten()
        stats['gt_total'] += len(gt)
        if not len(gt):
            continue
        if not len(ids):
            stats['gt_without_any_candidate'] += len(gt)
            stats['gt_without_distinct_candidate'] += len(gt)
            continue
        iou = temporal_iou(output['selected_proposals'][b, ids, :2], gt)
        eligible = iou >= threshold
        # One eligible edge is worth more than any possible sum of IoU gains.
        cost = -((min(iou.shape)+1)*eligible.float()+iou)
        cand, instance = linear_sum_assignment(cost.detach().cpu().numpy())
        cand = torch.as_tensor(cand, device=iou.device)
        instance = torch.as_tensor(instance, device=iou.device)
        supported = eligible[cand, instance]
        cand, instance = cand[supported], instance[supported]
        stats['gt_without_any_candidate'] += int((~eligible.any(0)).sum())
        stats['gt_without_distinct_candidate'] += len(gt)-len(cand)
        stats['matched_instances'] += len(cand)
        if not len(cand):
            continue
        best, owner = iou.max(-1)
        owner = owner.clone()
        owner[cand] = instance
        reliable = best >= threshold
        chosen = torch.zeros(len(ids), dtype=torch.bool, device=iou.device)
        chosen[cand] = True
        # Explicit coverage: all supported distinct GT representatives receive
        # positive quality supervision, without demanding uncovered instances.
        pieces = [F.binary_cross_entropy_with_logits(qlogit[b, ids[cand]], torch.ones_like(qlogit[b, ids[cand]]))]
        duplicate = reliable & ~chosen
        stats['duplicate_candidates'] += int(duplicate.sum())
        for local, inst in zip(cand, instance):
            duplicates = duplicate & (owner == inst)
            # Do not suppress an unmatched candidate when it could cover an
            # otherwise unsupported distinct GT (important for overlapping GT).
            missing = torch.ones(len(gt), dtype=torch.bool, device=iou.device)
            missing[instance] = False
            if missing.any():
                duplicates &= ~eligible[:, missing].any(-1)
            if duplicates.any():
                pieces.append(F.softplus(-(rank[b, ids[local]]-rank[b, ids[duplicates]])).mean())
        # Reliable candidates supervise distinct branch semantics. Temporal
        # overlap alone never determines instance equality or suppression.
        pair = reliable[:, None] & reliable[None, :]
        pair &= ~torch.eye(len(ids), dtype=torch.bool, device=iou.device)
        if pair.any():
            same = (owner[:, None] == owner[None, :]).float()
            comp = output['competition_logits'][b][ids[:, None], ids[None, :]]
            coop = output['cooperation_logits'][b][ids[:, None], ids[None, :]]
            pieces += [F.binary_cross_entropy_with_logits(comp[pair], same[pair]),
                       F.binary_cross_entropy_with_logits(coop[pair], 1-same[pair])]
        terms.append(torch.stack(pieces).mean())
    return (torch.stack(terms).mean() if terms else zero), stats
