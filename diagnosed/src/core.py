"""Core mathematical and matching primitives for SoccerGMR diagnostic experiments.

Unified definitions:
- IoU >= 0.4 is considered a Hit.
- Bipartite maximum matching maximizes independent cardinality first, then maximizes IoU sum.
- Exact decomposition: K = H_n (independent hits) + D_n (redundant candidates) + U_n (unhit candidates)
- Exact ranking decomposition: M = N_missing + N_ranking@k + N_selected@k
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_tiou(box1: Tuple[float, float], box2: Tuple[float, float]) -> float:
    """Computes temporal Intersection over Union between two intervals [s1, e1] and [s2, e2]."""
    s1, e1 = box1
    s2, e2 = box2
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 0.0 else 0.0


def compute_tiou_matrix(preds: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """Computes pairwise temporal IoU matrix of shape (K, M).

    Args:
        preds: Array of shape (K, 2) with [start, end]
        gts: Array of shape (M, 2) with [start, end]
    Returns:
        np.ndarray of shape (K, M)
    """
    K = len(preds)
    M = len(gts)
    if K == 0 or M == 0:
        return np.zeros((K, M), dtype=np.float32)

    # Vectorized computation
    p_s = preds[:, 0:1]  # (K, 1)
    p_e = preds[:, 1:2]  # (K, 1)
    g_s = gts[np.newaxis, :, 0]  # (1, M)
    g_e = gts[np.newaxis, :, 1]  # (1, M)

    inter = np.maximum(0.0, np.minimum(p_e, g_e) - np.maximum(p_s, g_s))
    union = (p_e - p_s) + (g_e - g_s) - inter
    union = np.maximum(union, 1e-8)
    return inter / union


def evaluate_candidates_against_gt(ious: np.ndarray, iou_threshold: float = 0.4) -> Dict[str, Any]:
    """Matches candidates against GTs using bipartite matching.

    Args:
        ious: np.ndarray of shape (K, M)
        iou_threshold: default 0.4
    Returns:
        Dictionary with:
          - K: number of candidate queries
          - M: number of ground truth events
          - raw_hit_candidates: queries hitting >= 1 GT (T_n)
          - distinct_gts_hit: GTs hit by >= 1 query (|S_n|)
          - independent_hits: one-to-one matched GT count (H_n)
          - redundant_candidates: D_n = T_n - H_n
          - unhit_candidates: U_n = K - T_n
          - gt_hit_counts: list of len M showing number of queries hitting each GT (C_{nj})
          - matched_pairs: list of (query_idx, gt_idx, iou)
          - matched_query_indices: set of query indices in the match
          - matched_gt_indices: set of GT indices in the match
    """
    K, M = ious.shape
    if K == 0 or M == 0:
        return {
            'K': K,
            'M': M,
            'raw_hit_candidates': 0,
            'distinct_gts_hit': 0,
            'independent_hits': 0,
            'redundant_candidates': 0,
            'unhit_candidates': K,
            'gt_hit_counts': [0] * M,
            'matched_pairs': [],
            'matched_query_indices': set(),
            'matched_gt_indices': set(),
            'query_to_gt_map': {}
        }

    hit_mask = (ious >= iou_threshold)
    raw_hit_candidates = int(hit_mask.any(axis=1).sum())
    distinct_gts_hit = int(hit_mask.any(axis=0).sum())
    gt_hit_counts = hit_mask.sum(axis=0).tolist()

    # Bipartite matching: maximize matched edges with IoU >= threshold, then maximize IoU sum
    # Cost = -(ious + 1000.0) if hit else 0.0
    cost_matrix = np.where(hit_mask, -(ious + 1000.0), 0.0)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs = []
    matched_q = set()
    matched_g = set()
    q_to_g = {}

    for r, c in zip(row_ind, col_ind):
        if hit_mask[r, c]:
            matched_pairs.append((int(r), int(c), float(ious[r, c])))
            matched_q.add(int(r))
            matched_g.add(int(c))
            q_to_g[int(r)] = int(c)

    independent_hits = len(matched_pairs)
    redundant_candidates = raw_hit_candidates - independent_hits
    unhit_candidates = K - raw_hit_candidates

    assert K == independent_hits + redundant_candidates + unhit_candidates, (
        f"Decomposition identity failed: K={K} != {independent_hits} + {redundant_candidates} + {unhit_candidates}"
    )

    return {
        'K': K,
        'M': M,
        'raw_hit_candidates': raw_hit_candidates,
        'distinct_gts_hit': distinct_gts_hit,
        'independent_hits': independent_hits,
        'redundant_candidates': redundant_candidates,
        'unhit_candidates': unhit_candidates,
        'gt_hit_counts': gt_hit_counts,
        'matched_pairs': matched_pairs,
        'matched_query_indices': matched_q,
        'matched_gt_indices': matched_g,
        'query_to_gt_map': q_to_g
    }


def evaluate_topk_ranking(
    ious: np.ndarray,
    scores: np.ndarray,
    k_list: Tuple[int, ...] = (1, 3, 5, 10),
    iou_threshold: float = 0.4
) -> Dict[str, Any]:
    """Evaluates candidate generation vs ranking over Top-K budgets.

    Args:
        ious: (K, M)
        scores: (K,)
        k_list: sequence of k thresholds
        iou_threshold: 0.4
    Returns:
        Dictionary containing:
          - H_all: independent hits using all K candidates
          - N_missing: M - H_all (events never generated)
          - H_at_k: dict mapping k -> H_n^{@k}
          - N_ranking_at_k: dict mapping k -> H_all - H_at_k
          - N_selected_at_k: dict mapping k -> H_at_k
          - first_hit_ranks: list of len M with 1-based rank of first candidate hitting GT j (-1 if not hit)
    """
    K, M = ious.shape
    order = np.argsort(-scores)  # descending order of native scores
    sorted_ious = ious[order]

    # Full set evaluation
    full_eval = evaluate_candidates_against_gt(sorted_ious, iou_threshold)
    h_all = full_eval['independent_hits']
    n_missing = M - h_all

    h_at_k = {}
    n_ranking_at_k = {}
    n_selected_at_k = {}

    for k in k_list:
        sub_k = min(k, K)
        topk_eval = evaluate_candidates_against_gt(sorted_ious[:sub_k], iou_threshold)
        h_k = topk_eval['independent_hits']
        h_at_k[k] = h_k
        n_selected_at_k[k] = h_k
        n_ranking_at_k[k] = h_all - h_k

        # Check identity: M == n_missing + n_ranking_at_k + n_selected_at_k
        assert M == n_missing + n_ranking_at_k[k] + n_selected_at_k[k], (
            f"Ranking identity violated for k={k}: M={M} != {n_missing} + {n_ranking_at_k[k]} + {n_selected_at_k[k]}"
        )

    # 1. Matching-consistent first hit rank for each GT (1-indexed)
    # Determined by prefix bipartite matching: which GT is newly matched when budget increases to k
    matched_first_hit_ranks = [-1] * M
    previously_matched_gts = set()
    for k in range(1, K + 1):
        sub_eval = evaluate_candidates_against_gt(sorted_ious[:k], iou_threshold)
        current_matched = set(sub_eval['matched_gt_indices'])
        newly_matched = current_matched - previously_matched_gts
        for g_idx in newly_matched:
            matched_first_hit_ranks[g_idx] = k
        previously_matched_gts = current_matched

    # 2. Raw unconstrained first hit rank for each GT (1-indexed)
    # min { rank(p_i) : IoU(p_i, g_j) >= tau } regardless of competition between GTs
    raw_first_hit_ranks = []
    hit_mask_sorted = (sorted_ious >= iou_threshold)
    for j in range(M):
        hit_indices = np.where(hit_mask_sorted[:, j])[0]
        if len(hit_indices) > 0:
            raw_first_hit_ranks.append(int(hit_indices[0]) + 1)
        else:
            raw_first_hit_ranks.append(-1)  # not found

    return {
        'H_all': h_all,
        'N_missing': n_missing,
        'H_at_k': h_at_k,
        'N_ranking_at_k': n_ranking_at_k,
        'N_selected_at_k': n_selected_at_k,
        'matched_first_hit_ranks': matched_first_hit_ranks,
        'raw_first_hit_ranks': raw_first_hit_ranks,
        'first_hit_ranks': matched_first_hit_ranks,  # default to matching-consistent
        'order': order.tolist()
    }


def evaluate_multigt_coverage_state(
    ious: np.ndarray,
    scores: Optional[np.ndarray] = None,
    budget: str = "all",
    iou_threshold: float = 0.4
) -> Dict[str, Any]:
    """Evaluates the coverage category of a multi-GT query (M >= 2).

    Categories:
      - full (找齐全部): m_matched == M
      - partial_multi (找到多个但未找齐): 1 < m_matched < M
      - single (只找到一个): m_matched == 1 (核心证据：多时刻候选分工不足)
      - zero (完全没找到): m_matched == 0

    Args:
        ious: (K, M) continuous IoU matrix
        scores: (K,) candidate scores (optional for 'all')
        budget: 'all', 'top10', 'top5', 'top1'
        iou_threshold: default 0.4
    """
    K, M = ious.shape
    if budget == "all" or scores is None:
        sub_ious = ious
    else:
        k = int(budget.replace("top", ""))
        order = np.argsort(-scores)[:min(k, K)]
        sub_ious = ious[order]

    eval_res = evaluate_candidates_against_gt(sub_ious, iou_threshold)
    m_matched = eval_res['independent_hits']

    if m_matched == M:
        category = "full"
    elif m_matched > 1:
        category = "partial_multi"
    elif m_matched == 1:
        category = "single"
    else:
        category = "zero"

    return {
        'M': M,
        'matched_hits': m_matched,
        'category': category
    }
