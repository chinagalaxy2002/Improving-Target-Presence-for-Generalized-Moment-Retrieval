"""Diagnostic Analyzers for Experiments 2, 3, 4, and 5.

Implements all equations and tables specified in 实验设计.md:
- Exp 2: Active query counts A_n(theta), effective queries E_n, slot collapse frequencies.
- Exp 3: Candidate composition K = H_n + D_n + U_n, per-GT crowding C_nj.
- Exp 4: Candidate generation vs ranking: M = N_missing + N_ranking@k + N_selected@k, first-hit ranks.
- Exp 5: Layer-by-layer evolution: H^(l), D^(l), U^(l), N_gain, N_loss, N_keep, slot transitions.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .core import evaluate_candidates_against_gt, evaluate_topk_ranking, evaluate_multigt_coverage_state


def analyze_records(records: List[Dict[str, Any]], iou_threshold: float = 0.4) -> Dict[str, Any]:
    """Runs all 4 diagnostic analyses (Exp 2, 3, 4, 5) on extracted inference records."""
    # Filter positive queries for primary event diagnostics
    pos_records = [r for r in records if r["num_gt"] > 0]
    neg_records = [r for r in records if r["num_gt"] == 0]

    # Group by M
    groups = {
        "all_pos": pos_records,
        "M1": [r for r in pos_records if r["num_gt"] == 1],
        "M2": [r for r in pos_records if r["num_gt"] == 2],
        "Mgte3": [r for r in pos_records if r["num_gt"] >= 3],
        "Mgte2": [r for r in pos_records if r["num_gt"] >= 2],
    }

    # Determine K and L
    sample = records[0]
    num_layers = sample["num_layers"]
    K = len(sample["layers"][-1]["native_scores"])

    # -------------------------------------------------------------
    # EXPERIMENT 2: Query Activity & Slot Utilization (Final Layer)
    # -------------------------------------------------------------
    thresholds = [0.1, 0.3, 0.5, 0.7]
    exp2_results = {}

    for g_name, g_recs in groups.items():
        n_queries = len(g_recs)
        total_gt = sum(r["num_gt"] for r in g_recs)
        if n_queries == 0:
            exp2_results[g_name] = {"count": 0, "total_gt": 0}
            continue

        act_counts = {th: [] for th in thresholds}
        effective_counts = []

        for r in g_recs:
            final_layer = r["layers"][-1]
            scores = np.array(final_layer["native_scores"], dtype=np.float32)
            ious = np.array(final_layer["ious"], dtype=np.float32)

            for th in thresholds:
                act_counts[th].append(int((scores >= th).sum()))

            full_eval = evaluate_candidates_against_gt(ious, iou_threshold)
            effective_counts.append(full_eval["independent_hits"])

        exp2_results[g_name] = {
            "query_count": n_queries,
            "total_gt": total_gt,
            "total_slots": n_queries * K,
            "active_counts_sum": {str(th): int(sum(act_counts[th])) for th in thresholds},
            "active_counts_mean": {str(th): float(np.mean(act_counts[th])) for th in thresholds},
            "effective_hits_sum": int(sum(effective_counts)),
            "effective_hits_mean": float(np.mean(effective_counts)),
        }

    # Slot Collapse Statistics (across all positive samples)
    slot_active_05 = np.zeros(K, dtype=np.int64)
    slot_top1 = np.zeros(K, dtype=np.int64)
    slot_hit = np.zeros(K, dtype=np.int64)

    for r in pos_records:
        final_layer = r["layers"][-1]
        scores = np.array(final_layer["native_scores"], dtype=np.float32)
        ious = np.array(final_layer["ious"], dtype=np.float32)

        # Active
        slot_active_05 += (scores >= 0.5).astype(np.int64)

        # Top 1
        top1_idx = int(np.argmax(scores))
        slot_top1[top1_idx] += 1

        # Hit in bipartite match
        full_eval = evaluate_candidates_against_gt(ious, iou_threshold)
        for q_idx in full_eval["matched_query_indices"]:
            slot_hit[q_idx] += 1

    exp2_slot_stats = {
        "K": K,
        "slot_active_count_th05": slot_active_05.tolist(),
        "slot_top1_count": slot_top1.tolist(),
        "slot_independent_hit_count": slot_hit.tolist(),
        "slot_active_pct_th05": (slot_active_05 / len(pos_records) * 100.0).tolist() if pos_records else [],
        "slot_top1_pct": (slot_top1 / len(pos_records) * 100.0).tolist() if pos_records else [],
        "slot_independent_hit_pct": (slot_hit / len(pos_records) * 100.0).tolist() if pos_records else [],
    }

    # -------------------------------------------------------------
    # EXPERIMENT 3: Query Duplication & Crowding (Final Layer)
    # -------------------------------------------------------------
    exp3_results = {}
    gt_hit_crowding = {"0_hits": 0, "1_hit": 0, "2_hits": 0, "3plus_hits": 0}

    for g_name, g_recs in groups.items():
        n_queries = len(g_recs)
        total_gt = sum(r["num_gt"] for r in g_recs)
        if n_queries == 0:
            exp3_results[g_name] = {"count": 0}
            continue

        raw_hits_list = []
        ind_hits_list = []
        redundant_list = []
        unhit_list = []

        for r in g_recs:
            final_layer = r["layers"][-1]
            ious = np.array(final_layer["ious"], dtype=np.float32)
            eval_res = evaluate_candidates_against_gt(ious, iou_threshold)

            raw_hits_list.append(eval_res["raw_hit_candidates"])
            ind_hits_list.append(eval_res["independent_hits"])
            redundant_list.append(eval_res["redundant_candidates"])
            unhit_list.append(eval_res["unhit_candidates"])

            if g_name == "all_pos":
                for c in eval_res["gt_hit_counts"]:
                    if c == 0:
                        gt_hit_crowding["0_hits"] += 1
                    elif c == 1:
                        gt_hit_crowding["1_hit"] += 1
                    elif c == 2:
                        gt_hit_crowding["2_hits"] += 1
                    else:
                        gt_hit_crowding["3plus_hits"] += 1

        exp3_results[g_name] = {
            "query_count": n_queries,
            "total_gt": total_gt,
            "total_slots": n_queries * K,
            "raw_hit_candidates_sum": int(sum(raw_hits_list)),
            "independent_hits_sum": int(sum(ind_hits_list)),
            "redundant_candidates_sum": int(sum(redundant_list)),
            "unhit_candidates_sum": int(sum(unhit_list)),
            "raw_hit_candidates_mean": float(np.mean(raw_hits_list)),
            "independent_hits_mean": float(np.mean(ind_hits_list)),
            "redundant_candidates_mean": float(np.mean(redundant_list)),
            "unhit_candidates_mean": float(np.mean(unhit_list)),
        }

    # -------------------------------------------------------------
    # EXPERIMENT 4: Candidate Generation vs. Candidate Ranking
    # -------------------------------------------------------------
    exp4_results = {}
    k_eval_list = (1, 3, 5, 10)
    
    # 1. Matching-consistent first hit rank (prefix bipartite matching)
    matched_first_hit_rank_hist = {f"rank_{k}": 0 for k in range(1, K + 1)}
    matched_first_hit_rank_hist["not_found"] = 0
    
    # 2. Raw unconstrained first hit rank (min rank where IoU >= tau)
    raw_first_hit_rank_hist = {f"rank_{k}": 0 for k in range(1, K + 1)}
    raw_first_hit_rank_hist["not_found"] = 0

    for g_name, g_recs in groups.items():
        n_queries = len(g_recs)
        total_gt = sum(r["num_gt"] for r in g_recs)
        if n_queries == 0:
            exp4_results[g_name] = {"count": 0}
            continue

        h_all_sum = 0
        n_missing_sum = 0
        h_at_k_sum = {k: 0 for k in k_eval_list}
        n_ranking_at_k_sum = {k: 0 for k in k_eval_list}
        n_selected_at_k_sum = {k: 0 for k in k_eval_list}

        for r in g_recs:
            final_layer = r["layers"][-1]
            scores = np.array(final_layer["native_scores"], dtype=np.float32)
            ious = np.array(final_layer["ious"], dtype=np.float32)

            rank_res = evaluate_topk_ranking(ious, scores, k_eval_list, iou_threshold)

            h_all_sum += rank_res["H_all"]
            n_missing_sum += rank_res["N_missing"]
            for k in k_eval_list:
                h_at_k_sum[k] += rank_res["H_at_k"][k]
                n_ranking_at_k_sum[k] += rank_res["N_ranking_at_k"][k]
                n_selected_at_k_sum[k] += rank_res["N_selected_at_k"][k]

            if g_name == "all_pos":
                for rnk in rank_res["matched_first_hit_ranks"]:
                    if rnk == -1:
                        matched_first_hit_rank_hist["not_found"] += 1
                    else:
                        matched_first_hit_rank_hist[f"rank_{rnk}"] += 1

                for rnk in rank_res["raw_first_hit_ranks"]:
                    if rnk == -1:
                        raw_first_hit_rank_hist["not_found"] += 1
                    else:
                        raw_first_hit_rank_hist[f"rank_{rnk}"] += 1

        exp4_results[g_name] = {
            "query_count": n_queries,
            "total_gt": total_gt,
            "H_all_sum": h_all_sum,
            "N_missing_sum": n_missing_sum,
            "missing_rate_pct": float(n_missing_sum / total_gt * 100.0) if total_gt else 0.0,
            "H_at_k_sum": {str(k): v for k, v in h_at_k_sum.items()},
            "N_ranking_at_k_sum": {str(k): v for k, v in n_ranking_at_k_sum.items()},
            "N_selected_at_k_sum": {str(k): v for k, v in n_selected_at_k_sum.items()},
            "ranking_omission_rate_top5_pct": float(n_ranking_at_k_sum[5] / total_gt * 100.0) if total_gt else 0.0,
            "ranking_omission_rate_top10_pct": float(n_ranking_at_k_sum[10] / total_gt * 100.0) if total_gt else 0.0,
        }

    # -------------------------------------------------------------
    # MULTI-GT QUERY COMPLETE EVENT COVERAGE ANALYSIS (M >= 2)
    # -------------------------------------------------------------
    multigt_recs = groups["Mgte2"]
    multigt_total_queries = len(multigt_recs)
    multigt_total_gt = sum(r["num_gt"] for r in multigt_recs)
    multigt_coverage = {}

    for budget in ["all", "top10", "top5", "top1"]:
        cat_counts = {"full": 0, "partial_multi": 0, "single": 0, "zero": 0}
        for r in multigt_recs:
            final_layer = r["layers"][-1]
            scores = np.array(final_layer["native_scores"], dtype=np.float32)
            ious = np.array(final_layer["ious"], dtype=np.float32)
            st = evaluate_multigt_coverage_state(ious, scores, budget=budget, iou_threshold=iou_threshold)
            cat_counts[st["category"]] += 1

        multigt_coverage[budget] = {
            "total_queries": multigt_total_queries,
            "total_gt": multigt_total_gt,
            "counts": cat_counts,
            "percentages": {
                c: float(v / multigt_total_queries * 100.0) if multigt_total_queries else 0.0
                for c, v in cat_counts.items()
            }
        }

    # -------------------------------------------------------------
    # IOU SENSITIVITY ANALYSIS (tau in {0.3, 0.4, 0.5})
    # -------------------------------------------------------------
    sens_thresholds = (0.3, 0.4, 0.5)
    iou_sensitivity = {}

    for tau in sens_thresholds:
        tau_all_pos_total_gt = sum(r["num_gt"] for r in pos_records)
        tau_h_all_sum = 0
        tau_h1_sum = 0
        tau_h3_sum = 0
        tau_h5_sum = 0
        tau_h10_sum = 0

        # Multi-GT coverage under tau
        tau_mgte2_counts = {b: {"full": 0, "partial_multi": 0, "single": 0, "zero": 0} for b in ["all", "top10", "top5"]}

        for r in pos_records:
            final_layer = r["layers"][-1]
            scores = np.array(final_layer["native_scores"], dtype=np.float32)
            ious = np.array(final_layer["ious"], dtype=np.float32)

            rank_res = evaluate_topk_ranking(ious, scores, (1, 3, 5, 10), tau)
            tau_h_all_sum += rank_res["H_all"]
            tau_h1_sum += rank_res["H_at_k"][1]
            tau_h3_sum += rank_res["H_at_k"][3]
            tau_h5_sum += rank_res["H_at_k"][5]
            tau_h10_sum += rank_res["H_at_k"][10]

            if r["num_gt"] >= 2:
                for b in ["all", "top10", "top5"]:
                    st = evaluate_multigt_coverage_state(ious, scores, budget=b, iou_threshold=tau)
                    tau_mgte2_counts[b][st["category"]] += 1

        tau_n_missing = tau_all_pos_total_gt - tau_h_all_sum
        tau_n_rank5 = tau_h_all_sum - tau_h5_sum

        iou_sensitivity[str(tau)] = {
            "iou_threshold": tau,
            "total_gt": tau_all_pos_total_gt,
            "H_all": tau_h_all_sum,
            "coverage_pct": float(tau_h_all_sum / tau_all_pos_total_gt * 100.0) if tau_all_pos_total_gt else 0.0,
            "N_missing": tau_n_missing,
            "missing_pct": float(tau_n_missing / tau_all_pos_total_gt * 100.0) if tau_all_pos_total_gt else 0.0,
            "H_at_1": tau_h1_sum,
            "H_at_3": tau_h3_sum,
            "H_at_5": tau_h5_sum,
            "H_at_10": tau_h10_sum,
            "N_ranking_at_5": tau_n_rank5,
            "ranking_loss_top5_pct": float(tau_n_rank5 / tau_all_pos_total_gt * 100.0) if tau_all_pos_total_gt else 0.0,
            "multigt_coverage": {
                b: {
                    "counts": tau_mgte2_counts[b],
                    "percentages": {
                        c: float(v / multigt_total_queries * 100.0) if multigt_total_queries else 0.0
                        for c, v in tau_mgte2_counts[b].items()
                    }
                }
                for b in ["all", "top10", "top5"]
            }
        }

    # -------------------------------------------------------------
    # EXPERIMENT 5: Layer-by-Layer Progression & Degradation
    # -------------------------------------------------------------
    exp5_layers = []

    # Track layer-wise hit counts
    for l_idx in range(num_layers):
        ind_hits_l = []
        distinct_gts_l = []
        redundant_l = []
        unhit_l = []

        for r in pos_records:
            layer = r["layers"][l_idx]
            ious = np.array(layer["ious"], dtype=np.float32)
            eval_res = evaluate_candidates_against_gt(ious, iou_threshold)

            ind_hits_l.append(eval_res["independent_hits"])
            distinct_gts_l.append(eval_res["distinct_gts_hit"])
            redundant_l.append(eval_res["redundant_candidates"])
            unhit_l.append(eval_res["unhit_candidates"])

        exp5_layers.append({
            "layer_idx": l_idx,
            "layer_name": f"D{l_idx + 1}",
            "independent_hits_sum": int(sum(ind_hits_l)),
            "distinct_gts_hit_sum": int(sum(distinct_gts_l)),
            "redundant_candidates_sum": int(sum(redundant_l)),
            "unhit_candidates_sum": int(sum(unhit_l)),
            "independent_hits_mean": float(np.mean(ind_hits_l)) if ind_hits_l else 0.0,
            "redundant_candidates_mean": float(np.mean(redundant_l)) if redundant_l else 0.0,
            "unhit_candidates_mean": float(np.mean(unhit_l)) if unhit_l else 0.0,
        })

    # Track inter-layer gain / loss / keep (l -> l+1)
    exp5_transitions = []
    for l_idx in range(num_layers - 1):
        gains = []
        losses = []
        keeps = []

        for r in pos_records:
            layer_curr = r["layers"][l_idx]
            layer_next = r["layers"][l_idx + 1]

            ious_curr = np.array(layer_curr["ious"], dtype=np.float32)
            ious_next = np.array(layer_next["ious"], dtype=np.float32)

            # Hit GT indices
            hits_curr = set(np.where((ious_curr >= iou_threshold).any(axis=0))[0])
            hits_next = set(np.where((ious_next >= iou_threshold).any(axis=0))[0])

            gain = len(hits_next - hits_curr)
            loss = len(hits_curr - hits_next)
            keep = len(hits_curr & hits_next)

            gains.append(gain)
            losses.append(loss)
            keeps.append(keep)

        exp5_transitions.append({
            "transition": f"D{l_idx + 1} -> D{l_idx + 2}",
            "from_layer": l_idx,
            "to_layer": l_idx + 1,
            "gain_sum": int(sum(gains)),
            "loss_sum": int(sum(losses)),
            "keep_sum": int(sum(keeps)),
            "net_gain": int(sum(gains) - sum(losses)),
            "gain_mean": float(np.mean(gains)) if gains else 0.0,
            "loss_mean": float(np.mean(losses)) if losses else 0.0,
            "keep_mean": float(np.mean(keeps)) if keeps else 0.0,
        })

    # Slot transitions across adjacent layers (tracked on positive samples)
    slot_transitions = []
    for l_idx in range(num_layers - 1):
        # unhit -> hit, hit -> unhit, hit -> hit (same GT), hit -> hit (different GT)
        u_to_h = np.zeros(K, dtype=np.int64)
        h_to_u = np.zeros(K, dtype=np.int64)
        h_to_same = np.zeros(K, dtype=np.int64)
        h_to_diff = np.zeros(K, dtype=np.int64)

        for r in pos_records:
            ious_curr = np.array(r["layers"][l_idx]["ious"], dtype=np.float32)
            ious_next = np.array(r["layers"][l_idx + 1]["ious"], dtype=np.float32)

            res_curr = evaluate_candidates_against_gt(ious_curr, iou_threshold)
            res_next = evaluate_candidates_against_gt(ious_next, iou_threshold)

            map_curr = res_curr["query_to_gt_map"]
            map_next = res_next["query_to_gt_map"]

            for q in range(K):
                in_c = q in map_curr
                in_n = q in map_next

                if not in_c and in_n:
                    u_to_h[q] += 1
                elif in_c and not in_n:
                    h_to_u[q] += 1
                elif in_c and in_n:
                    if map_curr[q] == map_next[q]:
                        h_to_same[q] += 1
                    else:
                        h_to_diff[q] += 1

        slot_transitions.append({
            "transition": f"D{l_idx + 1} -> D{l_idx + 2}",
            "unhit_to_hit": u_to_h.tolist(),
            "hit_to_unhit": h_to_u.tolist(),
            "hit_same_gt": h_to_same.tolist(),
            "hit_diff_gt": h_to_diff.tolist(),
        })

    return {
        "K": K,
        "num_layers": num_layers,
        "total_queries": len(records),
        "pos_queries": len(pos_records),
        "neg_queries": len(neg_records),
        "exp2_query_activity": exp2_results,
        "exp2_slot_stats": exp2_slot_stats,
        "exp3_duplication": exp3_results,
        "exp3_gt_crowding": gt_hit_crowding,
        "exp4_generation_vs_ranking": exp4_results,
        "exp4_matched_first_hit_rank_hist": matched_first_hit_rank_hist,
        "exp4_raw_first_hit_rank_hist": raw_first_hit_rank_hist,
        "exp4_first_hit_rank_hist": matched_first_hit_rank_hist,
        "exp4_multigt_coverage": multigt_coverage,
        "iou_sensitivity": iou_sensitivity,
        "exp5_layers": exp5_layers,
        "exp5_transitions": exp5_transitions,
        "exp5_slot_transitions": slot_transitions,
    }
