"""Presence & Existence Misjudgment Analyzer.

Evaluates how effectively each model discriminates target presence vs null events (M=0):
- P(exist) probability distributions for positive vs negative queries
- Presence contrast gap: Delta P = E[P | M>0] - E[P | M=0]
- False Positive Rate (FPR) on null queries across tau in [0.1, 0.9]
- True Positive Rate (TPR) / retention on positive queries across tau
- Rejection F1 score across tau
- AUROC for presence discrimination
- G-mIoU@1 across tau, Oracle envelope, and Oracle-gap (%) integral
"""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np


def compute_presence_metrics(records: List[Dict[str, Any]], num_tau_bins: int = 101) -> Dict[str, Any]:
    """Computes comprehensive target presence and null-event rejection diagnostics."""
    pos_records = [r for r in records if r["num_gt"] > 0]
    neg_records = [r for r in records if r["num_gt"] == 0]

    n_pos = len(pos_records)
    n_neg = len(neg_records)
    n_total = len(records)

    # Extract existence scores (fallback to max native score if pred_exist_score is None)
    pos_scores = []
    for r in pos_records:
        if r.get("pred_exist_score") is not None:
            pos_scores.append(float(r["pred_exist_score"]))
        else:
            final_scores = r["layers"][-1]["native_scores"]
            pos_scores.append(float(max(final_scores)) if final_scores else 0.0)

    neg_scores = []
    for r in neg_records:
        if r.get("pred_exist_score") is not None:
            neg_scores.append(float(r["pred_exist_score"]))
        else:
            final_scores = r["layers"][-1]["native_scores"]
            neg_scores.append(float(max(final_scores)) if final_scores else 0.0)

    pos_scores = np.array(pos_scores, dtype=np.float32)
    neg_scores = np.array(neg_scores, dtype=np.float32)

    # Separation statistics
    mean_pos_score = float(np.mean(pos_scores)) if n_pos > 0 else 0.0
    median_pos_score = float(np.median(pos_scores)) if n_pos > 0 else 0.0
    mean_neg_score = float(np.mean(neg_scores)) if n_neg > 0 else 0.0
    median_neg_score = float(np.median(neg_scores)) if n_neg > 0 else 0.0
    separation_gap = mean_pos_score - mean_neg_score

    # AUROC calculation (rank-based)
    if n_pos > 0 and n_neg > 0:
        y_true = np.concatenate([np.ones(n_pos, dtype=np.int32), np.zeros(n_neg, dtype=np.int32)])
        y_pred = np.concatenate([pos_scores, neg_scores])
        order = np.argsort(y_pred)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        pos_rank_sum = np.sum(ranks[:n_pos])
        # Mann-Whitney U
        u_stat = pos_rank_sum - n_pos * (n_pos - 1) / 2.0
        auroc = float(u_stat / (n_pos * n_neg) * 100.0)
    else:
        auroc = 50.0

    # Threshold sweeps: tau in [0.0, 1.0]
    tau_vals = np.linspace(0.0, 1.0, num_tau_bins)
    curves_data = []

    # Cache sample-level best IoU@1 for positive queries and 0 for negative queries
    sample_g1 = []
    for r in records:
        if r["num_gt"] == 0:
            sample_g1.append(0.0)
        else:
            final_layer = r["layers"][-1]
            scores = np.array(final_layer["native_scores"], dtype=np.float32)
            ious = np.array(final_layer["ious"], dtype=np.float32)
            if len(scores) > 0 and ious.shape[1] > 0:
                top1_idx = int(np.argmax(scores))
                # Top-1 candidate IoU with its closest GT
                best_iou = float(np.max(ious[top1_idx])) if ious.shape[1] > 0 else 0.0
                sample_g1.append(best_iou)
            else:
                sample_g1.append(0.0)
    sample_g1 = np.array(sample_g1, dtype=np.float32)

    all_scores = np.concatenate([pos_scores, neg_scores]) if (n_pos > 0 and n_neg > 0) else np.zeros(n_total, dtype=np.float32)
    labels = np.concatenate([np.ones(n_pos, dtype=np.int32), np.zeros(n_neg, dtype=np.int32)]) if (n_pos > 0 and n_neg > 0) else np.zeros(n_total, dtype=np.int32)

    g1_auc = 0.0
    oracle_g1_auc = 0.0
    prev_tau = 0.0

    tau_metrics = {}

    for i, tau in enumerate(tau_vals):
        # Predicted existence binary decisions: score >= tau
        # Positive prediction
        pred_pos = all_scores >= tau
        tp = int(np.sum((pred_pos == 1) & (labels == 1)))
        fp = int(np.sum((pred_pos == 1) & (labels == 0)))
        tn = int(np.sum((pred_pos == 0) & (labels == 0)))
        fn = int(np.sum((pred_pos == 0) & (labels == 1)))

        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr = fp / n_neg if n_neg > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        acc = (tp + tn) / n_total if n_total > 0 else 0.0
        f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) > 0 else 0.0

        # GMR Rejection F1 (rejection is the positive class of rejecting null queries)
        # Rejection True Positives = TN (correctly rejected nulls)
        # Rejection False Positives = FN (falsely rejected real events)
        # Rejection False Negatives = FP (failed to reject null queries)
        rej_tp = tn
        rej_fp = fn
        rej_fn = fp
        rej_prec = rej_tp / (rej_tp + rej_fp) if (rej_tp + rej_fp) > 0 else 0.0
        rej_rec = rej_tp / (rej_tp + rej_fn) if (rej_tp + rej_fn) > 0 else 0.0
        rej_f1 = 2 * rej_prec * rej_rec / (rej_prec + rej_rec) if (rej_prec + rej_rec) > 0 else 0.0

        # G-mIoU@1:
        # If score >= tau: if positive sample, achieves sample_g1; if negative sample, penalized with 0.
        # If score < tau: prediction rejected (0 IoU for positive, 1.0 IoU for correctly rejected negative)
        sample_g_iou = np.zeros(n_total, dtype=np.float32)
        # True positive predictions
        tp_mask = (pred_pos == 1) & (labels == 1)
        sample_g_iou[tp_mask] = sample_g1[tp_mask]
        # True negative predictions (correct rejection)
        tn_mask = (pred_pos == 0) & (labels == 0)
        sample_g_iou[tn_mask] = 1.0
        # False positives (fp_mask) remain 0.0
        # False negatives (fn_mask) remain 0.0

        g_miou = float(np.mean(sample_g_iou) * 100.0)

        # Oracle envelope for this tau:
        # An oracle with perfect existence knowledge achieves 1.0 on negatives and true top-1 IoU on positives
        # At tau=0, oracle g-miou is (sum(sample_g1_pos) + 0) / N
        # Perfect existence gate would give: (sum(sample_g1_pos) + N_neg * 1.0) / N
        oracle_miou = float((np.sum(sample_g1[labels == 1]) + n_neg * 1.0) / n_total * 100.0)

        curves_data.append({
            "tau": round(float(tau), 4),
            "tpr": round(float(tpr * 100.0), 2),
            "fpr": round(float(fpr * 100.0), 2),
            "accuracy": round(float(acc * 100.0), 2),
            "rej_f1": round(float(rej_f1 * 100.0), 2),
            "g_miou": round(g_miou, 2),
            "oracle_miou": round(oracle_miou, 2),
        })

        if i > 0:
            d_tau = tau - prev_tau
            g1_auc += g_miou * d_tau
            oracle_g1_auc += oracle_miou * d_tau
        prev_tau = tau

        if abs(tau - 0.4) < 1e-4:
            tau_metrics["0.4"] = {"acc": acc * 100.0, "rej_f1": rej_f1 * 100.0, "fpr": fpr * 100.0, "g_miou": g_miou}
        elif abs(tau - 0.5) < 1e-4:
            tau_metrics["0.5"] = {"acc": acc * 100.0, "rej_f1": rej_f1 * 100.0, "fpr": fpr * 100.0, "g_miou": g_miou}
        elif abs(tau - 0.6) < 1e-4:
            tau_metrics["0.6"] = {"acc": acc * 100.0, "rej_f1": rej_f1 * 100.0, "fpr": fpr * 100.0, "g_miou": g_miou}

    oracle_gap = oracle_g1_auc - g1_auc

    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "mean_pos_score": round(mean_pos_score, 4),
        "median_pos_score": round(median_pos_score, 4),
        "mean_neg_score": round(mean_neg_score, 4),
        "median_neg_score": round(median_neg_score, 4),
        "separation_gap": round(separation_gap, 4),
        "auroc": round(auroc, 2),
        "oracle_gap_pct": round(oracle_gap, 2),
        "g1_auc_pct": round(g1_auc, 2),
        "oracle_g1_pct": round(oracle_g1_auc, 2),
        "metrics_at_tau": tau_metrics,
        "curves_data": curves_data,
    }
