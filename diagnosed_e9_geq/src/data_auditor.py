"""Data Auditor for Experiment 1: Multi-GT Distribution & Overlap Audit.

Reads relevant_windows from annotation splits and reports:
- Negative queries (M=0)
- Single-GT queries (M=1)
- Dual-GT queries (M=2)
- Triple-or-more GT queries (M>=3)
- Total Multi-GT queries (M>=2)
- Total GT events inside Multi-GT queries
- Unique videos involved in Multi-GT queries
- Detailed pairwise GT overlap (tIoU) analysis
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from .core import compute_tiou


def audit_split(split_name: str, file_path: Path | str) -> Dict[str, Any]:
    """Audits a single dataset split JSONL file."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    total_queries = len(rows)
    all_vids = {r["vid"] for r in rows}

    m_counts: Dict[int, int] = {}
    multi_gt_vids = set()
    multi_gt_event_count = 0

    # Overlap metrics
    pairwise_ious = []
    query_max_overlap = []

    m_gte3 = 0
    m_gte2 = 0

    for r in rows:
        windows = r.get("relevant_windows", [])
        m = len(windows)
        m_counts[m] = m_counts.get(m, 0) + 1

        if m >= 2:
            m_gte2 += 1
            multi_gt_vids.add(r["vid"])
            multi_gt_event_count += m

            # Compute pairwise IoU
            max_iou_in_query = 0.0
            for i in range(len(windows)):
                for j in range(i + 1, len(windows)):
                    tiou = compute_tiou(windows[i], windows[j])
                    pairwise_ious.append(tiou)
                    if tiou > max_iou_in_query:
                        max_iou_in_query = tiou
            query_max_overlap.append(max_iou_in_query)

        if m >= 3:
            m_gte3 += 1

    pairwise_ious = np.array(pairwise_ious, dtype=np.float32) if pairwise_ious else np.zeros((0,), dtype=np.float32)

    # Overlap distribution bins
    overlap_bins = {
        "disjoint_0": int((pairwise_ious == 0.0).sum()),
        "low_0_to_0.1": int(((pairwise_ious > 0.0) & (pairwise_ious < 0.1)).sum()),
        "mid_0.1_to_0.3": int(((pairwise_ious >= 0.1) & (pairwise_ious < 0.3)).sum()),
        "high_0.3_to_0.5": int(((pairwise_ious >= 0.3) & (pairwise_ious < 0.5)).sum()),
        "severe_gte_0.5": int((pairwise_ious >= 0.5).sum()),
    }

    total_gt_events = sum(m * count for m, count in m_counts.items())

    stats = {
        "split": split_name,
        "file_path": str(path),
        "total_queries": total_queries,
        "total_unique_videos": len(all_vids),
        "total_gt_events": total_gt_events,
        "neg_queries_M0": m_counts.get(0, 0),
        "single_gt_queries_M1": m_counts.get(1, 0),
        "dual_gt_queries_M2": m_counts.get(2, 0),
        "triple_plus_gt_queries_Mgte3": m_gte3,
        "multi_gt_queries_Mgte2": m_gte2,
        "multi_gt_percentage": float(m_gte2 / total_queries * 100.0) if total_queries else 0.0,
        "multi_gt_events_total": multi_gt_event_count,
        "multi_gt_events_percentage": float(multi_gt_event_count / total_gt_events * 100.0) if total_gt_events else 0.0,
        "multi_gt_unique_videos": len(multi_gt_vids),
        "m_distribution": {str(k): v for k, v in sorted(m_counts.items())},
        "overlap_analysis": {
            "total_pairwise_comparisons": len(pairwise_ious),
            "mean_pairwise_iou": float(np.mean(pairwise_ious)) if len(pairwise_ious) else 0.0,
            "median_pairwise_iou": float(np.median(pairwise_ious)) if len(pairwise_ious) else 0.0,
            "max_pairwise_iou": float(np.max(pairwise_ious)) if len(pairwise_ious) else 0.0,
            "bins": overlap_bins,
            "disjoint_percentage": float(overlap_bins["disjoint_0"] / len(pairwise_ious) * 100.0) if len(pairwise_ious) else 0.0,
            "severe_overlap_percentage": float(overlap_bins["severe_gte_0.5"] / len(pairwise_ious) * 100.0) if len(pairwise_ious) else 0.0,
        }
    }
    return stats


def format_exp1_markdown_table(stats_list: List[Dict[str, Any]]) -> str:
    """Formats Experiment 1 audit results into publication-ready markdown tables."""
    lines = [
        "# 实验 1 诊断报告：Multi-GT 与时序标注重叠全景审计",
        "",
        "## 表 1.1：各数据子集 Multi-GT 统计清单",
        "",
        "| 数据集划分 (Split) | 总查询数 | 负样本 (M=0) | 单 GT (M=1) | 双 GT (M=2) | 三及以上 GT (M≥3) | 多 GT 总数 (M≥2) | 多 GT 查询占比 | 多 GT 真实事件总数 | 涉及多 GT 视频数 |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in stats_list:
        lines.append(
            f"| **{s['split']}** | {s['total_queries']} | {s['neg_queries_M0']} | "
            f"{s['single_gt_queries_M1']} | {s['dual_gt_queries_M2']} | "
            f"{s['triple_plus_gt_queries_Mgte3']} | **{s['multi_gt_queries_Mgte2']}** | "
            f"**{s['multi_gt_percentage']:.2f}%** | {s['multi_gt_events_total']} | "
            f"{s['multi_gt_unique_videos']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 表 1.2：多 GT 样本两两事件重叠度 (tIoU) 审计",
        "",
        "> **统计说明**：若多 GT 样本中不同事件的时间区间高度重叠 (如 $\\text{tIoU} \\ge 0.5$)，则该样本实际接近重复标注而非独立事件；若区间完全不相交 ($\\text{tIoU} = 0.0$) 或重叠极小，则充分证实其为时间上分离的多动作实例检索。",
        "",
        "| 数据集划分 (Split) | GT 区间对数 | 平均两两 tIoU | 中位数 tIoU | 最大 tIoU | 完全互斥对数 (tIoU=0) | 互斥占比 (%) | 严重重叠对数 (tIoU≥0.5) | 严重重叠占比 (%) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for s in stats_list:
        ov = s["overlap_analysis"]
        total_p = ov["total_pairwise_comparisons"]
        dis_cnt = ov["bins"]["disjoint_0"]
        sev_cnt = ov["bins"]["severe_gte_0.5"]
        dis_pct = ov["disjoint_percentage"]
        sev_pct = ov["severe_overlap_percentage"]
        lines.append(
            f"| **{s['split']}** | {total_p} | {ov['mean_pairwise_iou']:.4f} | "
            f"{ov['median_pairwise_iou']:.4f} | {ov['max_pairwise_iou']:.4f} | "
            f"{dis_cnt} | **{dis_pct:.2f}%** | {sev_cnt} | {sev_pct:.2f}% |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 🔍 实验 1 诊断结论与数据支撑判断",
        "",
    ])

    return "\n".join(lines)
