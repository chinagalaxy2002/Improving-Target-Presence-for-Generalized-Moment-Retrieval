"""Markdown Table and Diagnostic Report Generator.

Formats:
- Table A: Candidate Usage Across Different GT Counts (表 A：不同 GT 数量下的候选使用情况)
- Table B: Candidate Generation vs Native Score Ranking (表 B：候选是否生成及其排序结果)
- Table C: Layer-by-Layer Decoder Evolution (表 C：Decoder 逐层演化与退化全景表)
- Table D: Slot Index Collapse Audit (表 D：Decoder Slot 索引塌缩与利用均衡度审计)
- Master Mechanism Synthesis Report (综合机制诊断与因果论证报告)
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def generate_table_a(analysis_by_bb: Dict[str, Dict[str, Any]], split_name: str) -> str:
    """Generates Table A: Candidate Usage Across Different GT Counts."""
    lines = [
        f"# 表 A：不同 GT 数量下的候选使用情况 ({split_name})",
        "",
        "> **统计指标说明**：",
        "> - **查询数 (Queries)**：正样本查询总数 ($M \\ge 1$) 及各 GT 分组数。",
        "> - **GT 总数**：该分组内包含的全部真实动作时刻总数。",
        "> - **原生 Query 总数**：模型原生解码器候选槽位总预算 ($N_{queries} \\times K$)。",
        "> - **Active Query 数 (θ=0.5)**：原生置信度得分 $s_i \\ge 0.5$ 的候选总数。",
        "> - **独立命中 GT 数 ($H_n$)**：通过严格二分图最大匹配确定的、不同候选一一对应不同 GT 的有效命中数 (IoU $\\ge 0.4$)。",
        "> - **重复候选数 ($D_n$)**：命中了真实事件但在独立匹配中多出的冗余候选数 ($D_n = T_n - H_n$)。",
        "> - **未命中候选数 ($U_n$)**：完全未与任何真实事件重叠达标的候选数 ($U_n = K - T_n$)。",
        "> - 恒等式严格成立：$\\text{原生 Query 总数} = H_n + D_n + U_n$。",
        "",
        "| 骨干模型 (Backbone) | GT 分组 | 查询数 | GT 总数 | 原生 Query 总数 | Active 数 (θ=0.5) | 独立命中 GT ($H_n$) | 重复候选 ($D_n$) | 未命中候选 ($U_n$) | GT 独立覆盖率 (%) | 候选冗余率 (%) |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for bb, res in sorted(analysis_by_bb.items()):
        exp2 = res["exp2_query_activity"]
        exp3 = res["exp3_duplication"]
        K = res["K"]

        group_keys = [("M=1", "M1"), ("M=2", "M2"), ("M≥3", "Mgte3"), ("全部正样本 (M≥1)", "all_pos")]
        for g_label, g_key in group_keys:
            if g_key not in exp3 or exp3[g_key].get("query_count", 0) == 0:
                continue
            e2 = exp2[g_key]
            e3 = exp3[g_key]
            q_cnt = e3["query_count"]
            gt_cnt = e3["total_gt"]
            slots_cnt = e3["total_slots"]
            act_cnt = e2["active_counts_sum"]["0.5"]
            ind_hit = e3["independent_hits_sum"]
            rep_cnt = e3["redundant_candidates_sum"]
            unhit_cnt = e3["unhit_candidates_sum"]

            cov_pct = (ind_hit / gt_cnt * 100.0) if gt_cnt > 0 else 0.0
            rep_pct = (rep_cnt / (ind_hit + rep_cnt) * 100.0) if (ind_hit + rep_cnt) > 0 else 0.0

            bold_prefix = "**" if g_key == "all_pos" else ""
            bold_suffix = "**" if g_key == "all_pos" else ""

            lines.append(
                f"| {bold_prefix}{bb}{bold_suffix} | {bold_prefix}{g_label}{bold_suffix} | "
                f"{q_cnt} | {gt_cnt} | {slots_cnt} | {act_cnt} | "
                f"{bold_prefix}{ind_hit}{bold_suffix} | {rep_cnt} | {unhit_cnt} | "
                f"{cov_pct:.1f}% | {rep_pct:.1f}% |"
            )

    return "\n".join(lines)


def generate_table_b(analysis_by_bb: Dict[str, Dict[str, Any]], split_name: str) -> str:
    """Generates Table B: Candidate Generation vs Native Score Ranking."""
    lines = [
        f"# 表 B：候选是否生成及其排序结果 ({split_name})",
        "",
        "> **统计指标说明**：",
        "> - **All 独立命中 ($H^{all}$)**：完全忽略置信度排序，直接使用全部原生候选 ($K$) 进行最大匹配能够独立找回的最大 GT 数。",
        "> - **Top-k 命中 ($H^{@k}$)**：仅使用模型原生排序得分最高的前 $k$ 个候选进行独立匹配找回的 GT 数。",
        "> - **All 未生成事件 ($N^{missing} = M - H^{all}$)**：候选池中根本未曾生成合规预测窗口的事件数（**候选集合生成缺陷**）。",
        "> - **Top-5 排序遗漏 ($N^{ranking@5} = H^{all} - H^{@5}$)**：候选池中已成功定位，但被模型原生得分排在 5 名开外导致遗漏的事件数（**候选排序缺陷**）。",
        "> - 严格可加性恒等式：$M = N^{missing} + N^{ranking@k} + N^{selected@k}$。",
        "",
        "| 骨干模型 (Backbone) | GT 分组 | GT 总数 | All 独立命中 | Top-1 命中 | Top-3 命中 | Top-5 命中 | Top-10 命中 | All 未生成 ($N^{missing}$) | Top-5 排序遗漏 | 未生成率 (%) | Top-5 排序损失率 (%) |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for bb, res in sorted(analysis_by_bb.items()):
        exp4 = res["exp4_generation_vs_ranking"]

        group_keys = [("M=1", "M1"), ("M=2", "M2"), ("M≥3", "Mgte3"), ("全部正样本 (M≥1)", "all_pos")]
        for g_label, g_key in group_keys:
            if g_key not in exp4 or exp4[g_key].get("query_count", 0) == 0:
                continue
            e4 = exp4[g_key]
            gt_cnt = e4["total_gt"]
            h_all = e4["H_all_sum"]
            h1 = e4["H_at_k_sum"]["1"]
            h3 = e4["H_at_k_sum"]["3"]
            h5 = e4["H_at_k_sum"]["5"]
            h10 = e4["H_at_k_sum"]["10"]
            n_miss = e4["N_missing_sum"]
            n_rank5 = e4["N_ranking_at_k_sum"]["5"]

            miss_pct = (n_miss / gt_cnt * 100.0) if gt_cnt > 0 else 0.0
            rank5_pct = (n_rank5 / gt_cnt * 100.0) if gt_cnt > 0 else 0.0

            bold_prefix = "**" if g_key == "all_pos" else ""
            bold_suffix = "**" if g_key == "all_pos" else ""

            lines.append(
                f"| {bold_prefix}{bb}{bold_suffix} | {bold_prefix}{g_label}{bold_suffix} | "
                f"{gt_cnt} | {bold_prefix}{h_all}{bold_suffix} | "
                f"{h1} | {h3} | {h5} | {h10} | "
                f"**{n_miss}** | {n_rank5} | "
                f"{miss_pct:.1f}% | {rank5_pct:.1f}% |"
            )

    return "\n".join(lines)


def generate_table_c(analysis_by_bb: Dict[str, Dict[str, Any]], split_name: str) -> str:
    """Generates Table C: Layer-by-Layer Decoder Evolution."""
    lines = [
        f"# 表 C：Decoder 逐层演化与分化退化全景表 ({split_name})",
        "",
        "> **统计指标说明**：",
        "> - **Decoder 层**：D1 为第 1 层解码器输出，直至最终层。",
        "> - **独立命中 GT 数 ($H^{(l)}$)**：该层全部候选通过二分图最大匹配独立找回的 GT 数。",
        "> - **覆盖不同 GT 数 ($|S^{(l)}|$)**：该层至少被一个候选命中的不同 GT 数量集合大小。",
        "> - **重复候选数 ($D^{(l)}$)**：该层候选集中命中了 GT 但多余的冗余数量。",
        "> - **未命中候选数 ($U^{(l)}$)**：该层未命中任何 GT 的候选数量。",
        "> - **相邻层演化 ($l \\to l+1$)**：",
        ">   - **新增 (+Gain)**：$|S^{(l+1)} \\setminus S^{(l)}|$，即前一层未覆盖但本层新定位成功的 GT 数。",
        ">   - **丢失 (-Loss)**：$|S^{(l)} \\setminus S^{(l+1)}|$，即前一层已覆盖但本层反而在 Refinement 中丢失定位的 GT 数（**解码器退化核心标志**）。",
        ">   - **持续 (Keep)**：$|S^{(l)} \\cap S^{(l+1)}|$，两层连续保持命中的稳定 GT 数。",
        "",
        "| 骨干模型 (Backbone) | Decoder 层 | 独立命中 GT ($H^{(l)}$) | 覆盖不同 GT ($|S^{(l)}|$) | 重复候选 ($D^{(l)}$) | 未命中候选 ($U^{(l)}$) | 相邻演化阶段 | 新增 (+Gain) | 丢失 (-Loss) | 净增减 (Net) | 持续保持 (Keep) |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for bb, res in sorted(analysis_by_bb.items()):
        layers = res["exp5_layers"]
        transitions = res["exp5_transitions"]

        for l_idx, layer in enumerate(layers):
            l_name = layer["layer_name"]
            h = layer["independent_hits_sum"]
            distinct = layer["distinct_gts_hit_sum"]
            d = layer["redundant_candidates_sum"]
            u = layer["unhit_candidates_sum"]

            if l_idx == 0:
                t_str = "-"
                gain_str = "-"
                loss_str = "-"
                net_str = "-"
                keep_str = "-"
            else:
                t = transitions[l_idx - 1]
                t_str = t["transition"]
                gain_str = f"+{t['gain_sum']}"
                loss_str = f"-{t['loss_sum']}"
                net = t["net_gain"]
                net_str = f"+{net}" if net > 0 else f"{net}"
                keep_str = f"{t['keep_sum']}"

            lines.append(
                f"| **{bb}** | **{l_name}** | {h} | {distinct} | {d} | {u} | "
                f"{t_str} | {gain_str} | **{loss_str}** | {net_str} | {keep_str} |"
            )

    return "\n".join(lines)


def generate_table_d(analysis_by_bb: Dict[str, Dict[str, Any]], split_name: str) -> str:
    """Generates Table D: Decoder Slot Index Collapse Audit."""
    lines = [
        f"# 表 D：Decoder Slot 索引塌缩与利用均衡度审计 ({split_name})",
        "",
        "> **统计指标说明**：",
        "> - **K 槽位**：该骨干模型原生 Decoder Query 数量。",
        "> - **Top-1 占优 Slot**：成为 Top-1 预测次数最多的 Query Slot 编号及样本占比。",
        "> - **独立命中占优 Slot**：在二分图独立匹配中命中 GT 次数最多的 Query Slot 编号及样本占比。",
        "> - **Top-2 集中度 (Top-2 Concentration)**：前 2 个最活跃 Slot 贡献的 Top-1 或 Hit 比例之和。",
        "> - **索引塌缩判断 (Index Collapse Verdict)**：如果少数 1-2 个 Slot 垄断了 70% 以上的有效预测，则判定存在严重索引塌缩。",
        "",
        "| 骨干模型 (Backbone) | K 槽位 | 最活跃 Top-1 Slot | Top-1 占比 (%) | Top-2 Slot Top-1 集中度 (%) | 最多命中 Slot | 命中占比 (%) | Top-2 Slot 命中集中度 (%) | 索引塌缩判定 |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for bb, res in sorted(analysis_by_bb.items()):
        slot_stats = res["exp2_slot_stats"]
        K = slot_stats["K"]
        pos_cnt = res["pos_queries"]
        if pos_cnt == 0:
            continue

        top1_counts = np.array(slot_stats["slot_top1_count"])
        hit_counts = np.array(slot_stats["slot_independent_hit_count"])

        top1_order = np.argsort(-top1_counts)
        hit_order = np.argsort(-hit_counts)

        best_top1_slot = int(top1_order[0])
        best_top1_pct = float(top1_counts[best_top1_slot] / pos_cnt * 100.0)
        top2_top1_conc = float((top1_counts[top1_order[0]] + top1_counts[top1_order[1]]) / pos_cnt * 100.0) if K >= 2 else best_top1_pct

        best_hit_slot = int(hit_order[0])
        total_hits = int(hit_counts.sum())
        best_hit_pct = float(hit_counts[best_hit_slot] / total_hits * 100.0) if total_hits > 0 else 0.0
        top2_hit_conc = float((hit_counts[hit_order[0]] + hit_counts[hit_order[1]]) / total_hits * 100.0) if total_hits > 0 and K >= 2 else best_hit_pct

        if top2_top1_conc >= 70.0 or top2_hit_conc >= 60.0:
            verdict = "⚠️ 严重索引塌缩 (Severe Collapse)"
        elif top2_top1_conc >= 45.0 or top2_hit_conc >= 40.0:
            verdict = "⚡ 中度槽位偏置 (Moderate Bias)"
        else:
            verdict = "✅ 槽位使用较均匀 (Uniform)"

        lines.append(
            f"| **{bb}** | {K} | Slot Q{best_top1_slot} | {best_top1_pct:.1f}% | "
            f"**{top2_top1_conc:.1f}%** | Slot Q{best_hit_slot} | {best_hit_pct:.1f}% | "
            f"**{top2_hit_conc:.1f}%** | {verdict} |"
        )

    return "\n".join(lines)
