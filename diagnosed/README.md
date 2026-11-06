# SoccerGMR 底层机理诊断与全量对比实验目录 (`diagnosed/`)

本目录严格依据 [`diagnosed/实验设计.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/实验设计.md) 与审稿实证标准，实现了对 SoccerGMR 框架中全部 8 个 DETR 骨干网络（Method 1 原生基线与 Method 2 DQ-CGPv3）的底层机理诊断与对比实证分析。

所有核心实验、补充实验、全量指标表格、自动化复现脚本及高清可视化图表均**集中统一归档于本目录下**。

---

## 📖 核心诊断实验索引 (五大科学问题)

- **实验 1：Multi-GT 数据分布审计**（[`run_exp1_multigt.py`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/run_exp1_multigt.py) $\to$ [`tables/exp1_multigt_summary.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/exp1_multigt_summary.md)）
- **实验 2：Decoder Slot 活跃度与索引塌缩**（Table D: [`tables/table_d_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/)）
- **实验 3：候选预算分解与候选拥挤度**（Table A: [`tables/table_a_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/)）
- **实验 4：生成缺陷 vs 排序缺陷加法分解**（Table B: [`tables/table_b_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/)）
- **实验 5：Decoder 逐层演化与退化归因**（Table C: [`tables/table_c_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/)）
- **综合主报告**：[`实验报告_诊断分析结论.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/实验报告_诊断分析结论.md)

---

## 🔬 深度补充实证实验索引

针对审稿关切的一致性、排名口径、多时刻分工及层间演化问题，本目录提供了完整的实证闭环：

1. **检查点与评测指标一致性核验**：
   - 16 个权重物理 SHA-256 哈希值与清单 100% 逐字吻合；
   - 诊断提取最终层预测与官方推理预测窗口误差为 0；
   - 详见 [`tables/checkpoint_and_metric_consistency.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/checkpoint_and_metric_consistency.md)。

2. **首次命中排名口径修复（前缀二分图增量匹配）**：
   - 修复单候选无约束口径在多 GT 下导致的 Top-1 虚高与未生成数偏离；
   - 实现 $\sum \mathbb{I}[r \le k] \equiv H^{@k}$ 的严格数学闭合。

3. **多时刻查询 ($M \ge 2$) 完整事件覆盖度与候选分工实证**：
   - 建立 4 态互斥分类（找齐全部 $m=M$ / 找到多个 $1<m<M$ / **只找到一个 $m=1$** / 完全没找到 $m=0$）；
   - 揭示原生基线超 35%~45% 查询退化为“只找到一个”的单峰吸附困境；
   - 证明 DQ-CGPv3 使 UVCOM Top-5 找齐率暴增 7.67 倍（6 条 $\to$ 46 条）；
   - 详见 [`tables/table_multigt_coverage_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/) 与 [`tables/comparison_multigt_coverage_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/)。

4. **IoU 判定阈值敏感性检查 ($\tau \in \{0.3, 0.4, 0.5\}$)**：
   - 全量测算 8 大骨干在不同 IoU 门槛下的覆盖率与排序损失；
   - 证明“生成缺陷型”与“排序缺陷型”的机制界定在所有阈值下完全恒定不变；
   - 详见 [`tables/table_iou_sensitivity_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/)。

5. **加入 DQ-CGP 后的逐层命中变化全量对比 (Standard & Full 双测试集)**：
   - **专项学术深度报告**：👉 [`DQ_CGP逐层命中演化与分化分析报告_Standard_and_Full.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/DQ_CGP逐层命中演化与分化分析报告_Standard_and_Full.md)
   - **详尽对比数据表**：
     - Standard 测试集逐层演化表：[`tables/comparison_layer_evolution_standard.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/comparison_layer_evolution_standard.md)
     - Full-Decodable 测试集逐层演化表：[`tables/comparison_layer_evolution_full.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/comparison_layer_evolution_full.md)
   - **出版级高清曲线图**：
     - Standard 逐层演化对比图：[`figures/fig_layer_evolution_standard.png`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/figures/fig_layer_evolution_standard.png)
     - Full 逐层演化对比图：[`figures/fig_layer_evolution_full.png`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/figures/fig_layer_evolution_full.png)
   - **三大核心机制发现**：
     - **彻底根治层间退化**：Moment-DETR 原生 Decoder 在 D1 $\to$ D2 严重退化（丢弃 139 / 387 个已定位真实事件，净演化 -61 / -157）；DQ-CGPv3 彻底扭转为正向提精（净增 +33 / +71，保持数飙升 +120 / +376，最终独立命中增加 **+150 / +461**）。
     - **初始层 (D1) 高位先验赋能**：在第一层即确立压倒性优势（UVCOM D1 +126/+391, BM-DETR D1 +56/+178, Moment-DETR D1 +68/+258）。
     - **深层 Refinement 超强保持**：丢失数骤降 37%~70%，两层稳定保持数实现爆发式攀升（UVCOM Full 上保持数高达 1,807，比基线多锁定 +530 个真实事件）。

---

## 🚀 一键自动化复现指令

在仓库根目录下运行：
```bash
python3 diagnosed/run_supplementary_experiments.py
```
数秒内基于 `diagnosed/outputs/raw_predictions/` 下的无损全层预测数据，全自动重新生成本目录下的所有表格、结构化 JSON 及可视化曲线图。

---

## 📂 完整目录结构索引

```
diagnosed/
├── README.md                                                    # 本说明文件
├── 实验设计.md                                                   # 原始诊断实验设计白皮书
├── 实验报告_诊断分析结论.md                                        # 主诊断实验综合学术报告
├── 补充实验报告_一致性校验_多时刻分工与敏感性分析.md                # 补充实验综合报告
├── DQ_CGP逐层命中演化与分化分析报告_Standard_and_Full.md           # 逐层演化专项深度学术报告
├── run_all_experiments.py                                       # 核心 5 大实验运行入口
├── run_supplementary_experiments.py                             # 补充实验与对比图表全量生成脚本
├── figures/                                                     # 可视化成果图表
│   ├── fig_layer_evolution_standard.png                         # Standard 逐层演化曲线图 (Baseline vs DQ-CGPv3)
│   ├── fig_layer_evolution_full.png                             # Full 逐层演化曲线图 (Baseline vs DQ-CGPv3)
│   ├── fig2_missing_vs_ranking.png                              # 生成缺陷 vs 排序缺陷对比柱状图
│   └── fig5_m1_vs_m2_coverage.png                               # 基线 vs DQ-CGPv3 覆盖度对比柱状图
├── tables/                                                      # 全量 Markdown 实验表格库
│   ├── exp1_multigt_summary.md                                  # 实验 1: Multi-GT 数据集统计
│   ├── table_a_*.md                                             # 实验 3: 候选预算分解表 (M1/M2, Standard/Full)
│   ├── table_b_*.md                                             # 实验 4: 生成 vs 排序缺陷表 (M1/M2, Standard/Full)
│   ├── table_c_*.md                                             # 实验 5: 单模型 Decoder 逐层演化表
│   ├── table_d_*.md                                             # 实验 2: Query Slot 索引塌缩表
│   ├── comparison_m1_vs_m2.md                                   # 宏观最终层指标对比表
│   ├── checkpoint_and_metric_consistency.md                     # 检查点与指标一致性核验全景报告
│   ├── table_multigt_coverage_*.md                              # 多时刻 4 态覆盖表
│   ├── comparison_multigt_coverage_*.md                         # 多时刻分工改善对比表
│   ├── table_iou_sensitivity_*.md                               # IoU 敏感性全量表 (0.3/0.4/0.5)
│   ├── comparison_layer_evolution_standard.md                   # 实验 5 逐层演化对比表 (Standard)
│   └── comparison_layer_evolution_full.md                       # 实验 5 逐层演化对比表 (Full)
└── outputs/                                                     # 原始无损数据与结构化 JSON
    ├── raw_predictions/                                         # 32 个配置的逐样本无损全层预测 jsonl
    ├── analysis/                                                # 单模型逐层与 slot 分析 json
    ├── exp1_multigt/                                            # 数据集分布统计 json
    ├── checkpoint_consistency.json                              # 检查点物理哈希与 mAP 核验数据
    ├── multigt_coverage_summary.json                            # 多时刻 4 态覆盖分布数据
    ├── iou_sensitivity_summary.json                             # IoU 敏感性测试数据
    └── layer_evolution_summary.json                             # 逐层演化转移对比数据
```
