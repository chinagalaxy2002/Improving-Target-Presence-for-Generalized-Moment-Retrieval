# SoccerGMR 补充实验独立归档目录 (`diagnosed_supplementary/`)

本文件夹为 SoccerGMR 时序检索机制诊断补充实验的**专属独立归档目录**，涵盖四项深度实证成果：

1. **检查点与评测指标一致性核验**：
   - 8 个 DETR 骨干网络全部 16 个检查点的物理文件路径、大小及 SHA-256 哈希值与 [`docs/checkpoints_and_paths.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/docs/checkpoints_and_paths.md) 的 100% 逐字吻合核对。
   - 诊断提取最终层预测与官方推理预测窗口/得分（误差为 0）以及官方评估脚本重算指标（mAP 差值 $\le 0.05$）的复现比对。
   - 详细技术报告见 [`tables/checkpoint_and_metric_consistency.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed_supplementary/tables/checkpoint_and_metric_consistency.md)。

2. **首次命中排名口径修复与双口径对照实证**：
   - 剖析单点存在性口径在多 GT 下导致 Top-1 虚高与未生成数偏离的原因；
   - 升级为动态前缀二分图匹配保真口径（Prefix Bipartite Incremental Matching），实现 $\sum \mathbb{I}[r \le k] \equiv H^{@k}$ 与代数闭合。

3. **多时刻查询 ($M \ge 2$) 完整事件覆盖度与候选分工实证分析**：
   - 建立 4 态互斥分类（找齐全部 $m=M$ / 找到多个 $1<m<M$ / **只找到一个 $m=1$** / 完全没找到 $m=0$）；
   - 揭示原生基线模型在 Top-5 预算下多时刻找齐率普遍崩塌至 3%~4%，且超 35%~45% 查询退化为“只找到一个”的单峰吸附困境；
   - 证明 DQ-CGPv3 使 UVCOM Top-5 找齐率暴增 7.67 倍（6 条 $\to$ 46 条），零覆盖骤减 51 条。
   - 详见 [`tables/table_multigt_coverage_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed_supplementary/tables/) 与 [`tables/comparison_multigt_coverage_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed_supplementary/tables/)。

4. **IoU 判定阈值敏感性检查 ($\tau \in \{0.3, 0.4, 0.5\}$)**：
   - 全量测算 8 大骨干网络在宽松 ($\tau=0.3$)、基准 ($\tau=0.4$) 与严格 ($\tau=0.5$) 门槛下的覆盖率与排序损失；
   - 证实“生成缺陷型”与“排序缺陷型”的机理划分在所有阈值下完全恒定不变。
   - 详见 [`tables/table_iou_sensitivity_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed_supplementary/tables/)。

5. **实验 5 对比实证：候选在 Decoder 层间如何变化？(Method 1 Baseline vs Method 2 DQ-CGPv3)**：
   - 统一采用各层未经后处理的原生输出与 IoU $\ge 0.4$ 一对一最大二分图独立匹配；
   - 揭示原生 Moment-DETR 在 D1 $\to$ D2 提精中的严重退化（丢弃 139 / 387 个已定位事件，净减 -61 / -157）；证明 DQ-CGPv3 将退化彻底逆转为正向提精（净增 +33 / +71，最终独立命中暴增 +150 / +461）；
   - 证实 DQ-CGPv3 在初始层 D1 确立显著高位先验（UVCOM D1 +126/+391, BM-DETR D1 +56/+178），并在多层提精中实现更强的锁定保持（Keep 显著提升，丢失率骤降 37%~70%）；
   - 详见 [`tables/comparison_layer_evolution_standard.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed_supplementary/tables/comparison_layer_evolution_standard.md) 与 [`tables/comparison_layer_evolution_full.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed_supplementary/tables/comparison_layer_evolution_full.md)。

---

## 🚀 快速复现指令

在仓库根目录下运行：
```bash
python3 diagnosed_supplementary/run_supplementary_experiments.py
```
即可基于 `diagnosed/outputs/raw_predictions/` 下的无损全层预测数据，在数秒内全自动重新生成本目录下的所有表格与 JSON 产物。

---

## 📂 完整目录结构

```
diagnosed_supplementary/
├── README.md                                          # 本说明文件
├── 补充实验报告_一致性校验_多时刻分工与敏感性分析.md      # 完整学术级补充实验分析报告
├── run_supplementary_experiments.py                   # 一键运行与表格生成脚本
├── tables/                                            # 补充实验独立表格库
│   ├── checkpoint_and_metric_consistency.md           # 检查点加载与评测指标一致性核验全景报告
│   ├── table_multigt_coverage_m1_test_standard.md     # 多时刻 4 态覆盖表: M1 Standard
│   ├── table_multigt_coverage_m1_test_full.md         # 多时刻 4 态覆盖表: M1 Full
│   ├── table_multigt_coverage_m2_test_standard.md     # 多时刻 4 态覆盖表: M2 Standard
│   ├── table_multigt_coverage_m2_test_full.md         # 多时刻 4 态覆盖表: M2 Full
│   ├── comparison_multigt_coverage_standard.md        # 多时刻分工改善对比表: Standard
│   ├── comparison_multigt_coverage_full.md            # 多时刻分工改善对比表: Full
│   ├── comparison_layer_evolution_standard.md         # 实验5 Decoder逐层演化对比表: Standard
│   ├── comparison_layer_evolution_full.md             # 实验5 Decoder逐层演化对比表: Full
│   ├── table_iou_sensitivity_m1_test_standard.md      # IoU 敏感性表: M1 Standard (0.3/0.4/0.5)
│   ├── table_iou_sensitivity_m1_test_full.md          # IoU 敏感性表: M1 Full (0.3/0.4/0.5)
│   ├── table_iou_sensitivity_m2_test_standard.md      # IoU 敏感性表: M2 Standard (0.3/0.4/0.5)
│   └── table_iou_sensitivity_m2_test_full.md          # IoU 敏感性表: M2 Full (0.3/0.4/0.5)
└── outputs/                                           # 结构化补充分析 JSON 结果
    ├── checkpoint_consistency.json                    # 检查点与指标一致性核验结构化数据
    ├── multigt_coverage_summary.json                  # 多时刻覆盖统计结构化数据
    ├── iou_sensitivity_summary.json                   # IoU 敏感性多阈值统计结构化数据
    └── layer_evolution_summary.json                   # 实验5 Decoder逐层演化结构化数据
```
