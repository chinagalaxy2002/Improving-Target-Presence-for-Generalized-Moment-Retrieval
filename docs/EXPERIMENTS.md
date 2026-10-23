# 实验范围与结果解释

## 本仓库实际包含的实验

研究仅使用原始 Standard 数据。Moment-DETR、QD-DETR、CG-DETR、EaTR 各包含 Baseline、GMR、GMR + S4_minus_Fusion 三种设置；FlashVTG 包含已经验证的 E0/GMR 和 GMR + S4_minus_Fusion 两种设置，共 **14 组本地实验**。

没有无 GMR 的 FlashVTG 实验行，也没有数据增强实验。六组 MLLM 行是 README 标明的论文 Table 4 转录，不属于本地重跑结果，其 Rej-F1 的原始阈值为 .4，不能当成 .5 或 .6 的测量。

| 内容 | 仓库位置 |
| --- | --- |
| .4 完整表 | [TABLE2_COMPLETE_RESULTS.csv](../backbonefolder/TABLE2_COMPLETE_RESULTS.csv) |
| .5 完整表 | [TABLE2_COMPLETE_RESULTS_tau_0.5.csv](../backbonefolder/TABLE2_COMPLETE_RESULTS_tau_0.5.csv) |
| .6 完整表 | [TABLE2_COMPLETE_RESULTS_tau_0.6.csv](../backbonefolder/TABLE2_COMPLETE_RESULTS_tau_0.6.csv) |
| 本地预测重新计分 | [audit/metrics.csv](audit/metrics.csv) |
| S4 相对同主干 GMR 的差值 | [audit/gains.csv](audit/gains.csv) |
| 每个预测文件及配置的来源哈希 | [audit/sources.json](audit/sources.json) |

## 初始化、选模与测试记录

四个 DETR 主干从各自冻结的 GMR 检查点初始化，使用 seed 2026、六轮适配、全轮保存；本仓库固定报告 epoch 4。训练阶段曾记录 epoch 3–6 的 Test 诊断，不按 Test 最优结果选择 checkpoint。这些结果应表述为固定轮次的探索性比较。

FlashVTG 的 S4 epoch 4 来自仓库中已经归档的历史参考运行，不能称为与另外四个主干共同新训练的第六轮实验。其配置记录 Val `omni_spectrum` 选模。E0/GMR 的选择规则应以对应冻结记录为准，不应统一改写成同一规则。

共同 S4 超参数为 AdamW，常规模块学习率 3e-5、LoRA 学习率 8e-5、weight decay 1e-4、梯度裁剪 .1、StepLR 每 8 轮乘 .5。Moment-DETR/FlashVTG batch 为 8，QD/CG/EaTR 为 32。各自 native baseline 的训练预算与目标不完全相同，这些结果不是等预算、等选模规则的单组件消融。

依据见 [共同训练代码](<../Phase2(addbackbone_with_S4_minusFusion)/shared/train.py>)、[共同配置](<../Phase2(addbackbone_with_S4_minusFusion)/shared/config.json>)、各主干 `configuration/CONFIG.json` 和 [代码映射](CODE_MAP.md)。QD-DETR-GMR、CG-DETR-GMR 为本地适配扩展，不能冒充参考论文中同名实验的已发表结果。

## 指标定义

官方判决为 `existence_score > tau` 时保留候选，否则输出空集合。AUROC 衡量阈值无关的正负查询排序；负例拒识率为 `TN/(TN+FP)`，正例保留率为 `TP/(TP+FN)`。拒识 F1 为

$$
\operatorname{RejF1}=2TN/(2TN+FN+FP).
$$

mAP、mR@k、mR+@k 在正查询上计算，不施加存在性硬门控。mAP 使用最多十个预测和 .50:.05:.95 的 tIoU 阈值。mR 通过贪心一对一匹配计算每条正查询的召回并平均；mR+ 只评估多目标查询，使用 `max(0, matched-1)/(GT_count-1)`，不把第一个命中计作额外目标召回。

G-mIoU@k 在全部查询上先执行存在性门控，再评估 top-k 区间集合。预测与 GT 同为空时记 1，只有一方为空时记 0；两者均非空时按原实现匹配并做集合归一化。因此，较高的拒识率可显著抬高整体 G 指标。Test 全部拒识可得到 **47.49%** 的 G-mIoU@k，必须同时检查正例保留和定位。

指标源码见 [official_eval/metrics.py](<../Phase2(addbackbone_with_S4_minusFusion)/shared/official_eval/metrics.py>)。复现时保留仓库的候选顺序、top-k、清理和后处理逻辑，不另行修改指标实现。

## 固定 .4 的主干对比

下表为 S4_minus_Fusion 相对**同一主干 GMR** 的变化，单位为百分点。全部数值由本仓库预测重算，并与发布表格对齐。

| 主干 | ΔG-mIoU@1 | ΔmAP | ΔAUROC | Δ负例拒识率 | Δ正例保留率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Moment-DETR | +46.09 | +0.91 | +0.06 | +100.00 | −60.29 |
| QD-DETR | +10.84 | +1.56 | −0.80 | +24.80 | −17.46 |
| CG-DETR | +27.68 | +0.68 | +0.32 | +65.85 | −41.18 |
| EaTR | +15.26 | +0.81 | −0.53 | +36.79 | −26.47 |
| FlashVTG | +23.44 | +0.41 | +0.85 | +56.71 | −35.66 |

该工作点上，五个主干的 G-mIoU@1 和非硬门控 mAP 均提高，但正例保留均下降。QD-DETR、EaTR 的 AUROC 下降，不能概括为所有拒识能力指标均改善。

Moment-DETR-GMR 在 .4 下不拒识任何负例，S4 在该阈值下拒识所有负例；G@1 的巨大变化伴随 AUROC 仅 +.06。这说明分数分布及阈值行为对结果解释很重要。

FlashVTG 在 .4 下，G@1 从 28.10% 升至 51.54%，mAP 从 25.43% 升至 25.84%，正例保留从 86.76% 降至 51.10%。mAP 未硬门控，因此它的提高不能证明实际保留下来的正查询定位也同步提高。

## 三个阈值下的边界

| 主干 | ΔG@1，.4 | ΔG@1，.5 | ΔG@1，.6 |
| --- | ---: | ---: | ---: |
| Moment-DETR | +46.09 | +9.42 | −0.10 |
| QD-DETR | +10.84 | +5.28 | +2.82 |
| CG-DETR | +27.68 | +19.25 | +7.83 |
| EaTR | +15.26 | +12.27 | +4.70 |
| FlashVTG | +23.44 | +10.37 | +4.87 |

不能把 .4 下的“五主干改善”扩大成所有阈值上的改善。三个阈值使用相同冻结预测，不代表三次独立实验，也不应根据 Test 扫描挑选部署阈值。

当前材料支持五个具体适配器的实现与上述工作点比较。单个数据集、单个适配 seed、不同选模与预算，以及没有受控组件消融，限制了统计稳定性、因果机制和效率方面的结论。仓库没有为这些缺失证据提供新的实验。
