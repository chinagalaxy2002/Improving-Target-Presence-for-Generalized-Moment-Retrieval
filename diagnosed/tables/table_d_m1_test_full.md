# 表 D：Decoder Slot 索引塌缩与利用均衡度审计 (Method 1: GMR Baseline | Full-Decodable Test (2,970 queries))

> **统计指标说明**：
> - **K 槽位**：该骨干模型原生 Decoder Query 数量。
> - **Top-1 占优 Slot**：成为 Top-1 预测次数最多的 Query Slot 编号及样本占比。
> - **独立命中占优 Slot**：在二分图独立匹配中命中 GT 次数最多的 Query Slot 编号及样本占比。
> - **Top-2 集中度 (Top-2 Concentration)**：前 2 个最活跃 Slot 贡献的 Top-1 或 Hit 比例之和。
> - **索引塌缩判断 (Index Collapse Verdict)**：如果少数 1-2 个 Slot 垄断了 70% 以上的有效预测，则判定存在严重索引塌缩。

| 骨干模型 (Backbone) | K 槽位 | 最活跃 Top-1 Slot | Top-1 占比 (%) | Top-2 Slot Top-1 集中度 (%) | 最多命中 Slot | 命中占比 (%) | Top-2 Slot 命中集中度 (%) | 索引塌缩判定 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bam_detr** | 10 | Slot Q7 | 19.9% | **35.6%** | Slot Q9 | 12.7% | **24.9%** | ✅ 槽位使用较均匀 (Uniform) |
| **bm_detr** | 10 | Slot Q0 | 100.0% | **100.0%** | Slot Q2 | 21.3% | **39.1%** | ⚠️ 严重索引塌缩 (Severe Collapse) |
| **cg_detr** | 10 | Slot Q6 | 58.7% | **71.6%** | Slot Q6 | 34.3% | **45.4%** | ⚠️ 严重索引塌缩 (Severe Collapse) |
| **eatr** | 10 | Slot Q8 | 28.0% | **48.6%** | Slot Q8 | 17.5% | **30.9%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **moment_detr** | 10 | Slot Q1 | 56.9% | **97.1%** | Slot Q1 | 27.7% | **39.6%** | ⚠️ 严重索引塌缩 (Severe Collapse) |
| **qd_detr** | 10 | Slot Q8 | 62.6% | **76.5%** | Slot Q8 | 23.9% | **37.1%** | ⚠️ 严重索引塌缩 (Severe Collapse) |
| **sim_detr** | 10 | Slot Q0 | 35.4% | **54.4%** | Slot Q0 | 14.1% | **27.6%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **uvcom** | 30 | Slot Q2 | 45.8% | **87.7%** | Slot Q2 | 8.5% | **16.6%** | ⚠️ 严重索引塌缩 (Severe Collapse) |