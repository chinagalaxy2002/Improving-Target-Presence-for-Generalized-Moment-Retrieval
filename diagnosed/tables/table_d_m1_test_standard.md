# 表 D：Decoder Slot 索引塌缩与利用均衡度审计 (Method 1: GMR Baseline | Standard Test (1,036 queries))

> **统计指标说明**：
> - **K 槽位**：该骨干模型原生 Decoder Query 数量。
> - **Top-1 占优 Slot**：成为 Top-1 预测次数最多的 Query Slot 编号及样本占比。
> - **独立命中占优 Slot**：在二分图独立匹配中命中 GT 次数最多的 Query Slot 编号及样本占比。
> - **Top-2 集中度 (Top-2 Concentration)**：前 2 个最活跃 Slot 贡献的 Top-1 或 Hit 比例之和。
> - **索引塌缩判断 (Index Collapse Verdict)**：如果少数 1-2 个 Slot 垄断了 70% 以上的有效预测，则判定存在严重索引塌缩。

| 骨干模型 (Backbone) | K 槽位 | 最活跃 Top-1 Slot | Top-1 占比 (%) | Top-2 Slot Top-1 集中度 (%) | 最多命中 Slot | 命中占比 (%) | Top-2 Slot 命中集中度 (%) | 索引塌缩判定 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bam_detr** | 10 | Slot Q7 | 16.7% | **30.5%** | Slot Q7 | 13.1% | **26.0%** | ✅ 槽位使用较均匀 (Uniform) |
| **bm_detr** | 10 | Slot Q0 | 100.0% | **100.0%** | Slot Q2 | 21.7% | **40.9%** | ⚠️ 严重索引塌缩 (Severe Collapse) |
| **cg_detr** | 10 | Slot Q6 | 50.2% | **63.1%** | Slot Q6 | 29.3% | **41.0%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **eatr** | 10 | Slot Q8 | 25.7% | **45.2%** | Slot Q8 | 21.0% | **35.5%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **moment_detr** | 10 | Slot Q0 | 60.3% | **100.0%** | Slot Q1 | 26.7% | **39.8%** | ⚠️ 严重索引塌缩 (Severe Collapse) |
| **qd_detr** | 10 | Slot Q8 | 45.0% | **65.1%** | Slot Q8 | 23.3% | **37.6%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **sim_detr** | 10 | Slot Q0 | 37.1% | **56.2%** | Slot Q3 | 16.4% | **30.3%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **uvcom** | 30 | Slot Q7 | 58.5% | **98.2%** | Slot Q24 | 9.2% | **16.3%** | ⚠️ 严重索引塌缩 (Severe Collapse) |