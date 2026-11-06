# 表 D：Decoder Slot 索引塌缩与利用均衡度审计 (Method 2: GMR + DQ-CGPv3 | Standard Test (1,036 queries))

> **统计指标说明**：
> - **K 槽位**：该骨干模型原生 Decoder Query 数量。
> - **Top-1 占优 Slot**：成为 Top-1 预测次数最多的 Query Slot 编号及样本占比。
> - **独立命中占优 Slot**：在二分图独立匹配中命中 GT 次数最多的 Query Slot 编号及样本占比。
> - **Top-2 集中度 (Top-2 Concentration)**：前 2 个最活跃 Slot 贡献的 Top-1 或 Hit 比例之和。
> - **索引塌缩判断 (Index Collapse Verdict)**：如果少数 1-2 个 Slot 垄断了 70% 以上的有效预测，则判定存在严重索引塌缩。

| 骨干模型 (Backbone) | K 槽位 | 最活跃 Top-1 Slot | Top-1 占比 (%) | Top-2 Slot Top-1 集中度 (%) | 最多命中 Slot | 命中占比 (%) | Top-2 Slot 命中集中度 (%) | 索引塌缩判定 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bam_detr** | 10 | Slot Q7 | 19.5% | **34.2%** | Slot Q7 | 15.1% | **29.8%** | ✅ 槽位使用较均匀 (Uniform) |
| **bm_detr** | 10 | Slot Q6 | 98.3% | **99.4%** | Slot Q2 | 19.1% | **36.8%** | ⚠️ 严重索引塌缩 (Severe Collapse) |
| **cg_detr** | 10 | Slot Q9 | 51.5% | **62.5%** | Slot Q9 | 30.1% | **47.2%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **eatr** | 10 | Slot Q9 | 36.2% | **50.4%** | Slot Q9 | 20.5% | **35.9%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **moment_detr** | 10 | Slot Q6 | 48.9% | **66.4%** | Slot Q6 | 32.2% | **43.8%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **qd_detr** | 10 | Slot Q4 | 40.1% | **55.0%** | Slot Q4 | 28.8% | **43.8%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **sim_detr** | 10 | Slot Q6 | 27.8% | **47.2%** | Slot Q9 | 14.1% | **26.4%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **uvcom** | 30 | Slot Q2 | 54.6% | **73.0%** | Slot Q2 | 20.0% | **26.6%** | ⚠️ 严重索引塌缩 (Severe Collapse) |