# 表 D：Decoder Slot 索引塌缩与利用均衡度审计 (Method 2: GMR + DQ-CGPv3 | Full-Decodable Test (2,970 queries))

> **统计指标说明**：
> - **K 槽位**：该骨干模型原生 Decoder Query 数量。
> - **Top-1 占优 Slot**：成为 Top-1 预测次数最多的 Query Slot 编号及样本占比。
> - **独立命中占优 Slot**：在二分图独立匹配中命中 GT 次数最多的 Query Slot 编号及样本占比。
> - **Top-2 集中度 (Top-2 Concentration)**：前 2 个最活跃 Slot 贡献的 Top-1 或 Hit 比例之和。
> - **索引塌缩判断 (Index Collapse Verdict)**：如果少数 1-2 个 Slot 垄断了 70% 以上的有效预测，则判定存在严重索引塌缩。

| 骨干模型 (Backbone) | K 槽位 | 最活跃 Top-1 Slot | Top-1 占比 (%) | Top-2 Slot Top-1 集中度 (%) | 最多命中 Slot | 命中占比 (%) | Top-2 Slot 命中集中度 (%) | 索引塌缩判定 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **bam_detr** | 10 | Slot Q7 | 20.9% | **36.3%** | Slot Q7 | 15.2% | **29.6%** | ✅ 槽位使用较均匀 (Uniform) |
| **bm_detr** | 10 | Slot Q6 | 99.4% | **99.9%** | Slot Q6 | 21.2% | **38.3%** | ⚠️ 严重索引塌缩 (Severe Collapse) |
| **cg_detr** | 10 | Slot Q9 | 65.3% | **73.1%** | Slot Q9 | 32.9% | **48.4%** | ⚠️ 严重索引塌缩 (Severe Collapse) |
| **eatr** | 10 | Slot Q9 | 43.0% | **57.4%** | Slot Q9 | 22.1% | **34.2%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **moment_detr** | 10 | Slot Q6 | 67.0% | **78.5%** | Slot Q6 | 29.1% | **42.5%** | ⚠️ 严重索引塌缩 (Severe Collapse) |
| **qd_detr** | 10 | Slot Q4 | 59.3% | **69.8%** | Slot Q4 | 30.4% | **46.4%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **sim_detr** | 10 | Slot Q6 | 30.6% | **48.8%** | Slot Q9 | 13.1% | **25.4%** | ⚡ 中度槽位偏置 (Moderate Bias) |
| **uvcom** | 30 | Slot Q2 | 57.2% | **75.2%** | Slot Q2 | 16.9% | **23.7%** | ⚠️ 严重索引塌缩 (Severe Collapse) |