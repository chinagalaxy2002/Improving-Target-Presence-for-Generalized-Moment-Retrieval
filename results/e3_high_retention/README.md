# 高事件保留率场景：本轮六骨干配对评测

比较本轮 **GMR baseline → GEQ+aux**。GMR baseline 按用户要求命名，实际为带 Balanced BCE + 0.2 Dice 辅助监督的 Stage 1；不等同于历史无辅助 GMR。Moment-DETR 复用已完成实验，其余模型使用本轮固定源码。没有重新训练或重选 checkpoint。

骨干范围事先沿用参考分支的 Moment-DETR、Sim-DETR、BM-DETR、UVCOM、TaskWeave、LD-DETR，未依据本次 Test 增益删选。原六骨干集合来自历史探索性分析，不能视为所有11骨干的普遍结论。

## 阈值规则

每个骨干、每种方法分别在 Standard Val（465条，255正/210空）选择正查询保留率 ≥95% 的最大候选阈值，然后固定用于两套 Test。严格采用 `pred_exist_score > threshold`。主分析搜索0–1、步长0.01；精确复核候选为全部 Val 分数断点及0/1，与参考脚本一致（不是连续实数域优化）。Test 不参与阈值或模型选择。

保留率只表示事件查询通过存在性门控，不代表定位正确率。mAP 不使用门控，两种阈值搜索的 mAP 完全相同。

## 两张图应该怎么读

**两张图比较同一批模型与 checkpoint，每张都是左侧 Standard、右侧 Full。唯一改变的是在 Val 上选择存在性阈值时，允许使用哪些候选值。**

| 项目 | 图 A：0.01 网格主结果 | 图 B：Val 分数断点复核 |
|---|---|---|
| 候选阈值 | 0.00、0.01、0.02…1.00 | Val 中实际出现的全部分数，加0/1 |
| 选阈值目标 | Val 正查询保留率≥95%的最大候选阈值 | 相同 |
| 模型、数据及评分规则 | 同一批固定 checkpoint、两套 Test、严格 score > τ | 相同 |
| 用途 | 与原分支保持相同网格的主结果 | 检查结论是否依赖0.01这一网格粒度 |

例如，Moment-DETR 的 GMR baseline 在图 A 使用 τ=0.42（Val 保留率100%），图 B 使用 τ≈0.428048（Val 保留率95.29%）。图 B 可以在0.42和0.43之间选择，因而可能拒绝更多空查询，也可能多拒绝部分正查询，最终 Test G1 会变化。更细的搜索不保证 Test G1 更高；这里优化的是 Val 保留率约束下的阈值，并非 Test G1。完整精度见 DECISIONS.csv。

正文的常规11骨干汇总使用固定0.4/0.5/0.6三阈值均值，本节采用“Val 保留率≥95%”选出的阈值，回答不同的使用场景，分数不应直接混为同一评测设置。

## 图 A：0.01 网格主结果

主分析中 Standard / Full 都有5/6骨干提高，LD-DETR下降。每个方法的阈值只在 Val 上确定。

![图A：0.01网格主结果](PRIMARY.png)

[PDF](PRIMARY.pdf) · [SVG](PRIMARY.svg)

| 骨干 | Standard GMR→GEQ+aux | Δ百分点 | Full GMR→GEQ+aux | Δ百分点 |
|---|---:|---:|---:|---:|
| Moment-DETR | 4.774→7.105 | +2.331 | 4.991→6.278 | +1.286 |
| Sim-DETR | 15.494→18.810 | +3.316 | 15.180→17.880 | +2.700 |
| BM-DETR | 6.692→7.954 | +1.262 | 6.576→7.106 | +0.530 |
| UVCOM | 21.630→22.074 | +0.444 | 21.393→22.548 | +1.155 |
| TaskWeave | 3.211→5.314 | +2.103 | 3.045→4.934 | +1.888 |
| LD-DETR | 3.824→3.771 | -0.053 | 3.887→3.644 | -0.243 |

## 图 B：Val 分数断点复核（同一模型，更细的阈值候选）

复核中两套 Test 仍为5/6骨干提高，但平均增益缩小，LD-DETR下降更明显。这张图用于展示对阈值粒度的敏感性，不是一轮新训练。

![图B：Val分数断点复核](EXACT_THRESHOLDS.png)

[PDF](EXACT_THRESHOLDS.pdf) · [SVG](EXACT_THRESHOLDS.svg)

| 阈值搜索 | Test | GMR平均 | GEQ+aux平均 | Δ百分点 | 相对增益 | 提高骨干数 |
|---|---|---:|---:|---:|---:|---:|
| grid_101 | standard | 9.271 | 10.838 | +1.567 | +16.90% | 5/6 |
| grid_101 | full | 9.179 | 10.398 | +1.220 | +13.29% | 5/6 |
| val_breakpoints | standard | 11.488 | 12.384 | +0.896 | +7.80% | 5/6 |
| val_breakpoints | full | 10.781 | 11.457 | +0.676 | +6.27% | 5/6 |

汇总先等权平均六骨干原始 G1，再计算相对增益。图和表保留下降结果，不沿用旧分支“六个均提高”的结论。

## Test 实际保留率及空查询误返回率

95%只约束 Val，Test 保留率允许低于95%。以下为0.01网格下六骨干等权平均，完整逐模型值及阈值见 DECISIONS.csv。

| Test | GMR正查询保留率 | GEQ+aux正查询保留率 | GMR空查询误返回率 | GEQ+aux空查询误返回率 |
|---|---:|---:|---:|---:|
| standard | 98.96% | 97.15% | 96.27% | 92.72% |
| full | 99.27% | 98.40% | 95.84% | 93.01% |

Standard Test 1036条，Full-Decodable 2970/2986条；两个 Test 有重叠（185个视频ID、186个视频—文本输入），不能当作独立重复。单 seed=2023，不作统计显著性声明。主图关注高保留率场景，不能由此推断默认0.4/0.5/0.6门控或全部场景均改善。

## 数据与复算

[逐模型 mAP/G1/G3](METRICS.csv) · [汇总](SUMMARY.csv) · [阈值/保留率](DECISIONS.csv) · [来源与哈希](EVIDENCE.json) · [数值审计](AUDIT.json)

36份预测由12份本目录 Val 压缩预测及24份 ../e3_eleven 中 Test 压缩预测组成，同时提供 GT 快照。不需模型权重、特征或 GPU。逐文件核验哈希、覆盖与 checkpoint 绑定；96个 G1/G3 值与官方实现核验，24个 mAP 与11骨干报告独立交叉核验。

```bash
PYTHONPATH=src python scripts/report_e3_high_retention.py
PYTHONPATH=src python scripts/report_e3_high_retention.py --verify-only
```
