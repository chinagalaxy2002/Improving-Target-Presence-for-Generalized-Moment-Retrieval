# GMR baseline → GEQ+aux：11骨干实验

本分支研究：**查询相关的逐 clip 辅助监督，能否配合 GEQ 改善广义时刻检索的定位与事件存在性判断。** 当前方法采用“两阶段均保留辅助监督”的训练方案，已完成11个骨干的 Standard / Full Test 评测。

主要观察：相对各自的 GMR baseline，GEQ+aux 在 Standard / Full 上分别有 **9/11、7/11** 个骨干的 mAP 提高，**9/11、8/11** 个骨干的 G1 提高；存在下降结果。Moment-DETR 的同起点对照表明，在 GEQ 中继续加入 aux 提高了 mAP，但 G1 略降。

[方法与命名](#方法与命名) · [11骨干效果对比](#11骨干效果对比) · [高事件保留率](#高事件保留率场景全部11骨干) · [Moment-DETR 对照](#moment-detr同起点对照) · [复现与数据](#复现与数据) · [历史实验](docs/HISTORICAL_EXPERIMENTS.md)

## 方法与命名

本轮主图比较同一骨干的两个阶段：

| 图表名称 | 模型与训练方式 | 推理时的存在性判断 |
|---|---|---|
| **GMR baseline** | Stage 1：从头训练原生 GMR，并加入逐 clip 辅助监督 | 骨干原生存在性分数 |
| **GEQ+aux** | Stage 2：继承 Stage 1 的主干及辅助头，加入 GEQ 并联合微调；辅助头继续更新 | GEQ 输出的存在性分数 |

**本文的 GMR baseline 是本轮带辅助监督的 Stage 1。** 这是沿用当前图表的命名；历史无辅助 GMR 在旧结果包中另行标注。主图的两个阶段对比反映整套 GEQ 联合微调的变化，不能单独证明 aux 优于不加 aux。

### 辅助头和训练目标

Query-aware Clip Head 从视频 encoder 表示及掩码池化后的文本表示计算逐 clip 相关性。有效 clip 的标签来自 GT 时间区间；padding 不参与损失。按每个视频分别平均正区域和负区域 BCE，两个非空组各占一半；空查询只计算负区域 BCE。Dice 只对有真实事件的查询计算：

```text
L_aux    = L_balanced_BCE + 0.2 × L_Dice
Stage 1  = L_native_GMR + 0.1 × L_aux
Stage 2  = L_native_GMR + L_GEQ_BCE + 0.1 × L_aux
```

各骨干的原生定位、显著性等损失继续保留。GEQ token 先读取文本，再逐层读取视频 encoder 表示，学习事件是否存在。辅助头只提供训练监督；推理时定位候选和排序沿用原生骨干，存在性门控使用对应阶段的存在性分数。

**方法溯源：** GEQ 来自本项目[原分支算法](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-standard-geq-full/docs/ALGORITHM.md)；查询相关 Clip Head 来自本项目[最初设计](docs/CLIP_RELEVANCE_ORIGINAL_DESIGN.md)；Balanced BCE + Dice 对应早期配置名 `E3`。GEQ+aux 是这些组件在本项目中的训练组合名称。源码见 [clip.py](src/soccergmr/clip.py)、[geq.py](src/soccergmr/geq.py)、[train.py](src/soccergmr/train.py)，[完整演进记录](docs/HISTORICAL_EXPERIMENTS.md)保留各阶段来源提交。

### 统一训练与选模规则

| 项目 | Stage 1：GMR baseline | Stage 2：GEQ+aux |
|---|---|---|
| 数据与 seed | Standard Train / Val，seed=2023 | 相同 |
| 输入上限 / batch size | 128 video tokens / 32 text tokens，batch=8 | 相同 |
| 学习率 | 原生主干与辅助头3e-5 | 主干1e-5，GEQ与辅助头3e-5 |
| 选 best | Standard Val 正查询 mAP | Standard Val 三阈值 G1 均值 |
| 最大轮数 / patience | 200 / 20 | 200 / 10 |
| 后续 Test | 固定 best 评测 Standard 与 Full | 相同 |

Moment-DETR 复用已完成的模型和4份 Test 结果，保留原始源码身份；其余10个骨干在同一固定源码下完成20个训练阶段与40份 Test。10个新骨干的 Val 选择全部固定后再执行正式 Test，已有实验没有重训。[完整协议](E3_ELEVEN_PROTOCOL.md) · [来源与选模证据](results/e3_eleven/provenance/)

## 11骨干效果对比

**灰色方形为 GMR baseline，蓝色圆形为 GEQ+aux；每个骨干的两条横柱给出具体分数。** 左列 mAP 衡量正查询定位，右列 G1 衡量同时包含事件查询和空查询的广义检索。数值单位均为0–100，越大越好。

本节 G1 是固定存在性阈值 **0.4 / 0.5 / 0.6** 下的 G-mIoU@1 均值；mAP 使用未门控的正查询候选。它与后文“Val 保留率≥95%”场景采用不同的阈值设置。

### Standard Test：1,036条查询

mAP 和 G1 均有9/11个骨干提高；BAM-DETR、TaskWeave 的 G1 下降。

![Standard Test：11骨干mAP及G1对比](results/e3_eleven/STANDARD_SUMMARY.png)

[PDF](results/e3_eleven/STANDARD_SUMMARY.pdf) · [SVG](results/e3_eleven/STANDARD_SUMMARY.svg)

### Full-Decodable Test：2,970 / 2,986条查询

mAP 有7/11个骨干提高，G1有8/11个提高；BAM-DETR、TR-DETR、TaskWeave 的 G1 下降。两套 Test 的提升方向并不对所有骨干、所有指标一致。

![Full Test：11骨干mAP及G1对比](results/e3_eleven/FULL_SUMMARY.png)

[PDF](results/e3_eleven/FULL_SUMMARY.pdf) · [SVG](results/e3_eleven/FULL_SUMMARY.svg) · [两套汇总图合集](results/e3_eleven/TEST_SUMMARY_BOTH.pdf)

<details>
<summary>展开精确数值表（GMR baseline → GEQ+aux）</summary>

| 骨干 | Standard mAP | Standard G1 | Full mAP | Full G1 |
|---|---:|---:|---:|---:|
| Moment-DETR（复用） | 5.49 → 7.47 | 33.51 → 51.80 | 5.33 → 6.44 | 30.62 → 48.46 |
| EaTR | 16.37 → 17.16 | 39.12 → 53.83 | 13.77 → 14.68 | 37.35 → 50.44 |
| QD-DETR | 11.56 → 13.05 | 45.58 → 53.89 | 10.50 → 11.22 | 41.76 → 50.47 |
| CG-DETR | 6.37 → 6.91 | 22.62 → 52.47 | 5.82 → 6.19 | 21.77 → 48.69 |
| Sim-DETR | 20.29 → 21.00 | 55.19 → 55.20 | 19.60 → 19.46 | 51.64 → 52.36 |
| BM-DETR | 4.11 → 4.20 | 52.34 → 52.83 | 3.87 → 3.81 | 48.86 → 49.36 |
| BAM-DETR | 22.97 → 23.64 | 50.65 → 49.85 | 20.33 → 20.59 | 48.31 → 46.38 |
| UVCOM | 22.21 → 21.66 | 44.70 → 45.26 | 18.46 → 19.21 | 42.49 → 43.77 |
| TR-DETR | 2.58 → 2.81 | 48.45 → 48.48 | 2.85 → 2.84 | 45.30 → 45.15 |
| TaskWeave | 3.79 → 3.40 | 49.82 → 48.61 | 3.70 → 3.31 | 46.17 → 45.03 |
| LD-DETR | 6.32 → 6.52 | 34.56 → 49.66 | 6.06 → 6.31 | 32.79 → 46.40 |

</details>

[完整指标 CSV](results/e3_eleven/METRICS.csv) · [逐 tIoU 的 AP、短事件与多区间诊断](results/e3_eleven/DETAILS.json) · [汇总图数值审计](results/e3_eleven/SUMMARY_PLOT_AUDIT.json)

<details>
<summary>展开固定模型的 G-mIoU@1 / @3 完整阈值曲线</summary>

横轴是**存在性阈值**0–1、步长0.01；每张图左列 G-mIoU@1，右列 G-mIoU@3。曲线使用固定的 Val best，只用于观察阈值响应，未按 Test 曲线重选阈值或 checkpoint。

![Standard Test完整阈值曲线](results/e3_eleven/STANDARD_TEST.png)

![Full Test完整阈值曲线](results/e3_eleven/FULL_TEST.png)

[Standard 每骨干单页 PDF](results/e3_eleven/STANDARD_TEST_BY_BACKBONE.pdf) · [Full 每骨干单页 PDF](results/e3_eleven/FULL_TEST_BY_BACKBONE.pdf) · [曲线 CSV](results/e3_eleven/CURVES.csv)

</details>

## 高事件保留率场景：全部11骨干

本节问：**希望尽量保留真实事件查询、供用户继续复核片段时，GEQ+aux 表现如何？** “保留率”仅表示真实事件查询通过存在性门控，不等于定位正确率。

本节覆盖本轮**全部11个骨干**：Moment-DETR、EaTR、QD-DETR、CG-DETR、Sim-DETR、BM-DETR、BAM-DETR、UVCOM、TR-DETR、TaskWeave、LD-DETR；每个都展示 Standard 与 Full 上的 GMR baseline / GEQ+aux 结果，保留所有下降项。

每个骨干、每种方法分别在 **Standard Val 上选择正查询保留率≥95%的最大候选阈值**，再固定用于两套 Test，严格 `score > threshold`。阈值规则沿用[原分支分析](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/tree/682db34ab0511ce0ebfd69e9b966150cac6473a5/results/high_retention)；当前统计范围已从早期六骨干扩展至全部11骨干，未按 Test 增益筛选。

### 两张图的区别

**两张图使用相同模型、相同 checkpoint 和相同 Test 数据，仅阈值候选不同；每张图都是左侧 Standard、右侧 Full，包含全部11骨干。**

| 项目 | 图 A：0.01网格主结果 | 图 B：Val分数断点复核 |
|---|---|---|
| 在 Val 上试哪些阈值 | 0.00、0.01、0.02…1.00 | Val 中实际出现的所有分数，再加0/1 |
| 选阈值目标 | 保留至少95%正查询，取最大候选阈值 | 相同 |
| 作用 | 保持与原分支相同的网格规则 | 检查结论是否依赖0.01的阈值粒度 |

**实例：** Moment-DETR 的 GMR baseline 在图 A 选到 **0.42**，Val 保留率100%；图 B 选到约 **0.428048**，Val 保留率95.29%。细化候选后可以在0.42与0.43之间选择，从而改变 Test 上保留和拒绝的查询数量，G1也随之变化。更细的搜索不保证 Test G1 更高；mAP不使用门控，在两种规则下保持相同。

### 图 A：0.01网格主结果（全部11骨干）

Standard / Full 都有 **9/11** 个骨干提高，11骨干等权平均 G1 分别提高 **1.654 / 0.986 个百分点**。BAM-DETR、LD-DETR在两套 Test 上下降。

灰色方形为 GMR baseline，蓝色圆形为 GEQ+aux；横柱末端标注每个骨干的实际 G1。

![图A：全部11骨干高保留率0.01网格主结果](results/e3_high_retention/PRIMARY.png)

[主图 PDF](results/e3_high_retention/PRIMARY.pdf) · [SVG](results/e3_high_retention/PRIMARY.svg)

<details>
<summary>展开全部11骨干的精确数值（图 A）</summary>

| 骨干 | Standard G1：baseline → GEQ+aux | Δ百分点 | Full G1：baseline → GEQ+aux | Δ百分点 |
|---|---:|---:|---:|---:|
| Moment-DETR | 4.774 → 7.105 | +2.331 | 4.991 → 6.278 | +1.286 |
| EaTR | 21.060 → 22.524 | +1.464 | 21.116 → 22.108 | +0.992 |
| QD-DETR | 16.586 → 17.580 | +0.994 | 14.337 → 14.508 | +0.172 |
| CG-DETR | 4.026 → 9.626 | +5.600 | 4.253 → 7.461 | +3.208 |
| Sim-DETR | 15.494 → 18.810 | +3.316 | 15.180 → 17.880 | +2.700 |
| BM-DETR | 6.692 → 7.954 | +1.262 | 6.576 → 7.106 | +0.530 |
| BAM-DETR | 27.086 → 25.101 | -1.985 | 25.715 → 23.948 | -1.767 |
| UVCOM | 21.630 → 22.074 | +0.444 | 21.393 → 22.548 | +1.155 |
| TR-DETR | 1.725 → 4.447 | +2.722 | 2.380 → 3.300 | +0.920 |
| TaskWeave | 3.211 → 5.314 | +2.103 | 3.045 → 4.934 | +1.888 |
| LD-DETR | 3.824 → 3.771 | -0.053 | 3.887 → 3.644 | -0.243 |

</details>

<details>
<summary>展开图 B：全部11骨干的 Val分数断点复核</summary>

复核中 Standard / Full 分别有 **9/11、8/11** 个骨干提高，平均 G1 分别提高 **0.910 / 0.324 个百分点**。BAM-DETR、LD-DETR在两套 Test 下降，QD-DETR在 Full 也下降。增益幅度及个别骨干的方向对阈值粒度有敏感性。

![图B：全部11骨干高保留率Val分数断点复核](results/e3_high_retention/EXACT_THRESHOLDS.png)

[复核图 PDF](results/e3_high_retention/EXACT_THRESHOLDS.pdf) · [SVG](results/e3_high_retention/EXACT_THRESHOLDS.svg)

</details>

| 阈值规则 | Standard 平均G1：baseline → GEQ+aux | Δ百分点 | Full 平均G1：baseline → GEQ+aux | Δ百分点 |
|---|---:|---:|---:|---:|
| 图 A：0.01网格 | 11.464 → 13.119 | +1.654 | 11.170 → 12.156 | +0.986 |
| 图 B：Val分数断点 | 13.416 → 14.326 | +0.910 | 12.646 → 12.970 | +0.324 |

95%只约束 Val，不保证 Test 仍达到95%。主分析中 GEQ+aux 的11骨干平均 Test 正查询保留率为 **96.37% / 97.95%**，空查询误返回率为 **89.50% / 90.45%**。本节使用 Val 选出的阈值，不能与前文固定0.4/0.5/0.6均值混为同一设置。

[全部11骨干详细报告与 mAP/G1/G3 表](results/e3_high_retention/README.md) · [各模型阈值与实际保留率](results/e3_high_retention/DECISIONS.csv) · [完整指标 CSV](results/e3_high_retention/METRICS.csv) · [176项 G1/G3 核验](results/e3_high_retention/AUDIT.json)

## Moment-DETR：同起点对照

下面两种 GEQ 微调都从同一个带辅助监督的 GMR best 开始，用于区分“加入 GEQ 微调”与“GEQ阶段继续加 aux”的结果。G1仍按固定0.4/0.5/0.6三阈值均值报告。

| 模型 | Standard mAP | Full mAP | Standard G1 | Full G1 |
|---|---:|---:|---:|---:|
| GMR baseline：共同的 Stage 1 起点 | 5.49 | 5.33 | 33.51 | 30.62 |
| GEQ：Stage 2 不加 aux | 6.33 | 5.81 | 52.45 | 48.95 |
| GEQ+aux：Stage 2 继续加 aux（本轮方法） | 7.47 | 6.44 | 51.80 | 48.46 |

相对同起点、GEQ阶段不加 aux 的对照，GEQ+aux 的 Standard / Full **mAP提高1.14 / 0.63个百分点，G1下降0.65 / 0.49个百分点**。因此，当前结果支持定位 mAP 有改善，尚不支持“继续加入 aux 可同时改善存在性与定位”的结论。其余10骨干尚无这一组完整的同起点无 aux GEQ 对照，不能把它们的 Stage 1 → Stage 2 提升全部归因于 aux。

[原始对照数据与历史无辅助 GMR/GEQ 结果](results/E3_baseline_comparison/README.md)

## 复现与数据

Standard Test 共1036条；Full使用2970/2986条可解码查询。两套 Test 有185个视频ID、186个视频—文本输入重叠，不能视为独立重复验证。所有结果为单 seed，未作统计显著性或多 seed 稳定性声明。

### 从发布结果复算和绘图

本分支包含压缩预测、GT快照、指标及哈希。以下命令不需要GPU、模型权重或视频特征：

```bash
pip install -e '.[test,report]'
python scripts/plot_e3_test_summary.py
python scripts/plot_e3_test_results.py
PYTHONPATH=src python scripts/report_e3_high_retention.py --verify-only
PYTHONPATH=src python scripts/verify_e3_publication.py
```

44份 Test 指标与8,888个 G1/G3 曲线点已独立复算，汇总图的88根柱及数值标注逐一对应原始 CSV。[发布核验](PUBLICATION_CHECKS.json) · [文件哈希清单](PUBLICATION_MANIFEST.json)

### 训练与固定 Test

配置见 [configs/experiments](configs/experiments)，数据布局见 [DATA.md](docs/DATA.md)。配置中的数据、特征及架构路径保留了训练机器的绝对路径，迁移时需显式修改。模型权重和特征不随仓库发布。以下以 EaTR 为例说明单个骨干的完整两阶段流程：

```bash
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m soccergmr train --config configs/experiments/eatr.json \
  --stage gmr --device cuda:0 --output runs/reproduction/eatr/gmr
python -m soccergmr train --config configs/experiments/eatr.json \
  --stage geq_full --baseline runs/reproduction/eatr/gmr/best.pt \
  --device cuda:0 --output runs/reproduction/eatr/geq_full
# 全部计划中的 Val 模型固定后，再对固定 checkpoint 执行 Test；该命令评测两套 Test。
python -m soccergmr test --config configs/experiments/eatr.json \
  --checkpoint runs/reproduction/eatr/geq_full/best.pt \
  --device cuda:0 --output runs/reproduction/eatr/geq_full_test
```

Stage 1 的固定 Test 使用同一个 `test` 命令，将 `--checkpoint` 指向 `gmr/best.pt`、输出目录换为 `gmr_test`。10个新骨干的训练源码为 [`d22986a`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/commit/d22986a671076e700b7c420b259009215ce5cead)，源码哈希 `7d69e39b16dc91339b3c82f3d25e4b581896397f00283ef0dbd5f025fa6cb062`；Moment-DETR保留原始来源身份。续训使用完整 `last.pt` 及原身份，不跳过源码/配置/资产校验。

<details>
<summary>展开11骨干两阶段训练曲线</summary>

![11骨干训练曲线](results/e3_eleven/training_val_panel.png)

[PDF](results/e3_eleven/training_val_panel.pdf) · [SVG](results/e3_eleven/training_val_panel.svg)

</details>

### 历史记录与完整结果

- [早期方法演进、来源提交及 A/B/C/D 记录](docs/HISTORICAL_EXPERIMENTS.md)
- [E1–E6：只在 GMR 阶段增加不同辅助监督](results/clip_v2/REPORT.md)
- [后续 GEQ+aux 六组微调快照](results/clip_joint_snapshot/README.md)
- [本轮11骨干完整报告](results/e3_eleven/REPORT.md)
- [历史无辅助11骨干基线：保留原协议差异](results/test_thresholds_11_backbones/README.md)
