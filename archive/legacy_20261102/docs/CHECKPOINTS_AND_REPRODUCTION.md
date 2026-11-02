# 检查点与复现索引

本文保留完整实验说明和**服务器实际路径**。GitHub 源码已统一为 `backbones/`
与 `shared/`，布局与简明命令见 [仓库首页](../README.md)。下文所有 `/home/guoxiangyu/...`
路径仍指服务器旧运行目录，不表示 GitHub 里存在同名文件夹。

本分支专门记录将 Mr. DETR 的 Instructive Multi-Route Training 迁移到
Generalized Moment Retrieval（GMR）的实现与实验。目标是在保持
GMR-Adapter 空集拒绝能力的同时，通过训练期 Temporal One-to-Many（O2M）
监督提高多个不同相关片段的定位与召回。

简明交接入口：[`HANDOFF.md`](../HANDOFF.md)，仅包含
Baseline 复现、六种方法、检查点测试和后续微调的衔接步骤。

检查点交接入口：下方 [全部 24 组检查点目录](#10-全部检查点与实验目录) 和
[启动测试及复现命令](#11-启动测试与复现命令) 记录服务器绝对路径、未完成任务的
预期路径、历史 Baseline 来源以及各骨干的脚本入口。文件状态为 2026-10-31 快照；
训练过程中的最新状态以服务器 `status.json` 和检查点实际存在情况为准。

当前覆盖四个 DETR 类时序定位骨干：

- Moment-DETR-GMR
- EaTR-GMR
- QD-DETR-GMR
- CG-DETR-GMR

新训练的路线实验使用固定随机种子 `2023`，不进行多 Seed 重复。EaTR Baseline
按指令复用此前训练好的 seed `2026` GMR；其余 EaTR 方法继续原队列。主路线仍使用原生
Hungarian One-to-One（O2O）集合预测；辅助路线只在训练阶段执行，推理时不会
增加 Query 数量或辅助 Decoder 计算。

## 1. 核心设计

### 1.1 主路线：O2O 集合预测

主路线保留原始 GMR 定义：10 个主 Queries、Hungarian 匹配、分类损失、时间
边界 L1 与 Temporal GIoU 损失。一个 Query 最多匹配一个 GT，一个 GT 在主路线
最多监督一个 Query。推理结果只来自这条路线。

### 1.2 Temporal One-to-Many 辅助监督

每条辅助路线使用 24 个训练专用 Queries。固定匹配时，每个 GT 复制两次后执行
Hungarian 匹配，因此一个 GT 最多监督两个不同预测，而一个预测仍只对应一个 GT。
辅助路线复用原始分类、L1 和 Temporal GIoU 定义，不改变 GMR 的多目标标注。

需要区分两个概念：

- **Multi-Moment Retrieval**：一条文本查询可以有多个不同 GT 片段；
- **One-to-Many Matching**：训练时一个 GT 可以监督多个辅助预测。

本实验将两者结合，O2M 不替代 GMR 的多片段集合定义。

### 1.3 Separate FFN

Separate FFN 路线与主路线共享 Decoder Self-Attention 和 Cross-Attention，但
为辅助任务提供独立的 `linear1`、`linear2` 和 FFN 后 `norm3`。它用于隔离
O2O 与 O2M 对 FFN 参数可能产生的不同优化方向，同时继续共享跨模态注意力表示。

### 1.4 Instruction Tokens

Instruction 路线在每层 Decoder Self-Attention 中拼接 4 个可学习 Token。它们
参与 Query、Key、Value 交互，完成 Self-Attention 后立即丢弃，不进入
Cross-Attention、边界预测集合或 Hungarian 匹配。该机制通过路线条件信息调节
共享 Decoder，而不增加推理输出。

### 1.5 Adaptive O2M

固定 O2M 强制每个 GT 获得两个匹配，可能引入低质量正样本。Adaptive 路线至少
保留一个匹配；只有第二候选与 GT 的 Temporal IoU 不低于 `0.25` 时才增加第二个
监督预测。匹配决策不参与反向传播。

### 1.6 GMR-Adapter 与空集样本

GMR-Adapter 始终从主路线特征预测存在性分数。总损失为：

```text
L = L_o2o + 1.0 * L_exist + sum(lambda_route * L_o2m_route)
```

单辅助路线权重为 `0.30`；双辅助路线各为 `0.15`，因此 E1–E5 的辅助总权重
始终为 `0.30`。空 GT 样本继续计算主路线背景分类和存在性 BCE；辅助路线只计算
背景分类，不计算时间边界回归。

## 2. 六组实验方法

| ID | 配置 | 方法说明 | 辅助权重 |
|---|---|---|---:|
| E0 | `baseline` | 原始 O2O 主路线，无 O2M 辅助路线 | 0 |
| E1 | `shared` | 一条固定 K=2 O2M 路线；SA、CA、FFN、Norm 全共享 | 0.30 |
| E2 | `sepffn` | 一条固定 K=2 O2M 路线；共享 SA/CA，辅助 FFN 与 norm3 独立 | 0.30 |
| E3 | `instruction` | 一条固定 K=2 O2M 路线；共享 Decoder，Self-Attention 加 4 个 Instruction Tokens | 0.30 |
| E4 | `multi` | Separate FFN 与 Instruction 两条固定 O2M 路线同时训练 | 0.15 + 0.15 |
| E5 | `adaptive` | E4 的双路线结构，第二个正样本由 tIoU≥0.25 动态决定 | 0.15 + 0.15 |

关键消融比较为：`E1−E0` 检查密集监督；`E2−E1` 检查 FFN 隔离；
`E3−E1` 检查指令调节；`E4−E2/E3` 检查互补性；`E5−E4` 检查动态匹配。

## 3. Backbone 适配

| Backbone | 辅助 Query 实现 | 适配说明 |
|---|---|---|
| Moment-DETR-GMR | 24 个训练专用 Object Queries | 直接复用原生 Transformer Decoder、Hungarian Matcher 与 GMR-Adapter |
| EaTR-GMR | 24 个学习型辅助 Slots | EaTR 使用 Slot Attention，因此共享原生 Slot 投影与更新 MLP，而不是复制 DAB Query |
| QD-DETR-GMR | 24 个辅助 DAB reference queries | 保留原生 Query-Dependent Decoder 与主路线预测头 |
| CG-DETR-GMR | 24 个辅助 DAB reference queries | DataLoader worker 只执行 CPU collate，主进程再搬运到 CUDA，修复多 worker 验证错误 |

三种迁移骨干共用同一份 [`multiroute.py`](../shared/multiroute.py)
和 [`run.py`](../shared/run.py)，各自入口位于
[`backbones/`](../backbones/)。共享核心可以避免三份
Criterion、匹配与早停实现产生漂移。

## 4. 统一训练与测试协议

| 项目 | 设置 |
|---|---|
| Seed | 新训练为 2023；仅复用的 EaTR Baseline 为历史 2026 |
| Epoch 上限 | 400 |
| Optimizer / LR | AdamW / `5e-5` |
| 主 / 辅助 Queries | 10 / 24 |
| Microbatch / 累积 | 已运行和原 EaTR 路线 2 / 4；待启动 QD/CG instruction、multi、adaptive 为 32 / 1 |
| DataLoader | 每任务 `num_workers=2`、pin memory、prefetch 2、persistent workers |
| 验证间隔 | 每 5 epochs |
| 选模 | 验证集正样本定位 mAP 最大 |
| 早停 | 当前 epoch 距 best epoch 达到 20 |
| 队列并发 | 两张 GPU，每张最多同时 2 个任务 |
| 测试 | 训练结束后测试验证集选择的 `best.ckpt`；EaTR Baseline 已复用原冻结 Test 结果 |
| G-mIoU | 官方百分数保留，同时显式输出 `brief_0_1` 的 0–1 数值 |

队列阶段为 `smoke → train/resume → frozen-best test`。每个阶段最多重试三次；
已有 `last.ckpt` 时自动恢复，已有完整任务自动跳过。测试完成后可通过
`GMR_FIGURE_REFRESH` 调用外部图表刷新脚本，图表失败只产生警告，不改变已完成
测试的有效性。

当前任务覆盖规则见 [`execution_plan.json`](../shared/execution_plan.json)：
正在运行的四项不变；待启动的 QD/CG `instruction / multi / adaptive` 共六项已归档
旧状态，并以真实 batch 32、累积 1 从头训练。原始 GMR 检查点的训练 batch 分别为
Moment-DETR 8、EaTR 32、QD-DETR 32、CG-DETR 32。改变 batch 不表示同时改为历史
学习率或历史 seed。Moment-DETR 原六路线的停止过程另见其结果包说明。

## 5. 当前结果与状态

Moment-DETR-GMR 的六组冻结结果、预测、完整 0–1 G-mIoU 扫描和论文图位于
[`backbones/moment_detr/experiment/`](../backbones/moment_detr/experiment/)。这些结果来自固定
seed 2023，但六组按当时指令停止，实际训练 epoch 不完全一致，解释时不能把差异
全部归因于结构。

四个骨干的完整 24 组 Test 表、训练状态、协议和图表统一更新到
**[results/README.md](../results/README.md)**。本检查点文档不再重复维护局部结果表。
原始数值、G-mIoU 的 0–1 转换和指标来源哈希见
[results/snapshot.json](../results/snapshot.json)。后面的检查点路径仍可用于定位权重，
其中历史状态描述以结果 README 标注的快照时间为准。

EaTR Baseline 使用历史 seed2026/batch32/LR3e-5，而其他已完成路线使用
seed2023/microbatch2×accumulation4/LR5e-5；这部分是跨协议参考比较，不能标为严格
同 seed 消融。原 seed2023 EaTR Baseline（mAP 6.86）已归档，不再作为当前 Baseline。

这些是探索性结果。定位指标、额外 GT 覆盖、空集拒绝和 G-mIoU 之间可能
存在权衡，不能只依据单个指标宣称某种路线全面更优。

## 6. 目录结构

```text
.
├── README.md
├── LICENSE
├── HANDOFF.md
├── backbones/
│   ├── moment_detr/                 # source/、入口和 experiment/ 六路线结果包
│   ├── eatr/                        # source/ 与该骨干训练/测试入口
│   ├── qd_detr/
│   └── cg_detr/
├── shared/                          # 一份公共模型、损失、训练、评估、队列
└── docs/                            # 本索引、历史来源和实验记录
```

## 7. 使用方法

### 单个迁移骨干

```bash
# BACKBONE 可替换为 eatr、qd_detr 或 cg_detr
BACKBONE=eatr
VARIANT=sepffn

CUDA_VISIBLE_DEVICES=0 \
python backbones/$BACKBONE/run.py \
  --variant "$VARIANT" --mode smoke

CUDA_VISIBLE_DEVICES=0 \
python backbones/$BACKBONE/run.py \
  --variant "$VARIANT" --mode train

CUDA_VISIBLE_DEVICES=0 \
python backbones/$BACKBONE/run.py \
  --variant "$VARIANT" --mode test
```

也可以使用每个 Backbone 目录内的 `smoke.sh`、`train.sh` 和 `test.sh`。

### 六配置 × 三迁移骨干队列

```bash
# 首次执行或 smoke 发生变化时检查全部 18 个配置
python shared/validate_all.py

# GPU 0 和 GPU 1 各最多两个任务
GPU0=0 GPU1=1 bash shared/launch_queue.sh

# 查看状态与日志
cat shared/QUEUE.json
tail -f shared/logs/queue.log

# 汇总已完成 Test
python shared/collect_results.py
```

### Moment-DETR-GMR 原始六路线

完整说明见 [Moment-DETR 实验 README](../backbones/moment_detr/experiment/README.md)。
该目录包含实际使用的训练、评估、预检、早停监视、结果重算与绘图脚本；不包含
checkpoint、原始视频或预提取特征。

## 8. 分支范围与清理记录

本分支只保留 Mr. DETR Multi-Route × GMR 实验。以下旧内容与本实验无关，已从
当前分支删除，而不是改成容易误解的新名称：

- `Phase2(addbackbone_with_S4_minusFusion)/`：S4-minus-Fusion 历史实验；
- `ablations/`：GEQ、GAP、Hard-RoI、LoRA 等其他消融；
- `experiments/`：D-GMR、两阶段 GEQ 等历史实验；
- FlashVTG、旧 Table 2 汇总、旧发布清单与复现工具。

删除前内容仍可从提交
[`4e38e85`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/tree/4e38e850c2bc8960dba8a43b326e446b48c122ef)
恢复。当前分支没有改写 Git 历史，只是在后续提交中缩小实验范围。

## 9. 发布边界

仓库不上传 GitHub Token、`.ckpt/.pt/.pth`、CLIP/SlowFast/Text `.npz` 特征或
原始视频。重新训练前需配置本机 SoccerGMR Standard 标签与预提取特征路径。
上游模型代码、数据和评估器遵循各自许可证及引用要求。

## 10. 全部检查点与实验目录

### 10.1 路径约定与状态含义

服务器上存在两个不同的项目根目录，不能互换：

```bash
# Moment-DETR 六路线的实际训练根目录
PROJECT=/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval

# EaTR / QD-DETR / CG-DETR 迁移实验的实际代码根目录
REPO=/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval
CORE="$REPO/backbonefolder/multiroute_gmr"

# 三种迁移骨干的结果与历史权重
RUNS="$CORE/runs"
REUSED="$CORE/reused_baselines"
ARCHIVE="$CORE/archived_runs/original_batch_sizes_20261031"

# 统一图表根目录；每个 Backbone 一个子目录
FIGURES="$PROJECT/newidea/figure/gmr_mr_detr_test_20261030"
```

`best.ckpt` 表示验证集定位 mAP 最佳检查点；训练中已经产生的 best 仍可能更新，
只有完成训练选模后才冻结。三种迁移骨干的 `last.ckpt` 保存优化器、调度器和 RNG，
用于断点恢复。下表标注“预期”的文件尚不存在，不代表已训练或已完成测试。

“排队恢复”表示有历史断点，但当前没有该任务的训练进程。它与 `status.json` 中
可能残留的 `TRAINING` 不同。下面的状态已结合当前队列与进程核对。

### 10.2 Moment-DETR-GMR：六个已完成方法

代码：[`gmr_multiroute_kit/`](../backbones/moment_detr/experiment/gmr_multiroute_kit/)。
表中实验目录相对 `$PROJECT`；完整检查点路径如下。

| 方法 | 实验目录 | best 检查点绝对路径 | 状态 / Best epoch |
|---|---|---|---|
| E0 `baseline` | `mr_experiments/baseline/` | `/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/mr_experiments/baseline/best.ckpt` | 已完成 Test / 5 |
| E1 `shared` | `mr_experiments/shared/` | `/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/mr_experiments/shared/best.ckpt` | 已完成 Test / 5 |
| E2 `sepffn` | `mr_experiments/sepffn/` | `/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/mr_experiments/sepffn/best.ckpt` | 已完成 Test / 20 |
| E3 `instruction` | `mr_experiments/instruction/` | `/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/mr_experiments/instruction/best.ckpt` | 已完成 Test / 30 |
| E4 `multi` | `mr_experiments/multi/` | `/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/mr_experiments/multi/best.ckpt` | 已完成 Test / 15 |
| E5 `adaptive` | `mr_experiments/adaptive/` | `/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/mr_experiments/adaptive/best.ckpt` | 已完成 Test / 5 |

六组均为 seed2023、microbatch2×accumulation4。这个历史训练器只写 `best.ckpt`，
**没有 `last.ckpt` 自动续训文件**；不要使用后面的三骨干恢复规则解释它。训练按当时
指令提前停止，实际训练轮次不同。

Test 产物：`$PROJECT/mr_experiments/<方法>/test/submission.jsonl`、
`evaluation_manifest.json`、`metrics_gmiou_0.4.json`、`metrics_gmiou_0.5.json`、
`metrics_gmiou_0.6.json`。GitHub 中封存的预测与指标位于
[Moment-DETR 冻结结果](../backbones/moment_detr/experiment/mr_experiments/)。

### 10.3 EaTR-GMR：仅 Baseline 复用，其余方法继续原队列

入口：[`backbones/eatr/`](../backbones/eatr/)。
表中实验目录相对 `$CORE`。

| 方法 | 实验目录 | best 检查点绝对路径 | 状态 | bsz×累积 / Seed |
|---|---|---|---|---|
| E0 `baseline` | `runs/eatr/baseline/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/eatr/baseline/best.ckpt` | 已复用，Test mAP 16.20 | 32×1 / 2026 |
| E1 `shared` | `runs/eatr/shared/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/eatr/shared/best.ckpt` | 已完成 Test | 2×4 / 2023 |
| E2 `sepffn` | `runs/eatr/sepffn/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/eatr/sepffn/best.ckpt` | 已完成 Test | 2×4 / 2023 |
| E3 `instruction` | `runs/eatr/instruction/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/eatr/instruction/best.ckpt` | 正在训练，best 已生成 | 2×4 / 2023 |
| E4 `multi` | `runs/eatr/multi/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/eatr/multi/best.ckpt` | 排队恢复，best 已生成 | 2×4 / 2023 |
| E5 `adaptive` | `runs/eatr/adaptive/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/eatr/adaptive/best.ckpt` | 预期；目前只有 last.ckpt，排队恢复 | 2×4 / 2023 |

除复用 Baseline 外，五组的恢复路径均为同目录 `last.ckpt`。`adaptive/last.ckpt`
已存在，但其 `best.ckpt` 尚未产生。Baseline 是推理用格式转换权重，没有当前实验的
`last.ckpt`；`--mode train` 会识别复用标记并直接跳过，不会重训。

EaTR 原始 Baseline 权重保持在 `$REUSED/eatr/best.ckpt`；当前 `runs/eatr/baseline/best.ckpt`
仅增加 `base.` 和 Slot Attention 包装层参数名称，238 个权重张量均未改变。
来源与两份 SHA256 见同目录 `reused_baseline.json`，原训练配置见
`source_effective_config.json`。

### 10.4 QD-DETR-GMR：后三组从头使用 batch32

入口：[`backbones/qd_detr/`](../backbones/qd_detr/)。
表中实验目录相对 `$CORE`。

| 方法 | 实验目录 | best 检查点绝对路径 | 状态 | bsz×累积 / Seed |
|---|---|---|---|---|
| E0 `baseline` | `runs/qd_detr/baseline/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/qd_detr/baseline/best.ckpt` | 已完成 Test | 2×4 / 2023 |
| E1 `shared` | `runs/qd_detr/shared/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/qd_detr/shared/best.ckpt` | 已完成 Test | 2×4 / 2023 |
| E2 `sepffn` | `runs/qd_detr/sepffn/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/qd_detr/sepffn/best.ckpt` | 已完成 Test | 2×4 / 2023 |
| E3 `instruction` | `runs/qd_detr/instruction/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/qd_detr/instruction/best.ckpt` | 预期；排队从头训练 | 32×1 / 2023 |
| E4 `multi` | `runs/qd_detr/multi/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/qd_detr/multi/best.ckpt` | 预期；排队从头训练 | 32×1 / 2023 |
| E5 `adaptive` | `runs/qd_detr/adaptive/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/qd_detr/adaptive/best.ckpt` | 预期；排队从头训练 | 32×1 / 2023 |

后三组的 `best.ckpt` 和 `last.ckpt` 都尚未生成；预期恢复路径分别为对应实验目录的
`last.ckpt`。旧 microbatch2 断点已整体移入 `$ARCHIVE/qd_detr/<方法>/`，新任务不能
从这些旧断点续训。新 batch32 任务第一次失败重试后，可从自己新写入的 last 恢复。

### 10.5 CG-DETR-GMR：前三组继续运行，后三组从头使用 batch32

入口：[`backbones/cg_detr/`](../backbones/cg_detr/)。
表中实验目录相对 `$CORE`。

| 方法 | 实验目录 | best 检查点绝对路径 | 状态 | bsz×累积 / Seed |
|---|---|---|---|---|
| E0 `baseline` | `runs/cg_detr/baseline/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/cg_detr/baseline/best.ckpt` | 正在训练，best 已生成 | 2×4 / 2023 |
| E1 `shared` | `runs/cg_detr/shared/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/cg_detr/shared/best.ckpt` | 正在训练，best 已生成 | 2×4 / 2023 |
| E2 `sepffn` | `runs/cg_detr/sepffn/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/cg_detr/sepffn/best.ckpt` | 正在训练，best 已生成 | 2×4 / 2023 |
| E3 `instruction` | `runs/cg_detr/instruction/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/cg_detr/instruction/best.ckpt` | 预期；排队从头训练 | 32×1 / 2023 |
| E4 `multi` | `runs/cg_detr/multi/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/cg_detr/multi/best.ckpt` | 预期；排队从头训练 | 32×1 / 2023 |
| E5 `adaptive` | `runs/cg_detr/adaptive/` | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/cg_detr/adaptive/best.ckpt` | 预期；排队从头训练 | 32×1 / 2023 |

前三组的 `last.ckpt` 已存在；后三组的 best/last 都是预期路径。所有恢复文件均为
同目录 `last.ckpt`。当前实验统一写 `.ckpt`；历史 CG-GMR 原生权重的扩展名为 `.pt`，
两者不是同一个文件。

### 10.6 历史完整 GMR Baseline 与原始来源

下面四份均已复制到当前实验目录并核对冻结 SHA256。**只有 EaTR 已安装成当前
实验的 Baseline**；另外三份作为历史参考保留，不覆盖当前对应路线。

| Backbone | 当前归档权重绝对路径 | 原 bsz | Seed / LR | 历史 Test mAP |
|---|---|---:|---|---:|
| Moment-DETR | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/reused_baselines/moment_detr/best.ckpt` | 8 | 2026 / 3e-5 | 6.68 |
| EaTR | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/reused_baselines/eatr/best.ckpt` | 32 | 2026 / 3e-5 | 16.20 |
| QD-DETR | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/reused_baselines/qd_detr/best.ckpt` | 32 | 2026 / 3e-5 | 14.62 |
| CG-DETR | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/reused_baselines/cg_detr/best.pt` | 32 | 2026 / 3e-5 | 18.65 |

原始来源根目录为
`/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/`，
其下分别为 `moment_detr/gmr/best.ckpt`、`eatr/gmr/best.ckpt`、
`qd_detr/gmr/best.ckpt`、`cg_detr/gmr/best.pt`。原生权重的参数名称与 Multi-Route
包装不同，不能把它们直接当成当前实验的 `last.ckpt` 恢复优化器。

复制清单、完整原始路径和哈希见
[`reused_baselines/manifest.json`](baseline_sources/manifest.json)。
复用前后的 EaTR Baseline 以及六个重置任务的旧目录均保留在 `$ARCHIVE`；这些归档
不属于当前有效训练路径。

## 11. 启动测试与复现命令

### 11.1 脚本与方法的对应关系

同一个 Backbone 的六种方法使用相同入口，由 `--variant` 或 Shell 脚本第一个
参数选择方法。参数只能取 `baseline / shared / sepffn / instruction / multi / adaptive`，
与上面每一行检查点目录一一对应。

| Backbone | 训练 / Smoke 入口（GitHub 分支相对路径） | Test 入口（GitHub 分支相对路径） |
|---|---|---|
| Moment-DETR | `backbones/moment_detr/experiment/gmr_multiroute_kit/run_train.py`，Smoke 加 `--smoke` | 同目录 `run_eval.py --split test` |
| EaTR | `backbones/eatr/train.sh <方法>`；同目录 `smoke.sh` | 同目录 `test.sh <方法>` |
| QD-DETR | `backbones/qd_detr/train.sh <方法>`；同目录 `smoke.sh` | 同目录 `test.sh <方法>` |
| CG-DETR | `backbones/cg_detr/train.sh <方法>`；同目录 `smoke.sh` | 同目录 `test.sh <方法>` |

EaTR/QD/CG 的入口最终调用 `$CORE/run.py --backbone <eatr|qd_detr|cg_detr>`。
`execution_plan.json` 决定 batch 与复用/重置行为；它不是一个被 README 覆盖的默认建议。

### 11.2 服务器上单个 EaTR / QD / CG 方法的完整流程

下面命令使用第 10.1 节的变量。例子对应 **QD-DETR Separate FFN**；替换两个参数
即可复现三种迁移骨干的任意方法。执行单任务前先查看 GPU 上已有队列，保持每卡
最多两个任务。

```bash
BACKBONE=qd_detr     # eatr / qd_detr / cg_detr
VARIANT=sepffn      # 六种方法之一

# 真实数据前后向检查
CUDA_VISIBLE_DEVICES=0 python -u "$CORE/run.py" \
  --backbone "$BACKBONE" --variant "$VARIANT" --mode smoke

# 无 last.ckpt 时从头训练；有当前协议的 last.ckpt 时恢复
CUDA_VISIBLE_DEVICES=0 python -u "$CORE/run.py" \
  --backbone "$BACKBONE" --variant "$VARIANT" --mode train

# 加载上表对应 best.ckpt，对完整 Test（含空集）推理和评估
CUDA_VISIBLE_DEVICES=0 python -u "$CORE/run.py" \
  --backbone "$BACKBONE" --variant "$VARIANT" --mode test
```

如果只复现现有检查点的 Test，直接执行第三条命令，**不要先运行 train**。
正在训练的任务应等待冻结 best 后再做最终测试。尚无 best 的“预期”条目目前不能
直接测试；脚本会明确报告缺少检查点。

GitHub checkout 中也可使用每个骨干的 Shell 入口：

```bash
# 在本分支 checkout 根目录执行；默认输出位于该 checkout 的 shared/runs/
CUDA_VISIBLE_DEVICES=0 bash backbones/qd_detr/smoke.sh sepffn
CUDA_VISIBLE_DEVICES=0 bash backbones/qd_detr/train.sh sepffn
CUDA_VISIBLE_DEVICES=0 bash backbones/qd_detr/test.sh sepffn
```

复制到另一 checkout 时，默认把权重放到其 `shared/runs/<backbone>/<variant>/`。
也可设置 `GMR_RUNTIME_ROOT` 指向服务器现有 `$CORE`，直接读取旧权重，无需迁移。
上表的服务器绝对路径不会随 Git clone 自动生成，二进制权重也未上传 GitHub。

### 11.3 EaTR Baseline 的复用与测试

服务器上已执行以下两步，重复调用会核验/识别现有结果；它们不训练模型：

```bash
python -B "$CORE/archive_completed_baselines.py"
python -B "$CORE/reuse_eatr_baseline.py"
```

安装后的位置为 `$RUNS/eatr/baseline/best.ckpt`。`--mode smoke` 和 `--mode train`
会打印 `REUSED_COMPLETED_BASELINE` 并跳过训练。需要重新推理验证时：

```bash
CUDA_VISIBLE_DEVICES=0 python -u "$CORE/run.py" \
  --backbone eatr --variant baseline --mode test
```

现有 `test_predictions.jsonl` 和 `test_metrics.json` 来自原来冻结的完整测试，
不需要为了复制权重重复跑 Test。复用模型的 Seed 2026、batch32、LR3e-5 必须在
报告中保留，不能改写为其他方法的 Seed 2023。

### 11.4 Moment-DETR 六方法的训练与 Test

服务器原脚本目录为 `$PROJECT/gmr_multiroute_kit/`。本分支也在 Moment-DETR
独立文件夹中发布了脚本副本；该副本用 `GMR_PROJECT_ROOT` 指向服务器原项目根：

```bash
# CHECKOUT 为克隆本分支后所在目录的绝对路径
CHECKOUT=/path/to/Improving-Target-Presence-for-Generalized-Moment-Retrieval
MKIT="$CHECKOUT/backbones/moment_detr/experiment/gmr_multiroute_kit"
export GMR_PROJECT_ROOT="$PROJECT"
VARIANT=sepffn

CUDA_VISIBLE_DEVICES=0 python -u "$MKIT/run_train.py" --variant "$VARIANT" --smoke

# 仅适用于尚未初始化的输出目录；现有六组结果已经训练完成
CUDA_VISIBLE_DEVICES=0 python -u "$MKIT/run_train.py" --variant "$VARIANT"

# 复现现有 best 的完整 Test，只需本条
CUDA_VISIBLE_DEVICES=0 python -u "$MKIT/run_eval.py" \
  --variant "$VARIANT" --split test --device cuda
```

训练器发现已经存在的 `experiment_config.json` 会拒绝覆盖；它不支持三骨干 runner
那样的自动断点恢复。当前六组保留原始实验结果。如需重做训练，应使用另一个完整
项目根/输出目录，不能移走当前 best 来伪装成续训。

`launch6.sh` 和 `evaluate6.sh` 是当时每卡三进程的历史批处理脚本。当前限制是每卡
最多两任务，因此本次交接优先使用上面的单任务命令，并由操作者确认空闲槽位。

### 11.5 现有两 GPU 自动队列

```bash
GPU0=0 GPU1=1 bash "$CORE/launch_queue.sh"
cat "$CORE/QUEUE.json"
tail -f "$CORE/logs/queue.log"
python -B "$CORE/collect_results.py"
```

已有同名 tmux 队列时 launcher 会直接退出，避免重复启动。每卡两个 worker；
GPU0 的方法顺序为 `baseline → sepffn → multi`，GPU1 为
`shared → instruction → adaptive`，每种方法依次覆盖 EaTR/QD/CG。阶段顺序为
`smoke → train → test`，成功后自动进入下一项，失败阶段最多重试三次。

六个 batch32 条目在各自正常队列槽位内进行完整 batch32 Smoke；不需要额外开启
GPU 进程。旧目录通过 `execution_plan.py` 归档一次，新任务产生新 last 后的重试
只恢复新 last。**当前运行队列无需重启。**

### 11.6 数据、环境与换机路径配置

当前环境使用 `/home/guoxiangyu/miniconda3/bin/python`。训练依赖 PyTorch、SciPy、
NumPy 与各原始 Backbone 的依赖。代码保留 NumPy2 的 `trapz/trapezoid` 兼容处理。

Moment-DETR 训练器使用：

- 上游源码：`$PROJECT/generalized-moment-retrieval/`；
- 标签：`$PROJECT/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/`；
- 特征：`$PROJECT/mr_experiments/features/{clip,slowfast,clip_text}/`。

EaTR/QD/CG 当前数据接口使用同一套原始 SoccerGMR Standard 资产：

```text
/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/data/label/Standard/{train,val,test}.jsonl
/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr/{slowfast,clip,clip_text}/
```

换机时需要调整原始数据接口中的路径常量：EaTR 的 `backbones/eatr/source/deploy.py`
中的 `SOURCE`，QD 的 `backbones/qd_detr/source/train_gmr.py` 中的 `ASSETS/LABELS`，
CG 的 `backbones/cg_detr/source/train_gmr.py` 中的 `GMR`。这些文件仍导入服务器历史
`/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared/official_eval/`；
换机需要配置这个官方评估器依赖或将其导入路径调整为本分支保存的官方评估器。
只克隆本分支并不会自动准备这些外部资产。

分支的统一 runner 默认使用 `backbones/moment_detr/source/eval/` 中的评估器，
可通过 `GMR_EVAL_DIR` 覆盖其路径。原生 Backbone 模块的上述历史依赖仍应一并配置。

### 11.7 测试结果、图表和 CPU 重算入口

三种迁移骨干每个方法的产物都在自己的实验目录：

```text
runs/<backbone>/<variant>/
├── best.ckpt                    # 验证选择的权重；未完成时可能尚不存在
├── last.ckpt                    # 可恢复训练状态；复用 EaTR Baseline 无此文件
├── experiment_config.json       # 已开始训练后才生成
├── status.json
├── history.jsonl                # 逐 epoch 训练记录；复用模型保留来源记录
├── frozen_checkpoint.json
├── test_predictions.jsonl
└── test_metrics.json
```

批量汇总写入 `$CORE/RESULTS.md` 与 `$CORE/results.csv`。测试完成后自动调用
`GMR_FIGURE_REFRESH` 指向的刷新脚本；服务器默认路径如下，也可手动运行：

```bash
bash "$FIGURES/refresh_all.sh"
```

图表分别进入 `$FIGURES/moment_detr_gmr/`、`eatr_gmr/`、`qd_detr_gmr/`、
`cg_detr_gmr/`；跨骨干图进入 `cross_backbone_overview/`。每个完成骨干输出
`test_evaluation.{pdf,svg,png}`、`figure_data.csv` 和 101 点
`gmiou_threshold_sweep.csv`。G-mIoU 数值及阈值均使用 0–1 表示。未完成条目不会
填零。图表脚本的服务器默认路径不随 GitHub 克隆自动创建。

Moment-DETR 的冻结预测已包含在 GitHub 发布包中。在配置好包内说明的官方评估器
依赖后，不需要权重或 GPU 即可从冻结预测重算指标和图表：

```bash
cd "$CHECKOUT/backbones/moment_detr/experiment"
python -m pip install -r requirements-reproduction.txt
python gmr_multiroute_kit/recompute_frozen_results.py
```

这属于“冻结预测的指标复现”；加载本地检查点重新推理则使用第 11.2/11.4 节。
