# GMR × Mr. DETR Multi-Route Training

本分支专门记录将 Mr. DETR 的 Instructive Multi-Route Training 迁移到
Generalized Moment Retrieval（GMR）的实现与实验。目标是在保持
GMR-Adapter 空集拒绝能力的同时，通过训练期 Temporal One-to-Many（O2M）
监督提高多个不同相关片段的定位与召回。

当前覆盖四个 DETR 类时序定位骨干：

- Moment-DETR-GMR
- EaTR-GMR
- QD-DETR-GMR
- CG-DETR-GMR

所有实验使用固定随机种子 `2023`，不进行多 Seed 重复。主路线仍使用原生
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

三种迁移骨干共用同一份 [`multiroute.py`](backbonefolder/multiroute_gmr/multiroute.py)
和 [`run.py`](backbonefolder/multiroute_gmr/run.py)，各自入口位于
[`backbones/`](backbonefolder/multiroute_gmr/backbones/)。共享核心可以避免三份
Criterion、匹配与早停实现产生漂移。

## 4. 统一训练与测试协议

| 项目 | 设置 |
|---|---|
| Seed | 2023（单 Seed） |
| Epoch 上限 | 400 |
| Optimizer / LR | AdamW / `5e-5` |
| 主 / 辅助 Queries | 10 / 24 |
| Microbatch / 累积 | 2 / 4，有效 Batch 8 |
| DataLoader | 每任务 `num_workers=2`、pin memory、prefetch 2、persistent workers |
| 验证间隔 | 每 5 epochs |
| 选模 | 验证集正样本定位 mAP 最大 |
| 早停 | 当前 epoch 距 best epoch 达到 20 |
| 队列并发 | 两张 GPU，每张最多同时 2 个任务 |
| 测试 | 训练结束后仅测试冻结的 `best.ckpt` |
| G-mIoU | 官方百分数保留，同时显式输出 `brief_0_1` 的 0–1 数值 |

队列阶段为 `smoke → train/resume → frozen-best test`。每个阶段最多重试三次；
已有 `last.ckpt` 时自动恢复，已有完整任务自动跳过。测试完成后可通过
`GMR_FIGURE_REFRESH` 调用外部图表刷新脚本，图表失败只产生警告，不改变已完成
测试的有效性。

## 5. 当前结果与状态

Moment-DETR-GMR 的六组冻结结果、预测、完整 0–1 G-mIoU 扫描和论文图位于
[`moment_detr_gmr_20261030/`](moment_detr_gmr_20261030/)。这些结果来自固定
seed 2023，但六组按当时指令停止，实际训练 epoch 不完全一致，解释时不能把差异
全部归因于结构。

截至 2026-10-31，额外骨干已有以下 frozen-best Test 结果；其余配置仍由队列训练。
除 G-mIoU 外，表中指标沿用官方百分数；G-mIoU@3 使用 0–1 表示。

| Backbone | 配置 | mAP | mR+@5 | AUROC | Rej-F1@0.4 | G-mIoU@3 |
|---|---|---:|---:|---:|---:|---:|
| EaTR | baseline | 6.86 | 0.87 | 69.99 | 75.00 | 0.4900 |
| EaTR | shared | 16.65 | 8.60 | 77.57 | 61.23 | 0.3139 |
| EaTR | sepffn | 17.06 | 9.21 | 77.75 | 54.93 | 0.2655 |
| QD-DETR | baseline | 15.08 | 9.62 | 77.66 | 59.84 | 0.3035 |
| QD-DETR | shared | 15.24 | 9.78 | 77.13 | 63.37 | 0.3334 |
| QD-DETR | sepffn | 11.03 | 4.16 | 75.57 | 68.25 | 0.3783 |

这些是单 Seed 探索性结果。定位指标、额外 GT 覆盖、空集拒绝和 G-mIoU 之间可能
存在权衡，不能只依据单个指标宣称某种路线全面更优。

## 6. 目录结构

```text
.
├── README.md
├── LICENSE
├── moment_detr_gmr_20261030/       # 已完成的 Moment-DETR 六路线结果包
└── backbonefolder/
    ├── moment_detr/                 # Moment-DETR-GMR 原生/GMR 源码
    ├── eatr/                        # EaTR-GMR 源码与数据接口
    ├── qd_detr/                     # QD-DETR-GMR 源码与数据接口
    ├── cg_detr/                     # CG-DETR-GMR 源码与数据接口
    └── multiroute_gmr/
        ├── multiroute.py            # 多路线模型、O2M Matcher 与 Criterion
        ├── run.py                   # 统一 smoke/train/test runner
        ├── queue.py                 # 两 GPU、每卡最多两任务的可恢复队列
        ├── launch_queue.sh           # tmux 队列入口
        ├── validate_all.py           # 18 配置真实数据 smoke gate
        ├── collect_results.py        # 汇总 frozen-best Test 指标
        └── backbones/
            ├── moment_detr_gmr/
            ├── eatr_gmr/
            ├── qd_detr_gmr/
            └── cg_detr_gmr/
```

## 7. 使用方法

### 单个迁移骨干

```bash
# BACKBONE 可替换为 eatr_gmr、qd_detr_gmr 或 cg_detr_gmr
BACKBONE=eatr_gmr
VARIANT=sepffn

CUDA_VISIBLE_DEVICES=0 \
python backbonefolder/multiroute_gmr/backbones/$BACKBONE/run.py \
  --variant "$VARIANT" --mode smoke

CUDA_VISIBLE_DEVICES=0 \
python backbonefolder/multiroute_gmr/backbones/$BACKBONE/run.py \
  --variant "$VARIANT" --mode train

CUDA_VISIBLE_DEVICES=0 \
python backbonefolder/multiroute_gmr/backbones/$BACKBONE/run.py \
  --variant "$VARIANT" --mode test
```

也可以使用每个 Backbone 目录内的 `smoke.sh`、`train.sh` 和 `test.sh`。

### 六配置 × 三迁移骨干队列

```bash
# 首次执行或 smoke 发生变化时检查全部 18 个配置
python backbonefolder/multiroute_gmr/validate_all.py

# GPU 0 和 GPU 1 各最多两个任务
GPU0=0 GPU1=1 bash backbonefolder/multiroute_gmr/launch_queue.sh

# 查看状态与日志
cat backbonefolder/multiroute_gmr/QUEUE.json
tail -f backbonefolder/multiroute_gmr/logs/queue.log

# 汇总已完成 Test
python backbonefolder/multiroute_gmr/collect_results.py
```

### Moment-DETR-GMR 原始六路线

完整说明见 [`moment_detr_gmr_20261030/README.md`](moment_detr_gmr_20261030/README.md)。
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
