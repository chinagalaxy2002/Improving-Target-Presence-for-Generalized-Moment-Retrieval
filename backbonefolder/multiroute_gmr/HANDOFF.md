# GMR × Mr. DETR 多骨干实验交接

更新时间：2026-10-31。本文只交接 Baseline、六种方法、检查点测试和后续微调。
状态为交接时快照，继续工作前先读取各任务 `status.json` 并核对实际进程。

GitHub 分支：
[`gmr-mr-detr-multiroute-20261030`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/tree/gmr-mr-detr-multiroute-20261030)。
分支根 README 第 10–11 节列出全部 24 组检查点绝对路径和详细复现命令。

## 1. 环境与目录

```bash
PROJECT=/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval
REPO=/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval
CORE="$REPO/backbonefolder/multiroute_gmr"
RUNS="$CORE/runs"
REUSED="$CORE/reused_baselines"
ARCHIVE="$CORE/archived_runs/original_batch_sizes_20261031"
FIGURES="$PROJECT/newidea/figure/gmr_mr_detr_test_20261030"
PYTHON=/home/guoxiangyu/miniconda3/bin/python
```

Moment-DETR 实际运行脚本在 `$PROJECT/gmr_multiroute_kit/`；EaTR/QD/CG 共用
`$CORE/run.py`、`multiroute.py`。GitHub 中另有 `backbones/*_gmr/` 薄入口，服务器
当前工作目录不一定有这些薄入口，直接调用统一 runner 最明确。

原始标签和特征供 EaTR/QD/CG 使用：

```text
/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/data/label/Standard/
/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr/{slowfast,clip,clip_text}/
```

Moment-DETR 使用 `$PROJECT/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/`
和 `$PROJECT/mr_experiments/features/{clip,slowfast,clip_text}/`。保持现有特征归一化、
拼接顺序和时间尺度；Moment-DETR 为 CLIP→SlowFast→TEF，其余三骨干为
SlowFast→CLIP→TEF。GitHub 不包含权重、原始视频和 `.npz` 特征。

## 2. 四个骨干的 Baseline

这里的 Baseline 指保留存在性适配器的 **GMR**，不是不带 GMR 的原生定位器。

| 骨干 | 历史完整 GMR 权重（相对 `$REUSED`） | 原 bsz | Seed / LR | Test mAP |
|---|---|---:|---|---:|
| Moment-DETR | `moment_detr/best.ckpt` | 8 | 2026 / 3e-5 | 6.68 |
| EaTR | `eatr/best.ckpt` | 32 | 2026 / 3e-5 | 16.20 |
| QD-DETR | `qd_detr/best.ckpt` | 32 | 2026 / 3e-5 | 14.62 |
| CG-DETR | `cg_detr/best.pt` | 32 | 2026 / 3e-5 | 18.65 |

四份检查点均已复制、核对冻结 SHA256，并保留原训练配置和 Test 预测。来源、哈希
与完整原始路径在 `$REUSED/manifest.json`；更早的源目录为：

```text
/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/<骨干>/gmr/
```

**仅 EaTR Baseline 已复用到当前实验**：

```text
$RUNS/eatr/baseline/best.ckpt
$RUNS/eatr/baseline/reused_baseline.json
$RUNS/eatr/baseline/source_effective_config.json
```

其 238 个原始模型张量未改变，只映射了包装模型的参数名；这是推理用 checkpoint，
没有可恢复原训练的 `last.ckpt`。原 seed2023、mAP6.86 的重训结果在
`$ARCHIVE/eatr/baseline/`，不再用作当前 EaTR Baseline。

Moment-DETR/QD/CG 当前六路线中的 Baseline 仍是本轮 seed2023 训练的结果，未被
历史权重替换。EaTR 历史 Baseline 与新路线不是同 seed、同 batch 的严格消融比较。

如果需要从头重训历史 Baseline，应在**新的源码副本和输出目录**运行对应原始入口，
并保留其 `effective_config.json` 参数：

| 骨干 | 原始 GMR 训练入口 |
|---|---|
| Moment-DETR | `backbonefolder/moment_detr/deploy.py --mode smoke --variant gmr`，然后 `--mode train --variant gmr` |
| EaTR | `backbonefolder/eatr/deploy.py --mode smoke --variant gmr`，然后 `--mode train --variant gmr` |
| QD-DETR | `backbonefolder/qd_detr/train_gmr.py --smoke-then-train` |
| CG-DETR | `backbonefolder/cg_detr/train_gmr.py --smoke-then-train` |

这些入口原输出为对应骨干的 `gmr/`，没有统一 `--output-dir` 参数；重训前需在新副本
确认 `OUT` 指向新目录。当前 EaTR Baseline 按用户要求直接复用，无需重训。

## 3. 六种方法与当前队列

| 方法参数 | 含义 |
|---|---|
| `baseline` | 原 GMR O2O 主路线，不增加辅助路线 |
| `shared` | 一条固定 K=2 O2M 路线，Decoder 全共享 |
| `sepffn` | O2M 路线共享注意力，FFN 和 norm3 独立 |
| `instruction` | O2M 路线在 Self-Attention 加 4 个学习型指令 Token |
| `multi` | Separate FFN 与 Instruction 双路线，各权重 0.15 |
| `adaptive` | 双路线，第二正样本要求 tIoU≥0.25 |

主 Queries=10，辅助 Queries=24，辅助损失总权重=0.3。主路线与 GMR-Adapter 参与
推理，辅助路线仅训练执行。新路线使用 seed2023、AdamW、LR5e-5；每 5 epoch 验证，
验证 mAP 选择 best，20 epoch 无 best 提升早停，最多 400 epoch。Moment-DETR
原六组已按当时指令停止，实际轮次不同，详见其发布结果包。

当前配置由 `$CORE/execution_plan.json` 决定：

- 正在运行：CG `baseline/shared/sepffn`、EaTR `instruction`，保留 `bsz2×累积4`。
- EaTR `shared/sepffn` 已测试完成；`multi/adaptive` 排队恢复原断点，仍为 `2×4`。
- QD `baseline/shared/sepffn` 已测试完成。
- QD/CG 各自的 `instruction/multi/adaptive` 共六项排队从头训练，使用 `32×1`。
  旧运行目录在 `$ARCHIVE/<骨干>/<方法>/`，不能拿旧断点恢复新 batch32 训练。

GPU0 的方法顺序为 baseline→sepffn→multi；GPU1 为 shared→instruction→adaptive。
每张卡最多同时两个任务，**不要重启或重复开启当前活动训练**。

```bash
cat "$CORE/QUEUE.json"
tail -f "$CORE/logs/queue.log"

# 已有同名 tmux 会直接返回；队列终止后才用于恢复调度
GPU0=0 GPU1=1 bash "$CORE/launch_queue.sh"
```

## 4. 找检查点并复现 Test

检查点按“骨干 + 方法”确定，六种方法参数与目录名完全一致：

| 骨干 | 当前 best 的完整路径规则 |
|---|---|
| Moment-DETR | `/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/mr_experiments/<方法>/best.ckpt` |
| EaTR | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/eatr/<方法>/best.ckpt` |
| QD-DETR | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/qd_detr/<方法>/best.ckpt` |
| CG-DETR | `/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/cg_detr/<方法>/best.ckpt` |

交接时 Moment 六组 best 均存在；EaTR adaptive 的 best 尚不存在，其余 EaTR best
已存在；QD/CG 后三组的 best/last 均尚未产生，其余 best 已存在。尚未产生的文件
就是上表规则对应的预期路径。训练中 best 会更新，不等于已经冻结完成 Test。

三种迁移骨干以同目录 `last.ckpt` 续训；Moment 原训练器不保存 last、也不支持自动
续训。文件实际状态可直接核对：

```bash
find "$RUNS" -maxdepth 3 -name best.ckpt
cat "$RUNS/eatr/instruction/status.json"
```

对 EaTR/QD/CG 任意方法执行 Smoke、训练/恢复和 Test：

```bash
BACKBONE=qd_detr         # eatr / qd_detr / cg_detr
VARIANT=sepffn          # 六种方法参数之一

CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$CORE/run.py" \
  --backbone "$BACKBONE" --variant "$VARIANT" --mode smoke
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$CORE/run.py" \
  --backbone "$BACKBONE" --variant "$VARIANT" --mode train
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$CORE/run.py" \
  --backbone "$BACKBONE" --variant "$VARIANT" --mode test
```

**只测试现有权重时仅运行第三条。** runner 固定读取该方法目录的 best，没有
`--checkpoint` 任意路径参数；原生历史权重需要原生模型加载或先转换包装名称，
不能直接替换当前 best/last。EaTR Baseline 已完成转换，可直接 `--backbone eatr
--variant baseline --mode test`；其 smoke/train 会识别复用标记并跳过训练。

Moment-DETR 现有权重的测试入口：

```bash
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "$PROJECT/gmr_multiroute_kit/run_eval.py" \
  --variant sepffn --split test --device cuda
```

Moment 从头训练用同目录 `run_train.py --variant <方法>`，Smoke 再加 `--smoke`。
它会拒绝覆盖已经初始化的实验目录；重做需使用新的完整项目副本。历史 launch6
脚本每卡三进程，不适用于当前每卡最多两任务的并发约束。

EaTR/QD/CG Test 结果写入对应目录的 `test_predictions.jsonl`、`test_metrics.json`；
Moment 写入对应 `test/` 下的 `submission.jsonl`、`metrics_gmiou_*.json`。
G-mIoU 报告使用 0–1，其他官方指标为百分数。统一汇总和图表：

```bash
"$PYTHON" -B "$CORE/collect_results.py"
bash "$FIGURES/refresh_all.sh"
```

每个骨干的图在 `$FIGURES/<moment_detr_gmr|eatr_gmr|qd_detr_gmr|cg_detr_gmr>/`。

## 5. 后续微调如何衔接

当前 `run.py --mode train` 是原实验的训练/断点恢复，**没有独立的 fine-tune 或
init-checkpoint 参数**。不要把 best 复制为 last，也不要把新方法放到已有输出目录。

新微调实验应使用独立目录，例如 `$CORE/finetune_runs/<骨干>/<新方法>/`，并增加独立
入口或在隔离副本中改写初始化。操作顺序：

1. 选定源：原生 GMR 用 `$REUSED/<骨干>/best.*`；现有多路线方法用 `$RUNS/<骨干>/<方法>/best.ckpt`。
2. 构建对应模型，只加载 `checkpoint['model']`，先核对参数名和形状，再严格加载。
3. 新建 optimizer/scheduler，epoch、best、早停计数从头开始，记录源权重路径与 SHA256。
4. 如果新方法新增模块，只迁移可兼容的旧参数；明确记录 missing/unexpected keys，
   新模块随机初始化。不要用 `strict=False` 掩盖未解释的关键参数缺失。
5. 保持原数据 split、特征顺序、主 Queries 和推理定义。超参数按微调设计单独记录；
   如果研究模块增益，各对照须从同一个初始化出发。训练选模用 Val，冻结后再测 Test。

只有恢复同一轮未完成训练才加载 optimizer、scheduler 和 RNG，并继续旧 epoch。
EaTR 复用 Baseline 的转换文件是 inference-only；它可以提供微调的模型权重，但不能
作为恢复原优化器状态的 checkpoint。

微调前先确认可用 GPU 槽位；当前活动队列每卡两个任务已占满并发额度。本交接仅
记录后续做法，没有新启动微调，也没有中断当前训练。
