# 已完成 GMR Baseline 检查点

四个骨干的历史完整基线已归档到当前实验目录。每个子目录保留原始 checkpoint、
训练配置、冻结记录、验证指标以及完整 Test 指标和预测。`manifest.json` 记录原始
路径、SHA256 与使用范围。

| Backbone | 检查点 | 历史 Test mAP |
|---|---|---:|
| Moment-DETR | `moment_detr/best.ckpt` | 6.68 |
| EaTR | `eatr/best.ckpt` | 16.20 |
| QD-DETR | `qd_detr/best.ckpt` | 14.62 |
| CG-DETR | `cg_detr/best.pt` | 18.65 |

原始检查点的模型参数键和张量形状均与当前对应骨干兼容。当前 Multi-Route 包装
增加了 `base.` 前缀；EaTR 的 Slot Attention 额外增加 `slot_atten.shared.` 层级。
使用当前包装模型进行推理时，需要映射这些名字，不能直接把原始 checkpoint 当成
当前队列的 `last.ckpt` 恢复优化器。

这些检查点可以直接作为历史已训练基线使用，无需重复训练它们。但历史模型使用
seed 2026、学习率 3e-5，EaTR/QD/CG 使用 batch 32；当前路线实验使用 seed 2023、
学习率 5e-5、microbatch 2 × accumulation 4。因此它们不属于同训练协议的严格消融
基线。报告中应同时保留原始配置，不能将两套结果混成一个 seed2023 的实验。

首次归档时没有尚未启动的 Baseline 任务。随后按用户指令，EaTR Baseline 已复用
本目录中的原始权重和冻结预测，并安装到 `../runs/eatr/baseline/`；旧的 seed2023
低性能 Baseline 已移到 `../archived_runs/original_batch_sizes_20261031/eatr/baseline/`。
只映射了包装模型的参数名称，未改变原始权重张量。EaTR 其余路线继续原队列。

Moment-DETR/QD-DETR 已完成的 Baseline 和正在运行的 CG-DETR Baseline 保持原状。
待启动 QD/CG 的 instruction、multi、adaptive 改为 batch32 从头训练，详见
`../execution_plan.json`。

重新核验或补齐本地归档：

```bash
python -B backbonefolder/multiroute_gmr/archive_completed_baselines.py
```

脚本只复制缺失文件；若目标文件与来源不同则报错，避免覆盖归档。
