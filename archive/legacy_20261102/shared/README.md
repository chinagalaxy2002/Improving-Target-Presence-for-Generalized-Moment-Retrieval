# 公共实现

三种迁移骨干共用本目录的一份模型、损失、训练和评估实现；每个骨干的可调用入口
放在 [`../backbones/`](../backbones/) 中。

| 文件 | 职责 |
|---|---|
| `multiroute.py` | 多路线模型、独立 FFN、Instruction 与 Temporal O2M 损失 |
| `run.py` | `--backbone eatr|qd_detr|cg_detr --variant <方法> --mode smoke|train|test` |
| `execution_plan.json` | 各方法当前 Batch、复用或从头训练规则 |
| `execution_plan.py` | 读取规则，保护运行任务，归档被重置的旧目录 |
| `queue.py` / `launch_queue.sh` | 两 GPU、每卡最多两个任务的可恢复队列 |
| `validate_all.py` | 真实数据检查与复用权重检查 |
| `collect_results.py` | 汇总完成的 Test 指标 |
| `archive_completed_baselines.py` | 复制服务器原有四骨干 Baseline |
| `reuse_eatr_baseline.py` | 安装已完成 EaTR Baseline 的包装格式权重 |

默认运行产物保存在本目录的 `runs/`、`logs/`、`reused_baselines/` 和
`archived_runs/`，均不随代码上传。`GMR_RUNTIME_ROOT` 可以指定另一完整运行根目录：

```bash
export GMR_RUNTIME_ROOT=/absolute/path/to/runtime_root
CUDA_VISIBLE_DEVICES=0 bash backbones/qd_detr/test.sh sepffn
```

从仓库根目录启动队列：

```bash
GPU0=0 GPU1=1 bash shared/launch_queue.sh
```

运行根目录的 `execution_plan.json` 优先于代码目录内的计划。
服务器原有训练目录没有迁移；要使用已有权重，把 `GMR_RUNTIME_ROOT` 设置为
`/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr`。

检查点与复现：[详细索引](../docs/CHECKPOINTS_AND_REPRODUCTION.md#10-全部检查点与实验目录)。
后续微调：[HANDOFF.md](../HANDOFF.md)。
