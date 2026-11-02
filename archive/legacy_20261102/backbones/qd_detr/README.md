# QD-DETR-GMR

[完整结果表](../../results/README.md) · [本骨干指标与全域曲线](../../results/qd_detr_gmr/README.md)

`source/` 保存原始骨干代码与来源；本目录的 `run.py` 调用唯一的公共
[`shared/run.py`](../../shared/run.py)。

```bash
CUDA_VISIBLE_DEVICES=0 bash backbones/qd_detr/smoke.sh sepffn
CUDA_VISIBLE_DEVICES=0 bash backbones/qd_detr/train.sh sepffn
CUDA_VISIBLE_DEVICES=0 bash backbones/qd_detr/test.sh sepffn
```

在仓库根目录执行。默认权重在 `shared/runs/qd_detr/<方法>/best.ckpt`；
`GMR_RUNTIME_ROOT` 可指向服务器原有的完整运行目录。

baseline/shared/sepffn 已完成。排队的 instruction/multi/adaptive 按
`shared/execution_plan.json` 使用 batch32、accumulation1 从头训练，旧运行目录
已归档。新任务须在正常队列槽位内通过完整 batch32 Smoke。

[检查点与测试命令](../../docs/CHECKPOINTS_AND_REPRODUCTION.md#10-全部检查点与实验目录) ·
[交接与微调](../../HANDOFF.md)
