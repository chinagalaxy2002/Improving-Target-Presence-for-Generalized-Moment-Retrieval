# CG-DETR-GMR

`source/` 保存原始骨干代码与来源；本目录的 `run.py` 调用唯一的公共
[`shared/run.py`](../../shared/run.py)。公共实现包含 worker-safe CPU collate
和主进程 CUDA 搬运修复。

```bash
CUDA_VISIBLE_DEVICES=0 bash backbones/cg_detr/smoke.sh sepffn
CUDA_VISIBLE_DEVICES=0 bash backbones/cg_detr/train.sh sepffn
CUDA_VISIBLE_DEVICES=0 bash backbones/cg_detr/test.sh sepffn
```

在仓库根目录执行。默认权重在 `shared/runs/cg_detr/<方法>/best.ckpt`；
`GMR_RUNTIME_ROOT` 可指向服务器原有的完整运行目录。

活动 baseline/shared/sepffn 保持 microbatch2/accumulation4；排队的
instruction/multi/adaptive 从头使用 batch32/accumulation1，旧断点已归档。

[检查点与测试命令](../../docs/CHECKPOINTS_AND_REPRODUCTION.md#10-全部检查点与实验目录) ·
[交接与微调](../../HANDOFF.md)
