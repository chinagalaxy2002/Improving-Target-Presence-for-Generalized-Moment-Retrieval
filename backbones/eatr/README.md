# EaTR-GMR

`source/` 保存原始骨干代码与来源；本目录的 `run.py` 调用唯一的公共
[`shared/run.py`](../../shared/run.py)。

```bash
CUDA_VISIBLE_DEVICES=0 bash backbones/eatr/smoke.sh shared
CUDA_VISIBLE_DEVICES=0 bash backbones/eatr/train.sh shared
CUDA_VISIBLE_DEVICES=0 bash backbones/eatr/test.sh shared
```

在仓库根目录执行。默认权重在 `shared/runs/eatr/<方法>/best.ckpt`；
`GMR_RUNTIME_ROOT` 可以指向服务器原有的完整运行目录。

Baseline 复用 seed2026、batch32、LR3e-5 的已完成 GMR，Test mAP16.20；
其他 EaTR 路线继续 seed2023、microbatch2/accumulation4 的原队列。
Baseline smoke/train 识别复用标记后跳过新训练；`test.sh baseline` 可重新推理。

[检查点与测试命令](../../docs/CHECKPOINTS_AND_REPRODUCTION.md#10-全部检查点与实验目录) ·
[交接与微调](../../HANDOFF.md)
