# Moment-DETR-GMR

- [`source/`](source/)：原生 GMR 模型、数据与评估器代码。
- [`experiment/`](experiment/)：完整六路线结果包、唯一一份实验工具包、冻结 Test 预测与图表。
- `smoke.sh`、`train.sh`、`test.sh`：该骨干的统一 Shell 入口。

从仓库根目录执行，方法可替换为 baseline/shared/sepffn/instruction/multi/adaptive：

```bash
export GMR_PROJECT_ROOT=/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval
CUDA_VISIBLE_DEVICES=0 bash backbones/moment_detr/smoke.sh sepffn
CUDA_VISIBLE_DEVICES=0 bash backbones/moment_detr/train.sh sepffn
CUDA_VISIBLE_DEVICES=0 bash backbones/moment_detr/test.sh sepffn
```

原实验六组都已完成，只复现 Test 时运行第三条。训练器拒绝覆盖已初始化的结果，
没有 last.ckpt 自动续训功能。默认工作根为 `experiment/`；设置上面的环境变量后
读取服务器的原项目、特征和 `mr_experiments/<方法>/best.ckpt`。

CPU 冻结预测重算见 [`experiment/README.md`](experiment/README.md)。
全部服务器权重路径见 [检查点索引](../../docs/CHECKPOINTS_AND_REPRODUCTION.md#10-全部检查点与实验目录)。
