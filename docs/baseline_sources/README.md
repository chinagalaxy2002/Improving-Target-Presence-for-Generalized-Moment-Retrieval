# 历史 Baseline 来源

[`manifest.json`](manifest.json) 记录服务器上四份已归档 GMR 权重的原始路径、
SHA256、Batch、Seed 和历史 Test mAP。清单中的 `reused_baselines/...` 相对
服务器旧运行根目录，属于权重来源记录，不表示 GitHub 存有这些二进制文件。

| Backbone | 原 bsz | Seed / LR | Test mAP |
|---|---:|---|---:|
| Moment-DETR | 8 | 2026 / 3e-5 | 6.68 |
| EaTR | 32 | 2026 / 3e-5 | 16.20 |
| QD-DETR | 32 | 2026 / 3e-5 | 14.62 |
| CG-DETR | 32 | 2026 / 3e-5 | 18.65 |

只有 EaTR 已安装成当前实验 Baseline；其他历史权重是参考。
当前检查点完整路径和使用方式见 [检查点索引](../CHECKPOINTS_AND_REPRODUCTION.md#10-全部检查点与实验目录)。
