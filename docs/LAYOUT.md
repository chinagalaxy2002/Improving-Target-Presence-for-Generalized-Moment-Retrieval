# GitHub 目录迁移对照

本次只整理 GitHub 分支源码布局，训练服务器上的目录和检查点位置未变。

| 原 GitHub 路径 | 新 GitHub 路径 |
|---|---|
| `backbonefolder/moment_detr/` | `backbones/moment_detr/source/` |
| `backbonefolder/eatr/` | `backbones/eatr/source/` |
| `backbonefolder/qd_detr/` | `backbones/qd_detr/source/` |
| `backbonefolder/cg_detr/` | `backbones/cg_detr/source/` |
| `backbonefolder/multiroute_gmr/` 的公共代码 | `shared/` |
| `backbonefolder/multiroute_gmr/backbones/*_gmr/` | `backbones/<骨干>/` 的顶层入口 |
| `moment_detr_gmr_20261030/` | `backbones/moment_detr/experiment/` |
| 重复的 Moment-DETR kit | 合并到 `backbones/moment_detr/experiment/gmr_multiroute_kit/` 一份 |
| 旧目录内的 `HANDOFF.md` | 仓库根目录 `HANDOFF.md` |
| 原首页完整检查点长表 | `docs/CHECKPOINTS_AND_REPRODUCTION.md` |
| `reused_baselines/` 的来源文档 | `docs/baseline_sources/` |

新入口统一为 `bash backbones/<骨干>/smoke.sh|train.sh|test.sh <方法>`。
三种迁移骨干通过 `GMR_RUNTIME_ROOT` 读取服务器原运行根目录；Moment-DETR
通过 `GMR_PROJECT_ROOT` 读取服务器原项目。无需搬权重来匹配 GitHub 文件夹。

公共模型和损失计算未改动。为适配重排后的导入路径，GMR evaluator 增加包内导入，
避免与原生骨干的 `utils` 重名；独立 CLI 仍可使用。1036条冻结预测的19个指标与
原评估器一致。Moment-DETR 冻结结果包的校验和已更新，原冻结预测内容未改变。
