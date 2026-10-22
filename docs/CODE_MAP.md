# 五个主干：代码、配置与实验结果映射

下列路径相对于仓库根目录。CPU 图表复现统一使用 `tools/reproduce_results.py`；表中训练入口是**实验实际使用的历史代码快照**，重训前需要调整本机路径。旧子目录 README 或日志中的 TRAINING / WAITING 是当时的记录；本次发布的 14 组本地实验均已有完整测试预测。

## 1. Baseline 与 GMR

| 主干 | 模型/训练实现 | 实际配置与测试输出 |
|---|---|---|
| Moment-DETR | [模型](../backbonefolder/moment_detr/models/moment_detr_gmr/)、[训练与推理](../backbonefolder/moment_detr/training/moment_detr_gmr/)、[配置](../backbonefolder/moment_detr/configs/) | [Baseline](../backbonefolder/moment_detr/original/)、[GMR](../backbonefolder/moment_detr/gmr/)；各目录的 `effective_config.json`、`frozen_selection.json` 与 `test_predictions.jsonl` |
| QD-DETR | [Baseline 训练](../backbonefolder/qd_detr/train_soccer.py)、[GMR 训练](../backbonefolder/qd_detr/train_gmr.py)、[原生模型](../backbonefolder/qd_detr/deployment/qd_detr/)、[GMR 模型](../backbonefolder/qd_detr/gmr_deployment/qd_detr/) | [Baseline](../backbonefolder/qd_detr/original/)、[GMR](../backbonefolder/qd_detr/gmr/)；有效配置与 `test_predictions.jsonl` |
| CG-DETR | [Baseline 训练](../backbonefolder/cg_detr/train_soccer.py)、[GMR 训练](../backbonefolder/cg_detr/train_gmr.py)、[原生源码](../backbonefolder/cg_detr/source/)、[GMR 源码](../backbonefolder/cg_detr/gmr_source/) | [Baseline](../backbonefolder/cg_detr/original/)、[GMR](../backbonefolder/cg_detr/gmr/)；有效配置与 `final_test_predictions.jsonl` / `final_test_metrics.json` |
| EaTR | [训练](../backbonefolder/eatr/train.py)、[模型](../backbonefolder/eatr/models/)、[数据](../backbonefolder/eatr/start_end_dataset.py)、[推理](../backbonefolder/eatr/inference.py) | [Baseline](../backbonefolder/eatr/original/)、[GMR](../backbonefolder/eatr/gmr/)；有效配置与 `test_predictions.jsonl` |
| FlashVTG | [E0/S4 双模型 API](../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/code/api.py)、[训练](../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/code/train.py)、[GMR 基础模型](../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/baseline_source/) | [E0/GMR 预测](../backbonefolder/flashvtg/gmr/)、[原始配置](../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/configs/)、[来源](../backbonefolder/flashvtg/gmr/SOURCE_PROVENANCE.json)；没有无 GMR 的 FlashVTG 行 |

Baseline/GMR 采用各自冻结的验证集选模检查点。不得用其他主干或其他实验的权重替换。QD-DETR-GMR、CG-DETR-GMR 是本地适配扩展；相关说明和损失定义保留在训练/模型文件中。

## 2. GMR + S4_minus_Fusion

四个 DETR 主干共用 [shared/train.py](../Phase2(addbackbone_with_S4_minusFusion)/shared/train.py)，每个主干有独立 backend 与模型；共同超参数在 [shared/config.json](../Phase2(addbackbone_with_S4_minusFusion)/shared/config.json)。发布表格均固定使用 epoch 4。

| 主干 | S4 模型 | 数据/初始化与接口 | 实际配置和输出 |
|---|---|---|---|
| Moment-DETR | [model.py](../Phase2(addbackbone_with_S4_minusFusion)/moment_detr/model.py) | [backend.py](../Phase2(addbackbone_with_S4_minusFusion)/moment_detr/backend.py)、[公共模块](../Phase2(addbackbone_with_S4_minusFusion)/moment_detr/s4_reference.py) | [CONFIG.json](../Phase2(addbackbone_with_S4_minusFusion)/moment_detr/configuration/CONFIG.json)、[epoch 4 测试](../Phase2(addbackbone_with_S4_minusFusion)/moment_detr/epoch_004/test_results/) |
| QD-DETR | [model.py](../Phase2(addbackbone_with_S4_minusFusion)/qd_detr/model.py) | [backend.py](../Phase2(addbackbone_with_S4_minusFusion)/qd_detr/backend.py)、[data.py](../Phase2(addbackbone_with_S4_minusFusion)/qd_detr/data.py) | [CONFIG.json](../Phase2(addbackbone_with_S4_minusFusion)/qd_detr/configuration/CONFIG.json)、[epoch 4 测试](../Phase2(addbackbone_with_S4_minusFusion)/qd_detr/epoch_004/test_results/) |
| CG-DETR | [model.py](../Phase2(addbackbone_with_S4_minusFusion)/cg_detr/model.py) | [backend.py](../Phase2(addbackbone_with_S4_minusFusion)/cg_detr/backend.py)、[公共模块](../Phase2(addbackbone_with_S4_minusFusion)/cg_detr/reference_modules.py) | [CONFIG.json](../Phase2(addbackbone_with_S4_minusFusion)/cg_detr/configuration/CONFIG.json)、[epoch 4 测试](../Phase2(addbackbone_with_S4_minusFusion)/cg_detr/epoch_004/test_results/) |
| EaTR | [model.py](../Phase2(addbackbone_with_S4_minusFusion)/eatr/model.py) | [backend.py](../Phase2(addbackbone_with_S4_minusFusion)/eatr/backend.py)、[runtime.py](../Phase2(addbackbone_with_S4_minusFusion)/eatr/runtime.py)、[data.py](../Phase2(addbackbone_with_S4_minusFusion)/eatr/data.py) | [CONFIG.json](../Phase2(addbackbone_with_S4_minusFusion)/eatr/configuration/CONFIG.json)、[epoch 4 测试](../Phase2(addbackbone_with_S4_minusFusion)/eatr/epoch_004/test_results/) |
| FlashVTG | [code/model.py](../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/code/model.py) | [API](../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/code/api.py)、[初始化](../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/code/initialization.py)、[runtime](../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/code/runtime.py) | [实际配置](../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/configuration/CONFIG.json)、[epoch 4 测试](../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/epoch_004/test_results/) |

四个新主干是六轮微调的固定 epoch 4；FlashVTG 是历史无 QCD 实验的真实 epoch 4。各主干 `ARCHIVE_INDEX.json` 保存检查点身份和相对路径，`configuration/` 保存初始化/预检记录。源码中的 QCD 等辅助入口作为历史来源保留，但本次 14 行本地实验汇总不使用 QCD。

## 3. 重训前的路径准备

1. 准备原始 Soccer-GMR 特征：`slowfast/`、`clip/`、`clip_text/`。标签位于 [shared/labels](../Phase2(addbackbone_with_S4_minusFusion)/shared/labels/)。
2. 读取对应实验的 `effective_config.json` 或 `configuration/CONFIG.json`；保持特征顺序、维度、全部空 GT 与后处理协议。
3. 对应原生训练脚本中的 `ASSETS`、`LABELS`、`SHARED` 和输出目录需改为本机位置。S4 各 backend/runtime 的 `ROOT`、`SOURCE`、`DEFAULT_SOURCE` 及缓存 snapshot 路径也需按实际资产配置。
4. S4 初始化所需的 GMR 检查点及哈希见 [EXTERNAL_CHECKPOINTS.json](../provenance/EXTERNAL_CHECKPOINTS.json)；GitHub 不含二进制权重。历史 `snapshots/*.json` 是来源记录，不能当成已存在的本机权重。
5. FlashVTG 的便携环境支持 `GMR_ASSETS_DIR`、`GMR_E0_CHECKPOINT`、`GMR_S4F_CHECKPOINT`，详见 [environment.py](../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/code/environment.py)。

训练依赖、GPU、数据特征和检查点只在新训练/推理时需要。重建三张表使用 `python tools/reproduce_results.py`，不需要这些资产。原始归档中依赖 checkpoint 的 `verify_archive.py` 与严格权重核验入口不适用于移除权重后的发布子集；实际上传文件清单为根目录 `PUBLICATION_MANIFEST.json`。

## 4. CPU 评测与表格数据流

```text
14 组冻结测试预测 + shared/labels/test.jsonl
            │
            ├── 官方 evaluator：AUROC / 正查询定位指标
            └── τ = 0.4 / 0.5 / 0.6：Rej-F1 / G-mIoU@1 / G-mIoU@3
                            │
                  同主干 S4 − GMR，单位 pp
                            │
            + 论文 Table 4 的 6 行参考数据（原始 τ=0.4）
                            │
                 三张 20 行完整结果表 + 对应图片
```

`tools/reproduce_results.py --recompute-metrics` 从预测重新评分；默认入口使用已保存的阈值网格。原始历史 JSON 内的绝对路径仅溯源；已适配的缓存 `source_predictions` 使用仓库相对路径，同时保留 `historical_source_predictions` 和原预测 SHA256。
