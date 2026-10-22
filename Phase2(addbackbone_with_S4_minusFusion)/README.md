# S4_minus_Fusion：五个主干的 epoch 4 代码与结果

这是原始完整本地归档的 GitHub 代码/结果子集。保留五个主干的模型源码、有效配置、初始化记录、完整测试预测、指标、运行日志和图表；二进制权重与预提取特征未上传。

- [仓库主 README：三个阈值的完整表格](../README.md)
- [训练与模型代码映射](../docs/CODE_MAP.md)
- [发布文件清单](../PUBLICATION_MANIFEST.json)
- [外部检查点身份与 SHA256](../provenance/EXTERNAL_CHECKPOINTS.json)

## 主干目录

| 主干 | 目录 | 冻结预测 | 实际配置 |
|---|---|---|---|
| moment_detr | [moment_detr/](moment_detr/) | [epoch 4](moment_detr/epoch_004/test_results/test_predictions.jsonl) | [CONFIG.json](moment_detr/configuration/CONFIG.json) |
| qd_detr | [qd_detr/](qd_detr/) | [epoch 4](qd_detr/epoch_004/test_results/test_predictions.jsonl) | [CONFIG.json](qd_detr/configuration/CONFIG.json) |
| cg_detr | [cg_detr/](cg_detr/) | [epoch 4](cg_detr/epoch_004/test_results/test_predictions.jsonl) | [CONFIG.json](cg_detr/configuration/CONFIG.json) |
| eatr | [eatr/](eatr/) | [epoch 4](eatr/epoch_004/test_results/test_predictions.jsonl) | [CONFIG.json](eatr/configuration/CONFIG.json) |
| flashvtg | [flashvtg/](flashvtg/) | [epoch 4](flashvtg/epoch_004/test_results/test_predictions.jsonl) | [CONFIG.json](flashvtg/configuration/CONFIG.json) |

## 共同实现与结果复现

- `shared/train.py`：四个新 DETR 主干的共同训练入口；各主干的 `backend.py`、`model.py` 和 `source/` 保留独立实现。
- `shared/config.json`：S4_minus_Fusion 实际超参数。
- `shared/official_eval/`：官方数值评测；`shared/labels/`：原始 Standard 标签。
- `flashvtg/code/`、`flashvtg/baseline_source/`：历史无 QCD 的 FlashVTG 实现；辅助脚本版本差异见 `reports/FLASHVTG_SOURCE_AUDIT.json`。

在**仓库根目录**运行：

```bash
python -m pip install -r requirements-cpu.txt
python tools/reproduce_results.py --figures
# 可选，从原始冻结预测重算官方指标
python tools/reproduce_results.py --recompute-metrics
```

这些 CPU 入口不依赖外部权重。若重新训练或推理，需先配置本机特征、GMR 初始化权重和输出目录。历史训练代码不作静默路径改写。

## 原始归档记录

`MANIFEST.json`、`SHA256SUMS`、`ARCHIVE_SCOPE.json` 描述原始含权重的本地完整归档；本发布移除了二进制权重及无关数据集标签，所以它们仅作历史记录。实际 GitHub 文件的清单是根目录 `PUBLICATION_MANIFEST.json`。旧 `tools/verify_archive.py` 需要完整本地归档，不能用于本发布子集。

[原完整归档 README](../provenance/ORIGINAL_PHASE2_README.md) 保留其原始叙述。各主干的 `ARCHIVE_INDEX.json` 中检查点哈希保持原值；不表示二进制文件已包含于 GitHub。
