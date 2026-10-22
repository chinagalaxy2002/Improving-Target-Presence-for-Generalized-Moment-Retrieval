> GitHub publication: this is a code/results subset. Binary checkpoints mentioned below are external; see [publication scope](../../provenance/EXTERNAL_CHECKPOINTS.json). Original paths and hashes are retained for provenance.

# EaTR: epoch 4 归档

本轮六轮微调中的固定 epoch4，未按 best 筛选。累计图中的 epoch3 曲线仅作为绘图支撑数据保留，不含其他轮次 checkpoint。

- [epoch4 checkpoint](epoch_004/checkpoint.ckpt)，SHA256 `190ed20416ae10758ce2f6426dcd611d9cb403218ec6930fb90e14f050122053`。
- [测试指标](epoch_004/test_results/test_metrics.json)、[1036 条测试预测](epoch_004/test_results/test_predictions.jsonl)。
- [当轮对比图](epoch_004/figures/this_epoch/eatr_analysis.png)、[累计对比图](epoch_004/figures/all_epochs_so_far/eatr_analysis.png)。两组均包括 analysis、gmiou、tradeoff、rej_f1 的 PNG/PDF/SVG。
- `epoch_004/figures/S4_TEST_CURVES.json` 保存 101 点阈值曲线及来源信息。
- `configuration/` 保存训练配置、初始化/预检信息；`snapshots/` 保存对应 GMR 初始化基线。
- 模型/数据适配代码和原生主干源码在本目录；FlashVTG 使用 `code/`、`historical_code/`、`baseline_source/`。
- `ARCHIVE_INDEX.json` 使用归档内相对路径定位主要文件；原始 JSON 中的历史绝对路径保留用于溯源。

CPU 复核与绘图（在归档根目录执行）：

```bash
/home/guoxiangyu/miniconda3/envs/gmr/bin/python tools/reproduce_epoch4.py --backbone eatr --output /tmp/eatr_epoch4_reproduce
```

该工具读取已保存的测试预测，不训练、不调用 GPU；重新模型推理仍需原始 CLIP/SlowFast/文本特征，特征大文件未重复打包。
