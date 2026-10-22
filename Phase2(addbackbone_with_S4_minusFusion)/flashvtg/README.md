> GitHub publication: this is a code/results subset. Binary checkpoints mentioned below are external; see [publication scope](../../provenance/EXTERNAL_CHECKPOINTS.json). Original paths and hashes are retained for provenance.

# FlashVTG: epoch 4 归档

此前无 QCD 的 FlashVTG S4_minus_Fusion 参考实验，checkpoint 实际 epoch=4，SHA 与 TEST_COMPLETE 一致。基线只有已验证的 E0/GMR；不含未验证的 adapter-free Original。

- [epoch4 checkpoint](epoch_004/checkpoint.ckpt)，SHA256 `2e2ff2a728012bad5965e51b13b4ee4452b9ade09df25a571e1ba34a72dd4301`。
- [测试指标](epoch_004/test_results/test_metrics.json)、[1036 条测试预测](epoch_004/test_results/test_predictions.jsonl)。
- [当轮对比图](epoch_004/figures/this_epoch/flashvtg_analysis.png)、[累计对比图](epoch_004/figures/all_epochs_so_far/flashvtg_analysis.png)。两组均包括 analysis、gmiou、tradeoff、rej_f1 的 PNG/PDF/SVG。
- `epoch_004/figures/S4_TEST_CURVES.json` 保存 101 点阈值曲线及来源信息。
- `configuration/` 保存训练配置、初始化/预检信息；`snapshots/` 保存对应 GMR 初始化基线。
- 模型/数据适配代码和原生主干源码在本目录；FlashVTG 使用 `code/`、`historical_code/`、`baseline_source/`。
- `ARCHIVE_INDEX.json` 使用归档内相对路径定位主要文件；原始 JSON 中的历史绝对路径保留用于溯源。

CPU 复核与绘图（在归档根目录执行）：

```bash
/home/guoxiangyu/miniconda3/envs/gmr/bin/python tools/reproduce_epoch4.py --backbone flashvtg --output /tmp/flashvtg_epoch4_reproduce
```

该工具读取已保存的测试预测，不训练、不调用 GPU；重新模型推理仍需原始 CLIP/SlowFast/文本特征，特征大文件未重复打包。

## 历史代码来源说明

模型核心（含 Global Event Query、Gaussian refiner、LoRA）和39个历史源码文件与原始 PROVENANCE 哈希一致；对应原始 evaluation.py 已从本地旧实验副本恢复到 training_code_exact/。environment.py、runtime.py、train.py 的当前可用副本与历史记录存在差异，均原样归档并在 ../reports/FLASHVTG_SOURCE_AUDIT.json 中明确列出，未将其标作完全相同的历史版本。归档内 CPU 指标重算和图表重绘已独立验证，不依赖这些历史训练辅助脚本。
