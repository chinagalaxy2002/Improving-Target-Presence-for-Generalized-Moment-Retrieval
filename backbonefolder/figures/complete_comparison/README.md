# 五个主干的六面板对比

每张图只包含 Baseline、GMR、GMR + S4_minus_Fusion(epoch 4)；FlashVTG 没有无 GMR 的 Baseline。

六个面板为：(a) 阈值与 G-mIoU@1；(b) 阈值与 G-mIoU@3；(c) 同阈值下相对 GMR 的 G-mIoU@1 变化；(d) N-acc 与 Positive recall；(e) 阈值与 Rej-F1；(f) 0.4 / 0.5 / 0.6 固定工作点。

- [moment_detr](moment_detr_fig7_test_threshold_sweep.png)：[PDF](moment_detr_fig7_test_threshold_sweep.pdf)、[SVG](moment_detr_fig7_test_threshold_sweep.svg)。
- [qd_detr](qd_detr_fig7_test_threshold_sweep.png)：[PDF](qd_detr_fig7_test_threshold_sweep.pdf)、[SVG](qd_detr_fig7_test_threshold_sweep.svg)。
- [cg_detr](cg_detr_fig7_test_threshold_sweep.png)：[PDF](cg_detr_fig7_test_threshold_sweep.pdf)、[SVG](cg_detr_fig7_test_threshold_sweep.svg)。
- [eatr](eatr_fig7_test_threshold_sweep.png)：[PDF](eatr_fig7_test_threshold_sweep.pdf)、[SVG](eatr_fig7_test_threshold_sweep.svg)。
- [flashvtg](flashvtg_fig7_test_threshold_sweep.png)：[PDF](flashvtg_fig7_test_threshold_sweep.pdf)、[SVG](flashvtg_fig7_test_threshold_sweep.svg)。

完整结果表有三个阈值版本，每表 14 行本地实验 + 6 行论文 MLLM 参考数据；见 [结果导航](../../README.md)。

在仓库根目录运行 `python tools/reproduce_results.py --figures` 重绘。`TABLE2_OUR_RESULTS*.csv` 是排版输入；`data/` 和相邻 Phase2 的 `shared/plot_baselines/` 是封存阈值曲线。论文转录记录见 `PAPER_TABLE4_RESULTS.json`。

本目录旧 `SOURCE_MANIFEST.json`、`ARTIFACT_AUDIT.json` 是加入论文参考行之前 14 组实验的历史核对记录；当前发布范围和文件哈希以仓库根目录 `PUBLICATION_SCOPE.json`、`PUBLICATION_MANIFEST.json` 为准。
