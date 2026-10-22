# 五个主干的完整结果

三张表均包含 14 组本地实验和 6 组论文 Table 4 的 MLLM 参考结果，S4 行标注相对同主干、同阈值 GMR 的百分点变化。

| 本地实验阈值 | Markdown | PDF | PNG |
|---|---|---|---|
| 0.4 | [表格](TABLE2_OUR_RESULTS.md) | [PDF](TABLE2_COMPLETE_RESULTS.pdf) | [图片](TABLE2_COMPLETE_RESULTS.png) |
| 0.5 | [表格](TABLE2_OUR_RESULTS_tau_0.5.md) | [PDF](TABLE2_COMPLETE_RESULTS_tau_0.5.pdf) | [图片](TABLE2_COMPLETE_RESULTS_tau_0.5.png) |
| 0.6 | [表格](TABLE2_OUR_RESULTS_tau_0.6.md) | [PDF](TABLE2_COMPLETE_RESULTS_tau_0.6.pdf) | [图片](TABLE2_COMPLETE_RESULTS_tau_0.6.png) |

论文 MLLM 的 Rej-F1 始终为原始 0.4 阈值结果；未报告的 mR@5、G-mIoU 留空。

- [仓库主 README：实验口径、复现命令与发布范围](../README.md)
- [各主干模型、训练、配置和结果映射](../docs/CODE_MAP.md)
- [五个主干的六面板图](figures/complete_comparison/README.md)
- CPU 重建三表：在仓库根目录运行 `python tools/reproduce_results.py`。

`moment_detr/`、`qd_detr/`、`cg_detr/`、`eatr/` 分别包含 `original/` 与 `gmr/` 的配置、冻结记录和完整测试预测；`flashvtg/gmr/` 是已验证的 E0。S4 epoch 4 的预测、代码与配置分别位于相邻 `Phase2(addbackbone_with_S4_minusFusion)/<backbone>/`。

历史子目录文档中的训练状态及本机绝对路径作为来源记录保留；所有 14 组本地实验均已完成。二进制权重与特征不在本次代码发布中，身份见 [外部检查点清单](../provenance/EXTERNAL_CHECKPOINTS.json)。
