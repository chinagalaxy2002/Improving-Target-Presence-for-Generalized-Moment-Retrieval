# CG-DETR-GMR

已完成配置在定位、额外片段召回与存在性指标上呈现不同变化。指标图报告完整 SoccerGMR Standard Test 的 mAP、mR+@5、AUROC 与阈值 0.4 下的 G-mIoU@3；G-mIoU 为 0–1，其余为百分数。曲线图对同一份冻结 Test 预测扫描 101 个存在性阈值，两轴均覆盖 0–1；未完成配置省略，单 Seed 结果不绘制跨 Seed 误差条。

Seed 2023; Base / Shared / SepFFN: effective batch 8; Instr / Multi / Adapt: 32.
