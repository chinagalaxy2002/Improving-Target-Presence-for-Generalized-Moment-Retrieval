# Multi-Route GMR 阶段性实验报告

更新时间：2026-10-31 07:54 CST

## 1. 实验口径

- 数据集：Soccer-GMR Standard。
- 随机种子：单 Seed `2023`。
- 主路线：10 queries，Hungarian O2O。
- 辅助路线：24 queries，Temporal O2M，最大 `K=2`。
- 辅助总损失权重：0.3；双路线时每条路线 0.15。
- Instruction Tokens：4。
- Adaptive tIoU 阈值：0.25。
- 有效 Batch Size：`2 × 4 = 8`。
- 每任务 `num_workers=2`；每张 GPU 最多并行两个任务。
- 每 5 Epoch 验证；连续 20 Epoch 未更新最佳验证集 mAP 时早停。
- Checkpoint 由验证集 mAP 选择，冻结后执行一次测试集评估。
- mAP、mR、AUROC 和 Rej-F1 沿用官方评估器的百分制；G-mIoU 在表中转换为 0–1。

## 2. 当前完成状态

| Backbone | Baseline | Shared | Separate FFN | Instruction | Multi | Adaptive |
|---|---|---|---|---|---|---|
| EaTR | 完成 | 完成 | 完成 | 训练中 | 训练中 | 排队中 |
| QD-DETR | 完成 | 完成 | 完成 | 训练中 | 训练中 | 排队中 |
| CG-DETR | 已修复并续训 | 已修复并续训 | 已修复并续训 | 排队中 | 排队中 | 排队中 |

CG-DETR 的前三项曾在 Epoch 5 验证推理阶段失败。原因是启用 DataLoader worker 后，CG-DETR 原生 `infer()` 直接使用了 CPU batch，而模型参数位于 CUDA，触发 CPU/CUDA LayerNorm 设备不一致。该问题已通过 main-process device wrapper 修复，并用 Epoch 4 checkpoint 完成正/空样本真实推理验证；Baseline、Shared 和 Separate FFN 已从 Epoch 5 重新续训。当前仍没有有效的 CG-DETR 最终测试结果，因此暂不能与另外两个骨干比较。

## 3. 已完成测试集结果

| Backbone | Variant | Best Epoch | Val mAP | Test mAP | mR@1 | mR@5 | mR+@5 | AUROC | Rej-F1@0.4 | G-mIoU@3 (0–1) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EaTR | Baseline | 15 | 8.08 | 6.86 | 3.60 | 10.12 | 0.87 | 69.99 | **75.00** | **0.4900** |
| EaTR | Shared O2M | 80 | **19.42** | 16.65 | 11.20 | **23.55** | 8.60 | 77.57 | 61.23 | 0.3139 |
| EaTR | Separate FFN | 70 | 19.18 | **17.06** | **11.32** | 22.59 | **9.21** | **77.75** | 54.93 | 0.2655 |
| QD-DETR | Baseline | 85 | 16.40 | 15.08 | 10.04 | **20.11** | 9.62 | **77.66** | 59.84 | 0.3035 |
| QD-DETR | Shared O2M | 110 | **18.50** | **15.24** | **10.64** | 20.00 | **9.78** | 77.13 | 63.37 | 0.3334 |
| QD-DETR | Separate FFN | 35 | 14.28 | 11.03 | 7.34 | 15.36 | 4.16 | 75.57 | **68.25** | **0.3783** |

粗体仅表示当前同一骨干三项已完成结果中的最佳值，不包括仍在运行或缺失的配置。

## 4. 相对 Baseline 的变化

| Backbone | Variant | ΔmAP | ΔmR@1 | ΔmR@5 | ΔmR+@5 | ΔAUROC | ΔRej-F1 | ΔG-mIoU@3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| EaTR | Shared O2M | +9.79 | +7.60 | +13.43 | +7.73 | +7.58 | -13.77 | -0.1761 |
| EaTR | Separate FFN | +10.20 | +7.72 | +12.47 | +8.34 | +7.76 | -20.07 | -0.2245 |
| QD-DETR | Shared O2M | +0.16 | +0.60 | -0.11 | +0.16 | -0.53 | +3.53 | +0.0299 |
| QD-DETR | Separate FFN | -4.05 | -2.70 | -4.75 | -5.46 | -2.09 | +8.41 | +0.0748 |

## 5. 阶段性观察

### EaTR

Shared O2M 和 Separate FFN 均显著提高定位指标。Separate FFN 相比 Baseline 的测试集 mAP 提升 10.20 个百分点，mR+@5 提升 8.34 个百分点，说明密集监督确实提高了多个真实片段的覆盖能力。

但固定存在性阈值 0.4 下，Rej-F1 和 G-mIoU 均明显下降。与此同时 AUROC 上升约 7.6 个百分点，说明排序区分能力改善、固定阈值决策却变差，更像是存在性分数校准发生漂移，而不是存在性表示完全失效。在不使用测试集调参的前提下，应进一步检查验证集上的存在性分数分布和阈值校准。

### QD-DETR

Shared O2M 的定位结果基本与 Baseline 持平：mAP 提升 0.16，mR+@5 提升 0.16；同时 Rej-F1 提升 3.53，G-mIoU@3 提升 0.0299。这是当前已完成结果中最均衡的 QD-DETR 配置。

Separate FFN 呈现明显的任务权衡：G-mIoU@3 和 Rej-F1 更高，但 mAP、mR@5 和 mR+@5 明显下降。因此当前结果不支持“独立 FFN 在所有骨干上都改善主定位路线”的强结论。

### 跨骨干结论边界

当前只能认为 O2M 对 EaTR 的定位增益很强，对 QD-DETR 的增益较弱且依赖路线结构。Instruction、Multi 和 Adaptive 尚未完成，CG-DETR 尚无有效结果，因此 H4–H6 和完整的跨骨干泛化结论均需等待。

## 6. 尚不能宣称的结论

- 不能宣称 Multi-Route 优于单辅助路线，因为 Multi 尚未完成。
- 不能宣称 Adaptive matching 有效，因为 Adaptive 尚未完成。
- 不能宣称 Separate FFN 普遍有效；它在 EaTR 与 QD-DETR 上表现方向相反。
- 不能进行 CG-DETR 横向比较，因为当前 CG-DETR 结果无效。
- 所有结果来自单个 Seed 2023，不包含跨 Seed 均值、标准差或显著性检验。

## 7. 文件位置

- 简表：`RESULTS.md`
- CSV：`results.csv`
- 队列状态：`QUEUE.json`
- 每项最终指标：`runs/<backbone>/<variant>/test_metrics.json`
- 每项最佳验证指标：`runs/<backbone>/<variant>/best_val_metrics.json`
- 每项训练轨迹：`runs/<backbone>/<variant>/history.jsonl`
- 训练日志：`logs/<backbone>_<variant>.log`
