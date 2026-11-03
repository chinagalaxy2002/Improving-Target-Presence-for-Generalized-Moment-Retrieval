# Moment-DETR + Query-conditioned Clip Supervision

本分支只包含 **逐 clip 辅助监督**的 Moment-DETR 实现、必要的 GMR/GEQ 基础组件，以及 A/B/C/D 实验。它研究：训练时逐段判断“这个视频片段是否属于文字查询描述的事件”，能否改善最终时间定位。

**本轮结果：mAP 有提升，G-mIoU 未改善；Clip Head 低召回，不能据此宣称整体性能提升。**

| 组别 | Stage 1 GMR 加 Clip Loss | Stage 2 GEQ 加 Clip Loss | Standard mAP | Full mAP | Standard G1 均值 | Full G1 均值 |
|---|---|---|---:|---:|---:|---:|
| A 对照 | 否 | 否 | 6.05 | 5.26 | **52.77** | **49.12** |
| B | 否 | 是 | **6.83** | 5.28 | 52.74 | 49.09 |
| C | 是 | 否 | 6.82 | **6.01** | 52.60 | 48.91 |
| D | 是 | 是 | 6.69 | 5.89 | 51.06 | 47.49 |

指标为 0–100；G1 均值为存在性阈值 0.4/0.5/0.6 下 G-mIoU@1 的均值。Standard Test 为 1036 条，Full-Decodable Test 为 2970/2986 条。权重 λ=0.1 由 **Standard Val** 选择，未用 Test 调参。单 seed=2023，各组实际早停轮次不同；两个 Test 集有样本重叠。

[方法与完整结果](docs/METHOD_AND_RESULTS.md) · [详细结果](results/clipseg/RESULTS_ZH.md) · [CSV](results/clipseg/METRICS.csv) · [各 tIoU / 短事件 / 多区间结果](results/clipseg/DETAILS.json)

![全部训练曲线](results/clipseg/training_val_panel.png)

[训练图 PDF](results/clipseg/training_val_panel.pdf) · [SVG](results/clipseg/training_val_panel.svg)

![固定 Test 对比](results/clipseg/COMPARISON.png)

## 实现

- [`clip.py`](src/soccergmr/clip.py)：文本掩码池化、视频/文本显式融合、逐 clip logits、masked BCE、GMR wrapper。
- [`geq.py`](src/soccergmr/geq.py)：原有 GEQ 存在性判断与辅助 Clip Head。
- [`train.py`](src/soccergmr/train.py)：阶段损失、头部优化器参数组、严格父 checkpoint 继承、断点恢复和 Val 选模。
- [`data.py`](src/soccergmr/data.py)：真实 duration 到 clip 标签的映射、多区间并集、空查询全零标签。

Clip Head **仅参与训练监督**；不修改推理边界、候选排序或 GEQ 门控。C/D 共用有 Clip Loss 的 GMR best；C 在 GEQ 阶段保留并冻结 Clip Head，D 继续更新它。A/B 共用无 Clip Loss 的 GMR best。

## 安装和数据

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test,report]'
export SOCCERGMR_LABEL_ROOT=/path/to/labels
export SOCCERGMR_STANDARD_FEATURES=/path/to/standard_features
export SOCCERGMR_FULL_FEATURES=/path/to/full_features
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

数据目录见 [DATA.md](docs/DATA.md)。标签/特征/模型权重不随分支发布。归档预测已压缩保存，提供原始标签后可重算所报告的指标。

## 从头复现

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONPATH=src \
python scripts/run_experiments.py --output runs/reproduction --all-gmr-candidates
```

该入口从头训练 A 的 GMR，完成 A GEQ 与 B 的 λ=0.1/0.3/1.0；根据 B 的 Val 三阈值 G1 均值选 λ（并列选较小 λ），随后完成 C/D。`--all-gmr-candidates` 还训练其余两个 GMR 权重候选，对应历史中提前并行进行的三个父模型。所有已选模型固定评测两套 Test，B/C/D 额外生成 Val Clip 诊断与 query 热力图。断点重跑同一命令会从合法 `last.pt` 恢复；完整结果写入输出目录 `summary.json`。

单阶段也可运行：

```bash
PYTHONPATH=src python -m soccergmr train --config configs/experiments/A.json \
  --stage gmr --output runs/A/gmr
PYTHONPATH=src python -m soccergmr train --config configs/experiments/B_01.json \
  --stage geq_full --baseline runs/A/gmr/best.pt --output runs/B_01/geq_full
```

训练身份包含源码、配置及 Train/Val 资产；Stage 1 只忽略未来 Stage 2 的 Clip 权重，以允许 A/B 或 C/D 共享严格匹配的父模型。禁止跳过身份检查。

## 验证与绘图

```bash
PYTHONPATH=src pytest -q
OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/verify_pipeline.py --output runs/verification
PYTHONPATH=src python scripts/verify_results.py
PYTHONPATH=src python scripts/plot_training.py --output runs/plots
```

`verify_pipeline.py` 生成微型合成数据，验证完整 A/B/C/D 训练→权重继承→双 Test 以及 C 头冻结/D 头更新；合成分数不是性能结果。`verify_results.py` 验证归档预测的哈希、覆盖率、选 λ 规则和表格；额外传入 `--label-root /path/to/labels` 可重算 12 组正式 Test 指标。`plot_training.py` 不依赖原数据或 checkpoint，直接从归档历史重绘训练图。

## 历史结果与公开版的关系

归档结果来自本轮原始源码快照，原哈希与执行记录保存在 [`results/clipseg/provenance`](results/clipseg/provenance)。公开版去掉了无关骨干和本机调度脚本，因而源码身份不同；结果不是由精简后的源码重新训练得到的。请在本分支源码下从头训练，不应直接绕过校验加载历史 checkpoint。

原实验按用户后续指令使用双 GPU、每卡最多三个任务；C_10 的 GMR 在完整 epoch checkpoint 上从 GPU0 恢复到 GPU1。C/D 微调共享同一个 λ=0.1 的父模型。原 GPU1-only 审计失败记录与双 GPU 重审通过的结论在来源说明中如实区分。

只发布与此思路有关的代码、配置、归档历史/预测、图和审计；不包含其他骨干实验、数据集、特征、权重或服务器服务配置。
