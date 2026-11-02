# Sim-DETR-GMR：SoccerGMR Standard GEQ 扩展

**本入口仅适配并验证 SoccerGMR Standard Test（1036条，544正例、492负例）。没有在本次扩展中适配或评测 Charades。** Git工作目录的历史名称与本入口的数据集无关。

[Standard结果与图表](../../results/geq_epoch_study/sim_detr/README.md) · [原生源码清单](SOURCE_MANIFEST.json) · [GEQ适配器](adapter.py)

`adapter.py`直接复用[`shared/geq_protocol/geq_model.py`](../../shared/geq_protocol/geq_model.py)中的`GlobalEventQueryHead`，包括相同的文本读出、逐层视频读出、残差FFN及最终存在性head。Sim-DETR两层encoder会被HD2MR重复调用，适配器只收集第一次encoder前向的两层视频证据，保持原训练检查点的取证位置。

Full的定义与main E1一致：保留原生GMR损失并加GEQ BCE，原模型与GEQ联合更新，学习率为1e-5/3e-5；Frozen只优化GEQ BCE并冻结原模型。`objective()`提供相同的损失组合接口。当前发布使用已有Val选定best，未重新训练。

Sim-DETR的原生GMR初始化来自它自己的checkpoint，不能称为原四骨干main E1的同一检查点。原生loss、层数、后处理及batch随骨干不同而保留；这属于同一GEQ方法的骨干扩展，不是所有训练细节相同的受控消融。

## 执行入口

推理使用原Sim环境，CPU绘图使用有NumPy/Matplotlib的环境：

```bash
CUDA_VISIBLE_DEVICES=0 /home/guoxiangyu/miniconda3/envs/sim_detr/bin/python \
  shared/geq_epoch_study/sim_detr_extension.py --action evaluate

python shared/geq_epoch_study/sim_detr_extension.py --action plot
python shared/geq_epoch_study/sim_detr_report.py
```

默认checkpoint路径在入口中给出，也可通过`--baseline`和`--source`指定。`--output`控制独立运行目录。已有COMPLETE只有在检查点、GT、指标、预测哈希一致时复用，不重复推理。入口包含独占锁。`source/`只含原生模型构造、数据和推理所需的传递依赖及原官方评估器，不含权重、特征或视频；实际数据/特征路径来自checkpoint的opt。

Standard推理保持原生CLIP→SlowFast拼接、文本处理、75步上限及原生窗口导出。Frozen在确认原生权重逐张量一致后复用GMR的候选边界、顺序和分数，只替换存在性分数。

## 离线重绘

无需模型、权重或GPU：

```bash
python shared/geq_protocol/plot_snapshot.py \
  --snapshot results/geq_epoch_study/sim_detr/full_frozen_comparison_snapshot.json \
  --output /tmp/soccer_simdetr_figures
```

本入口没有执行Full数据集评测，也没有把外部迁移报告的Full数据集结果纳入本次发布。Standard三种方法的固定阈值指标已与原迁移报告逐项复现一致；完整预测、官方指标、101点曲线及检查点SHA256随结果提供。
