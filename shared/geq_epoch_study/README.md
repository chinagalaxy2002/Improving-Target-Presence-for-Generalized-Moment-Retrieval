# main E1 同源：GEQ全量与冻结对照

四骨干各两个方案，八组已完成训练和完整Val/Test。GEQ微调seed2023。
原预训练GMR检查点来自main E1历史实验；不使用未完成的EaTR重训。
Moment-DETR按检查点记录恢复CLIP→SlowFast输入顺序。

每轮以完整Val在0.4/0.5/0.6的G-mIoU@1均值选best，记录三点是否均超过Baseline；
连续超过5轮（6轮）不提升早停，上限20轮。Test只评估Val选定的best。
full：原GMR损失+GEQ BCE，原模型lr1e-5、GEQ lr3e-5；frozen只训练GEQ BCE。
native batch8/32，num_workers2，无AMP，每GPU最多两个CUDA任务。

本工具依赖服务器原仓库 `ablations/end_to_end_geq_pv_20261026` 及其记录的
官方骨干、数据、特征和原始检查点；完整来源SHA256保存在各CONFIG/COMPLETE中。
该初始化与当前多路线实验不同，结果单列，不混作受控消融。

结果与图表：`results/geq_epoch_study/README.md`。绘图及自动上传统一由
`../geq_protocol/report.py --baseline-study` 和 `../geq_protocol/publish.py` 执行。
PNG/PDF/SVG、101点CSV、离线快照均按骨干保存，GitHub根README直接展示图片。

离线重绘只需NumPy和Matplotlib：

```bash
python shared/geq_protocol/plot_snapshot.py \
  --snapshot results/geq_epoch_study/eatr/full_frozen_comparison_snapshot.json \
  --output /tmp/geq_epoch_figures
```

旧队列名保留在启动脚本的兼容迁移记录中；当前调度是full/frozen共同排队，
不是全量阶段全部完成后才运行冻结。当前八组已完成，重复运行不会重新训练。
