# SoccerGMR：Standard训练 → Standard / Full两套测试

本目录集中介绍 **Moment-DETR、EaTR、QD-DETR、CG-DETR** 四个骨干的当前六种方法：GMR、GMR+GEQ Frozen、GMR+GEQ Full、GMR+Multi、GMR+Multi+GEQ Frozen、GMR+Multi+GEQ Full。

**同一批24个既有Standard Train/Val best，在两套测试域上各24条，共48条正式Test记录。不是48次独立训练。** Standard Test覆盖1036/1036；Full结果严格命名为 **Standard-trained checkpoints → SoccerGMR Full-Decodable Test (2970/2986)**，原始Full中16条历史排除另列。

“Full数据集”与“GEQ Full联合微调模式”是两种不同概念。本目录不混入Charades、FlashVTG、Shared、组件消融、历史main E1或固定四轮GEQ。其他历史实验不被本目录覆盖或改写。

## 结果入口

| 内容 | 文件 |
|---|---|
| 两套Test完整结果与mAP对照 | [RESULTS.md](RESULTS.md) |
| 全部正式指标PDF / LaTeX | [PDF](tables/ALL_TEST_METRICS.pdf) · [LaTeX](tables/ALL_TEST_METRICS.tex) |
| 48条数值宽表 / JSON | [CSV](data/RESULTS.csv) · [JSON](data/RESULTS.json) |
| 全部指标相对同骨干GMR的增减 | [DELTAS_VS_GMR.csv](data/DELTAS_VS_GMR.csv) |
| Standard主表 | [PNG](tables/STANDARD_PRIMARY.png) · [PDF](tables/STANDARD_PRIMARY.pdf) |
| Full-Decodable主表 | [PNG](tables/FULL_DECODABLE_PRIMARY.png) · [PDF](tables/FULL_DECODABLE_PRIMARY.pdf) |
| Standard四骨干曲线 | [总览PNG](figures/standard/ALL_BACKBONES.png) · [总览PDF](figures/standard/ALL_BACKBONES.pdf) · [分骨干PDF](figures/standard/BY_BACKBONE.pdf) |
| Full-Decodable四骨干曲线 | [总览PNG](figures/full_decodable/ALL_BACKBONES.png) · [总览PDF](figures/full_decodable/ALL_BACKBONES.pdf) · [分骨干PDF](figures/full_decodable/BY_BACKBONE.pdf) |
| 全部阈值曲线 / 各模型oracle值 | [CURVES.csv](data/CURVES.csv) · [ORACLE_BY_METHOD.csv](data/ORACLE_BY_METHOD.csv) |

各子图左为G-mIoU@1、右为@3，图例给出该方法同一Test的mAP（%）。每图一条黑色水平虚线表示该骨干六方法中，**存在性判断完全正确且定位候选保持不变时**的最高可恢复G-mIoU；不是实测阈值峰值，也不是可部署模型成绩。

## 训练、数据与复现

- [TRAINING.md](TRAINING.md)：原生训练及GEQ父子关系、seed/batch差异、选模、源码入口和复现约束。
- [DATA.md](DATA.md)：Standard/Full身份、GT规范化、16条排除、Full时间输入限制和测试集重叠。
- [REPRODUCE.md](REPRODUCE.md)：从本目录附带的压缩预测离线重算指标、重建表格与曲线。
- [HANDOFF.md](HANDOFF.md)：新会话继续工作、定位模型、训练/测试交接。
- [PROMPT.md](PROMPT.md)：清空上下文后可直接使用的完整提示词。

没有重新训练、微调、重选best、重新运行模型或按Test调参。本次整理仅读取原检查点/预测/日志，校验SHA256，并用官方定义离线核对两套Test。**全部48项与已发布数值一致，16项Frozen在对应测试域复用父候选且全部定位指标一致**，见[AUDIT.json](AUDIT.json)。

## 阅读指标

公共指标AUROC、mAP、mR@1、mR@5、mR+@5每行只列一次；N-acc、Rej-F1、G-mIoU@1/@3分别给出存在性阈值τ=0.4/0.5/0.6。额外官方定位与分类指标放在PDF附页及[metrics/](metrics/)中。

**G-mIoU及其差值为0–1；其余指标为百分数，差值为百分点。** 所有增减一律为“方法−同骨干、同测试集GMR”，增益绿、下降红、持平灰；GMR baseline明确标出。定位指标使用正例未门控候选；分类与G-mIoU使用完整正负域。N-acc=TN/负例数，不是整体分类Acc。

## 必须保留的比较边界

Standard Train/Val为4138/465条，Test为1036条；Full-Decodable Test为2970条（1633正/1337负），原始Full为2986条（1638正/1348负）。Full和Standard使用独立qid及文本特征，本轮Full预测独立生成。

Full Test与Standard Test共享**1036个规范化视频stem+查询对**，不是独立实验样本。Standard Train/Val与原始Full Test的规范化video stem初查交集为0。两段Full长视频受原75步上限截断，另3段存在特征时长/GT duration偏差，详见DATA.md。2970条覆盖指每条查询均推理，不保证所有视频全时间范围都被编码。

已有路线的训练预算/seed不同：EaTR GMR是seed2026的既有baseline复用；其他当前路线主要为seed2023。部分GMR有效batch8而Multi为32，Moment日志快照也未达到配置最大400轮。完整方法比较不能据此写成相同预算的组件消融。

Test与曲线已经被研究者查看。后续优化应在Train/Val拟合及选模，不能继续宣称这些Test是从未触及的独立验证；在已查看Full Test上的后续提升应标明探索性。

## 仓库提供什么

本目录附带48份gzip逐查询预测、测试GT、训练/选模配置与历史、小型严格加载/完成证据、24个检查点的路径/SHA256/参数布局、官方评估器、原环境训练与Full评估源码快照、数值/图表和交接文档。

**大型模型权重和视频/文本特征没有上传。** 权重的原本机绝对路径、SHA256、父子关系见[data/CHECKPOINTS.json](data/CHECKPOINTS.json)；特征协议和指纹见[data/](data/)。这些绝对路径用于原机器定位，不代表GitHub下载链接。离线指标/图表重建不需要权重、特征或GPU；新训练/模型推理仍需要外部输入及路径适配。

原环境只读导出工具为[tools/export_local.py](tools/export_local.py)，离线可移植入口为[rebuild.py](rebuild.py)。源码快照保留历史硬编码路径，不能把它们当成无需配置的一键跨机训练包。来源与原始SHA256见[PROVENANCE.json](PROVENANCE.json)。
