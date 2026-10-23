# 两组件消融：Global Event Query + Hard-RoI

本实验将精简方法定义为 **GMR + Global Event Query + Hard-RoI 边界修正**。不使用 LoRA、Gaussian-RoI、6D Saliency、候选存在性融合、QCD 或合成负样本。旧的 S4_minus_Fusion 六配置消融保留为历史证据，不用旧的「LoRA + Hard-RoI」行冒充新的无 LoRA 方法。

采用完整 2×2 因子设计，在 Moment-DETR、QD-DETR、CG-DETR、EaTR、FlashVTG 上各设置四组，共 20 个对照单元：

| 配置 | Global Event Query | Hard-RoI | 新训练轮数 | 作用 |
|---|---|---|---:|---|
| GMR | 关闭，保留原 GMR 存在性头 | 关闭 | 0 | 预训练 GMR 参考 |
| GMR + Global | 开启 | 关闭 | 4 | 独立检验全局存在性建模 |
| GMR + Hard | 关闭，保留原 GMR 存在性头 | 开启 | 4 | 独立检验局部边界修正 |
| GMR + Global + Hard | 开启 | 开启 | 4 | 精简后的完整方法 |

每组都独立从对应 backbone 的同一个预训练 GMR checkpoint 初始化。冻结原始 GMR 的全部参数，仅训练已开启的新增模块；不串接上一行 checkpoint，不重新训练原 GMR backbone。优化器与调度器均重新创建。Global-only 与 GMR 在原始消融中已有完全匹配结果，因此核验后逐字节复用；Hard-only 与 Global+Hard 是新增的 **10 次四轮训练**。总计 20 个对照单元，其中 15 个含四轮微调，5 个为冻结基线。

统一固定 seed=2026、**第 4 个微调 epoch**、原始 Standard Train/Val/Test=4,138/465/1,036；Test 含 544 条正查询和 492 条负查询。主工作点 τ=0.4，τ=0.5/0.6 仅作预先固定的敏感性分析。新增训练在四轮结束后才评估 Val/Test；不按 Test 选择轮次或阈值。各 backbone 沿用自身原生 batch size，同一 backbone 的后处理完全一致。AdamW 的 lr=3e−5、wd=1e−4，梯度裁剪=0.1，边界损失权重=0.5；其他冻结超参数见 [config.json](config.json)。

Hard-RoI 使用候选区间内的有效帧中心进行均匀池化，离散网格内无帧时沿用最近帧回退；使用 top-8 候选、几何嵌入、候选对文本的交叉注意力及边界/置信度修正。最大边界偏移为 1.5 秒，监督包含归一化 Smooth-L1、GIoU 与 IoU 置信度损失，只作用于有目标查询。Hard-only 的全负批次没有可训练目标，正常遍历但不更新参数。Global 分支读取文本及分层视频表示，使用存在性 BCE；存在性不读取 RoI token 或修正候选。

所有不需要的 LoRA wrapper 都被拆除，还原原始预训练线性层。Saliency 参数冻结，并在 refiner 的输入处整段旁路，包括偏置的作用。为对齐初始化，历史构造器仍先按共同顺序创建张量；`CONFIG.json` 的 `initialization.construction_arguments` 仅记录该构造过程，`job` 与 `EFFECTIVE_CONFIG.json` 才是实际开关。旧构造器可能保留不参与输出或梯度的休眠诊断分支；不会启用 Gaussian、saliency 或候选存在性融合。

四个 DETR 风格模型沿用既有两秒网格导出；FlashVTG 沿用原生 soft presence confidence calibration、四舍五入及 NMS。FlashVTG 的输出置信度排序可受存在性分数影响，不能把最终 mAP 的全部变化归因于边界坐标。原生退化候选也沿用既有官方评估规则。

结果按百分比报告 N-acc、AUROC、Rej-F1、mAP、mR@1、mR@5、mR+@5、G-mIoU@1、G-mIoU@3。正查询定位指标不使用存在性硬门控；存在性分数严格大于 τ 才在 G-mIoU 中保留候选。增益以同 backbone、同 split、同阈值的 GMR 为参照，单位 pp；显示增益按表中两位小数相减。因子表另外报告 Global/Hard-RoI 的四个条件效应与原始数值的差中差，不能据此声称多随机种子的统计交互。

- [完整 Val/Test 结果](reports/RESULTS.md)
- [精简方法的结果解读](CONCLUSIONS.md)
- [全部九项指标 CSV](reports/all_metrics.csv)
- [相对 GMR 的增益 CSV](reports/gains_vs_gmr_pp.csv)
- [四个条件效应及交互 CSV](reports/factorial_effects_pp.csv)
- [Test τ=0.4 表格 PDF](tables/ABLATION_TEST_tau_0.4.pdf) / [PNG](tables/ABLATION_TEST_tau_0.4.png)
- [固定协议](PROTOCOL.json)
- [复用审计](reports/REUSE_AUDIT.json)
- [官方指标复算](reports/METRIC_VERIFICATION.json)

这些是对已经观察过 Standard 标签与旧消融结果的探索性单种子实验。固定 epoch 4 能避免新一轮按 Test 选轮次，但不能使已知 Test 恢复为独立确认集。结果支持范围限于本数据、种子和四轮微调预算；下降项原样报告。

## 复现

冻结预测的表格与官方指标复算不需要 GPU、特征或 checkpoint：

```bash
pip install -r requirements-cpu.txt
python summarize.py
python summarize.py --verify
python tables.py
```

依赖参考仓库根目录 requirements 与原组件实验环境。训练需另备原始 Soccergmr 特征和五个对应 GMR checkpoint；历史绝对路径保存在各 backend、checkpoint 配置与 provenance 中，迁移机器时需先重定向这些路径。checkpoint 与特征不上传到 GitHub。公开包中的运行预测目录使用 `results/`，复算脚本自动识别该目录；训练脚本使用本地 `runs/`。

本地完整实验：

```bash
/home/guoxiangyu/miniconda3/envs/gmr/bin/python reuse.py
/home/guoxiangyu/miniconda3/envs/gmr/bin/python launch.py
/home/guoxiangyu/miniconda3/envs/gmr/bin/python audit.py --backbone moment_detr
/home/guoxiangyu/miniconda3/envs/gmr/bin/python summarize.py --verify
/home/guoxiangyu/miniconda3/envs/gmr/bin/python tables.py
```

`launch.py` 通过两个 GPU 子进程工作线程顺序执行各任务，不覆盖已有目录。每个新增任务记录训练历史、epoch_004.ckpt 哈希、严格重载、可训练参数分组以及全部冻结参数和 buffer 的逐张量检查。`audit.py` 在训练后独立重建模型，比较两组共同初始化，检查硬窗口的区间外不变性、padding 不变性、saliency 扰动不变性、RoI 扰动下存在性不变性，以及 Hard-only 原 GMR 存在性分数不变性。

启动阶段曾暴露 Hard-only 全负批次无梯度的情况；初次不完整运行被停止并保存在本地 `attempts/initial/`，修复后全部新训练从头重跑。失败尝试不进入正式计数，未改变训练超参数，也未据此选择 Test 结果。
