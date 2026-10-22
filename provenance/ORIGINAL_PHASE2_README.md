# Phase 2：S4_minus_Fusion 跨主干 epoch4 归档

按 backbonefolder 的主干划分组织，五个主干分别独立存放：

- [moment_detr/](moment_detr/README.md)
- [qd_detr/](qd_detr/README.md)
- [cg_detr/](cg_detr/README.md)
- [eatr/](eatr/README.md)
- [flashvtg/](flashvtg/README.md)

[五个主干的 epoch4 测试结果表](EPOCH4_TEST_RESULTS.md) · [CSV](EPOCH4_TEST_RESULTS.csv)

## 各主干内的目录

```text
<backbone>/
├── epoch_004/
│   ├── checkpoint.ckpt
│   ├── test_results/                  # 原始测试预测和官方测试指标
│   ├── figures/
│   │   ├── this_epoch/                # 原基线曲线 + epoch4 S4 曲线
│   │   └── all_epochs_so_far/         # 原基线曲线 + 截至 epoch4 的 S4 曲线
│   └── supporting_curves/             # 四个新主干的 epoch3 绘图支撑数据
├── configuration/                    # 完整训练配置、初始化、预检/历史来源
├── snapshots/                        # 对应 GMR 初始化基线权重
├── logs/                             # 四个新主干的完整训练日志及 epoch4 记录
├── source/ 或 baseline_source/        # 相关原生模型源代码
└── ARCHIVE_INDEX.json                 # 主要文件的归档内相对路径与哈希
shared/                               # 原始公共训练/评测/绘图代码，基线曲线和标签
tools/                                # 归档核验、测试结果重算与图表重绘入口
reports/                              # 来源、环境、复制清单、完整性验证
```

五个 epoch4 权重均为真实 epoch4。Moment/QD/CG/EaTR 来自六轮固定轮次微调，不按 best 筛选。FlashVTG 来自原有无 QCD 的 S4_minus_Fusion 参考实验，其历史 best 恰为 epoch4；已核对 checkpoint 和测试完成记录的 SHA256。未将 QCD 增强实验混入本归档。

所有 S4 曲线均在完整 Standard test（1036 条）上测得，不是 validation 曲线。四个新主干保留 Original/GMR 的 k=1、k=3 四条基线；FlashVTG 只有已验证的 E0/GMR 基线，未补造 Original。图像包括 analysis、gmiou、tradeoff、rej_f1 四类，均有 PNG/PDF/SVG。累计图中的其他轮次仅保留绘图支撑曲线；此归档只含 epoch4 微调 checkpoint。

## 复核归档与重算结果

在本归档根目录执行：

```bash
/home/guoxiangyu/miniconda3/envs/gmr/bin/python tools/verify_archive.py
/home/guoxiangyu/miniconda3/envs/gmr/bin/python tools/reproduce_epoch4.py --metrics-only --output /tmp/epoch4_metrics_recheck
/home/guoxiangyu/miniconda3/envs/gmr/bin/python tools/reproduce_epoch4.py --backbone qd_detr --output /tmp/epoch4_qd_figures
```

后两个命令只读取归档内预测、测试标签和曲线数据，使用原官方数值评测及绘图代码，不训练、不调用 GPU。输出目录必须是新目录。`requirements_frozen.txt` 和 `reports/ENVIRONMENT.json` 记录实际 gmr 运行环境。

`shared/train.py`、各主干模型/数据代码及 FlashVTG 参考代码为来源代码快照，不作静默路径改写；其中历史绝对路径仅用于溯源/原运行环境。若需再次执行模型推理或训练，仍需原始 CLIP、SlowFast、文本特征并配置相应路径；大体积特征未复制。用于本归档的 CPU 指标/绘图复现工具不依赖这些外部路径。

归档使用独立复制，原实验的检查点、结果、脚本与图像全部保留。`MANIFEST.json`/`SHA256SUMS` 覆盖归档的所有固定文件；原始 JSON 内的历史路径不作为归档内入口，主要文件请查各主干的 `ARCHIVE_INDEX.json`。

FlashVTG 历史模型核心源码与来源哈希一致；3个历史训练辅助脚本的可用副本存在版本差异，已在 [源码核对记录](reports/FLASHVTG_SOURCE_AUDIT.json) 中列出。保存预测的测试指标重算已对五个主干全部通过，图表重绘入口也已验证。
