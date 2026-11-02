# 四骨干 Shared / Multi + Global Event Query

四曲线六面板：每个骨干包含 **GMR baseline、GMR baseline + GEQ、multi + GEQ、shared + GEQ**，按 main 的 `backbonefolder/figures/complete_comparison` 版式，报告 G-mIoU@1 / @3 阈值曲线、相对 baseline 的 ΔG-mIoU@1、拒绝与保留关系、Rej-F1、三个固定工作点。所有 G-mIoU 绝对图轴均为 0–1，差值允许为负；图例和 CSV 同时报告四个方法的 mAP（官方正查询定位百分数，不按存在性阈值门控）。EaTR 继续使用已完成的历史 seed 2026、mAP 16.20 baseline，所有 GEQ 使用 seed 2023；未采用未完成的 EaTR 重训快照。

从公开快照离线重绘六面板和 @1 / @3 双面板，只需要 NumPy / Matplotlib：

```bash
python shared/geq_finetune/plot_complete_comparison.py \
  --snapshot results/geq_finetune/four_way_comparison_snapshot.json \
  --output results/geq_finetune
```

服务器上可用 `python shared/geq_finetune/plot_complete_comparison.py --refresh` 从冻结预测复算全部 16 条模型曲线。统一入口 `collect_and_plot.py` 也会自动更新四曲线图。每个骨干目录独立存储六面板、双面板 PNG/PDF/SVG 和曲线数据；结果根目录 `four_way_map.csv`、`four_way_threshold_sweep.csv`、`four_way_fixed_points.csv` 提供 mAP、101 点曲线和预先固定的 0.4/0.5/0.6 工作点。`three_way_*` 为此前三曲线的历史数据。

新增四组 baseline + GEQ 控制，与原八组使用相同的固定四轮 GEQ-only 协议，严格从图中同一个 baseline 初始化。新增输出在 `<backbone>/baseline/`，`BASELINE_GEQ_QUEUE.json` 与原八组队列分开；每卡最多两个任务，计入已运行 CUDA 进程。启动器完成四组训练/测试后自动绘制四曲线，不覆盖原八组。

```bash
bash shared/geq_finetune/launch_baseline.sh
python shared/geq_finetune/audit.py --include-baseline
```

十二组均已完成训练与完整 Val/Test。服务器完整权重审计见 `results/geq_finetune/FINAL_AUDIT_12_GEQ.json`；原八组审计保留在 `FINAL_AUDIT.json`。

八组均已完成，见 [结果、阈值对照与图表](../../results/geq_finetune/README.md)。发布包仅对 `common.py` 与启动器作路径适配；模型与训练逻辑与服务器使用的脚本一致。原始运行配置内的代码 SHA256 记录实际训练时版本。

此工具在四个骨干各自已完成的 `baseline`、`shared`、`multi` best 上，独立微调新增 GEQ，共十二组。原有实验目录只读。

参考 main commit `39ab0eb74427709851476b2bb86b698c8a3f4186` 的 `ablations/component_ablation_20261023` row 2 与 `ablations/global_hard_roi_20261023` Global-only 协议：冻结原模型所有参数及 buffer，并保持原模型 eval；只训练 GEQ，固定最终 epoch 4。GEQ 依次读取投影后的文本和每层 encoder 的视频 token；QD/CG 跳过开头的 saliency token。辅助 O2M decoder 不执行。不存在 LoRA、RoI、边界修正、候选融合或数据增强。

AdamW lr=3e-5、wd=1e-4，grad clip=0.1，StepLR(8, gamma=0.5)，AMP 关闭。沿用本轮单 seed 2023（main 参考为 2026）；Moment-DETR 使用原生 batch 8，其余骨干 batch 32；每任务 num_workers=2。所有组都从对应原路线 best 初始化，GEQ 使用相同 seed 初始化。非从其他 GEQ 串接。Val/Test 在固定四轮结束后评估一次，不通过 Test 调参。

`final.ckpt` 仅保存新增 GEQ，附带冻结来源路径和 SHA256；完整推理需同时加载记录的原始 best。`last.ckpt` 另存优化器、调度器和 RNG 以便故障续训。每轮逐张量状态摘要核验原模型所有参数和 buffer 未变。原检查点不覆盖。

Test 使用原 best 已导出的固定候选，再用 GEQ 在完整 Test 输入上输出的分数替换存在性分数；候选窗口、候选分数、顺序逐样本精确不变。对照指标重新计算并要求复现原指标；mAP/mR/mR+ 不允许变化。新增 GEQ 改变 AUROC、Rej-F1 和 G-mIoU。两条曲线均覆盖阈值 0–1 的 101 点，G-mIoU 以 0–1 输出；0.4/0.5/0.6 为预先固定工作点。

```bash
bash shared/geq_finetune/launch.sh
# 每 GPU 最多两个 CUDA 进程，计入既有任务；tmux 保持调度器存活。
tail -f geq_finetune_experiments/20261031/queue.log
python shared/geq_finetune/collect_and_plot.py
```

队列每个子进程先检查混合 null/single/multi 与全 null 真数据、GEQ 梯度、冻结模型无梯度、原始定位精确相等、GEQ 严格重载和推理不执行辅助路线。最多重试三次，已有完成标记不会重跑。重启调度器时如发现上次子任务仍活着则拒绝重复启动。

结果：`geq_finetune_experiments/20261031/<backbone>/<variant>/`；图：`newidea/figure/gmr_mr_detr_test_20261030/geq_finetune/<backbone>/`。状态与已完成比较汇总在结果根目录 README。训练损失、检查点选择和对照限制均随实验配置记录。

纯 CPU 离线重绘本分支的已发布数据，不访问服务器路径或加载 checkpoint：

```bash
python -m pip install numpy matplotlib
GMR_GEQ_OUTPUT="$PWD/results/geq_finetune" \
GMR_GEQ_FIGURES="$PWD/results/geq_finetune" \
python shared/geq_finetune/collect_and_plot.py
```

换机训练需提供源 GMR/Mr-DETR 代码、8 个对应 best、原冻结 Test 预测及预计算特征。`GMR_PROJECT_ROOT`、`GMR_BACKBONE_REPO`、`GMR_RUNTIME_ROOT` 分别重定向原项目、骨干仓库和三骨干运行目录；数据加载路径沿用各原生骨干配置，需一并配置。权重和特征不上传。

公开结果包含 12 组配置、真实数据 smoke、四轮历史、原/新完整 Test 预测、Val 预测、官方指标、101 点扫描与最终审计。`audit.py` 的完整权重审计需要服务器源权重和 `final.ckpt`；公开审计 JSON 记录实际服务器上已通过的结果。
