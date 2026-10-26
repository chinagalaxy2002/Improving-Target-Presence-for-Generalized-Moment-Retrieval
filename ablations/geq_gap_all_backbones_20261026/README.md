# GAP 固定配置迁移：五 backbone

固定迁移 FlashVTG 已按 Val 选定的 GAP margin=0.6、λ=2；不在其他 backbone 上重新搜索。Moment-DETR、QD-DETR、CG-DETR、EaTR 各从自己原始预训练 GMR checkpoint 独立微调4轮，seed2026，native batch size 8/32/32/32。FlashVTG 复用 GAP_m60_w200 第4轮。

总损失：native GMR loss + GEQ BCE + 2*[max(0,0.6−(μ+−μ−))]^2。保留各 backbone 原生训练目标和 native presence BCE，不使用 scratch adapter，不加PV/LoRA/refiner/融合gate。全量更新GMR/GEQ；新optimizer，GMR lr1e-5、GEQ lr3e-5、AdamW wd1e-4、clip0.1、constant scheduler、无AMP。外部离线特征提取器不在计算图中。

`run.py`复用现有原生Adapter、JointGMR和原生损失，检查与对应E1初始张量完全一致；preflight包含GAP公式/梯度方向、单类/混合/多GT/native batch更新、GEQ/Transformer梯度、原定位/native existence监督、checkpoint严格重建及官方预测一致。CG-DETR保留targets传入训练forward及其原生损失；EaTR保留原event/moment匹配。禁止修改旧实验代码/结果。

`launch.py --launch`启动4个后台工作进程，两张GPU各2个；每组先preflight，再重新初始化正式训练。第4轮自动官方Val/Test与独立复算；全部成功后report.py计算五backbone×三方法的全split均分、差距、端点分数与原生性能指标并制图。以Val/Test原始预测按qid严格对齐，分别在正负类内求均值，不按阈值筛样本。

最终报告：reports/RESULTS.md；精确均分：reports/existence_score_means.csv；完整阈值0.4/0.5/0.6指标：reports/all_metrics.csv；图：figures/{val,test}_mean_gap_comparison.{pdf,svg,png}、对应五行两列分布图和性能比较图。GAP与E1初始权重/数据顺序一致。S=50*(1+mean_gap)，不视为独立证据。单seed，不声称显著性或跨seed稳定性。checkpoint保留本地，不自动上传。


## 结果解释

Test 上，GAP相对E1在五个backbone均扩大正负均分差距，增加约0.0235–0.0417。五个backbone的正均分均降低，但负均分下降更多。因此结论是均分间隔扩大，不能声称同时抬高正分、压低负分。新增四个backbone的AUROC相对E1均下降（约0.07–0.26个百分点），AUPRC也均略降；FlashVTG的AUROC几乎不变。纯定位mAP在Moment-DETR/QD-DETR小幅提高，在CG-DETR/EaTR/FlashVTG下降；mR+@5仅FlashVTG提高。故当前单seed固定4轮结果不支持GAP稳定改善全局排序或多时刻定位。


## 已完成与图表复现

四个新训练已全部完成4轮，官方Val/Test及独立复算通过。均分、分布和性能图均提供PDF/SVG/PNG。运行 `python ablations/geq_gap_all_backbones_20261026/reproduce_figures.py` 可从压缩保存的真实分数与GT重画，不需要权重或GPU。预检查首次因沿用FlashVTG定位头层名而停止，失败记录保留本地attempts/；修正为各backbone实际定位头后，四组全部重新preflight并从原始checkpoint正式训练，无失败smoke权重进入训练。
