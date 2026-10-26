# GAP 五backbone迁移实测

## GAP 相对 E1 的 Test 变化

| Backbone | Δ均分差距 | ΔS | ΔAUROC | ΔmAP | ΔmR+@5 |
|---|---:|---:|---:|---:|---:|
| Moment-DETR | +0.0300 | +1.50 | -0.26 | +0.25 | -0.67 |
| QD-DETR | +0.0389 | +1.95 | -0.24 | +0.28 | -0.17 |
| CG-DETR | +0.0245 | +1.22 | -0.07 | -0.58 | -1.12 |
| EaTR | +0.0417 | +2.09 | -0.09 | -0.80 | -1.76 |
| FlashVTG | +0.0235 | +1.17 | +0.00 | -0.34 | +1.75 |

均分差距与S分别为概率差和百分制；S=50*(1+均分差距)，不是两项独立证据。各类均值在整个split计算，不平均batch均值。存在性分类固定阈值0.4；0.5/0.6另存CSV。定位不经过存在性hard gating，G-mIoU与纯定位分开解释。单seed，不报告统计显著性。

图：../figures/test_mean_gap_comparison.pdf；../figures/test_existence_score_distributions.pdf；../figures/test_performance_comparison.pdf；Val有对应图。


## 结果解释

Test 上，GAP相对E1在五个backbone均扩大正负均分差距，增加约0.0235–0.0417。五个backbone的正均分均降低，但负均分下降更多。因此结论是均分间隔扩大，不能声称同时抬高正分、压低负分。新增四个backbone的AUROC相对E1均下降（约0.07–0.26个百分点），AUPRC也均略降；FlashVTG的AUROC几乎不变。纯定位mAP在Moment-DETR/QD-DETR小幅提高，在CG-DETR/EaTR/FlashVTG下降；mR+@5仅FlashVTG提高。故当前单seed固定4轮结果不支持GAP稳定改善全局排序或多时刻定位。
