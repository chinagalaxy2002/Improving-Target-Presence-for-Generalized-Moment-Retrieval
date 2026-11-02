# SoccerGMR两套Test：完整正式主指标

同一批24个Standard训练best在Standard Test及Full-Decodable Test上各24项，共48条。G-mIoU为0–1；其他指标为%，差值为百分点（G-mIoU差值仍为0–1）。括号一律相对同骨干、同Test的GMR。

[全部指标PDF](tables/ALL_TEST_METRICS.pdf) · [CSV](data/RESULTS.csv) · [JSON](data/RESULTS.json) · [全部增减](data/DELTAS_VS_GMR.csv)

| Test | 骨干 | GMR baseline | GEQ Frozen | GEQ Full | Multi | Multi+GEQ Frozen | Multi+GEQ Full |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard Test (1036/1036) | Moment-DETR | **5.91** | 5.91<br><sub>⚪ (+0.00)</sub> | 6.09<br><sub>🟢 (+0.18)</sub> | 7.27<br><sub>🟢 (+1.36)</sub> | 7.27<br><sub>🟢 (+1.36)</sub> | 7.72<br><sub>🟢 (+1.81)</sub> |
| Standard Test (1036/1036) | EaTR | **16.20** | 16.20<br><sub>⚪ (+0.00)</sub> | 17.15<br><sub>🟢 (+0.95)</sub> | 17.52<br><sub>🟢 (+1.32)</sub> | 17.52<br><sub>🟢 (+1.32)</sub> | 19.88<br><sub>🟢 (+3.68)</sub> |
| Standard Test (1036/1036) | QD-DETR | **15.08** | 15.08<br><sub>⚪ (+0.00)</sub> | 16.54<br><sub>🟢 (+1.46)</sub> | 14.67<br><sub>🔴 (-0.41)</sub> | 14.67<br><sub>🔴 (-0.41)</sub> | 13.71<br><sub>🔴 (-1.37)</sub> |
| Standard Test (1036/1036) | CG-DETR | **16.09** | 16.09<br><sub>⚪ (+0.00)</sub> | 16.58<br><sub>🟢 (+0.49)</sub> | 16.59<br><sub>🟢 (+0.50)</sub> | 16.59<br><sub>🟢 (+0.50)</sub> | 18.59<br><sub>🟢 (+2.50)</sub> |
| Full-Decodable Test (2970/2986) | Moment-DETR | **5.41** | 5.41<br><sub>⚪ (+0.00)</sub> | 5.28<br><sub>🔴 (-0.13)</sub> | 6.06<br><sub>🟢 (+0.65)</sub> | 6.06<br><sub>🟢 (+0.65)</sub> | 6.39<br><sub>🟢 (+0.98)</sub> |
| Full-Decodable Test (2970/2986) | EaTR | **15.24** | 15.24<br><sub>⚪ (+0.00)</sub> | 15.27<br><sub>🟢 (+0.03)</sub> | 16.12<br><sub>🟢 (+0.88)</sub> | 16.12<br><sub>🟢 (+0.88)</sub> | 16.81<br><sub>🟢 (+1.57)</sub> |
| Full-Decodable Test (2970/2986) | QD-DETR | **12.31** | 12.31<br><sub>⚪ (+0.00)</sub> | 13.50<br><sub>🟢 (+1.19)</sub> | 12.04<br><sub>🔴 (-0.27)</sub> | 12.04<br><sub>🔴 (-0.27)</sub> | 11.85<br><sub>🔴 (-0.46)</sub> |
| Full-Decodable Test (2970/2986) | CG-DETR | **13.58** | 13.58<br><sub>⚪ (+0.00)</sub> | 14.01<br><sub>🟢 (+0.43)</sub> | 15.48<br><sub>🟢 (+1.90)</sub> | 15.48<br><sub>🟢 (+1.90)</sub> | 15.93<br><sub>🟢 (+2.35)</sub> |

## Standard Test (1036/1036)

![Standard Test (1036/1036)完整主指标](tables/STANDARD_PRIMARY.png)

[主表PDF](tables/STANDARD_PRIMARY.pdf) · [额外定位指标](tables/STANDARD_LOCALIZATION_EXTRA.pdf) · [额外阈值指标](tables/STANDARD_THRESHOLD_EXTRA.pdf)

![Standard Test (1036/1036)曲线](figures/standard/ALL_BACKBONES.png)

[曲线总览PDF](figures/standard/ALL_BACKBONES.pdf) · [按骨干分页PDF](figures/standard/BY_BACKBONE.pdf)

## Full-Decodable Test (2970/2986)

![Full-Decodable Test (2970/2986)完整主指标](tables/FULL_DECODABLE_PRIMARY.png)

[主表PDF](tables/FULL_DECODABLE_PRIMARY.pdf) · [额外定位指标](tables/FULL_DECODABLE_LOCALIZATION_EXTRA.pdf) · [额外阈值指标](tables/FULL_DECODABLE_THRESHOLD_EXTRA.pdf)

![Full-Decodable Test (2970/2986)曲线](figures/full_decodable/ALL_BACKBONES.png)

[曲线总览PDF](figures/full_decodable/ALL_BACKBONES.pdf) · [按骨干分页PDF](figures/full_decodable/BY_BACKBONE.pdf)
