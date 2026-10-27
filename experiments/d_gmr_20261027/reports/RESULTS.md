# D-GMR results

Seed 2026; four training epochs; checkpoint selected by Val mAP, then Val mR@1, then earlier epoch.
Standard Test: 1,036 queries (544 positive / 492 negative); presence threshold 0.4.
Positive-query localization has no presence hard gate. Values are percentages; deltas are percentage points.

| Backbone | Epoch | Val mAP | Original mAP | D-GMR mAP | ΔmAP | mR@1 | mR@5 | mR+@5 | AUROC | Rej-F1 | G-mIoU@1 | G-mIoU@3 | N-acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Moment-DETR | 4 | 9.38 | 6.68 | 9.29 | +2.61 | 4.31 | 15.40 | 1.42 | 70.33 | 0.00 | 5.71 | 2.71 | 0.00 |
| QD-DETR | 4 | 18.15 | 14.62 | 15.21 | +0.59 | 9.59 | 21.59 | 6.40 | 76.84 | 64.48 | 39.37 | 34.67 | 63.82 |
| CG-DETR | 3 | 22.23 | 18.66 | 18.88 | +0.22 | 12.16 | 26.62 | 14.41 | 76.43 | 46.13 | 27.54 | 22.16 | 34.55 |
| EaTR | 3 | 21.42 | 16.49 | 18.30 | +1.81 | 10.99 | 26.39 | 10.64 | 75.04 | 55.49 | 32.49 | 27.74 | 47.76 |
| FlashVTG | 1 | 28.64 | 25.34 | 25.73 | +0.39 | 15.48 | 35.51 | 20.03 | 72.95 | 53.66 | 33.22 | 27.68 | 46.14 |

[All epochs: Val/Test CSV](../results/all_metrics.csv) · [Recomputed metrics](reproduced_metrics.csv)
