# D-GMR two-stage results

Seed 2026; four training epochs; checkpoint selected by Val mAP, then Val mR@1, then earlier epoch.
Standard Test: 1,036 queries (544 positive / 492 negative); presence threshold 0.4.
Positive-query localization has no presence hard gate. Values are percentages; deltas are percentage points.

| Backbone | Epoch | Val mAP | Original mAP | Two-stage mAP | ΔmAP | mR@1 | mR@5 | mR+@5 | AUROC | Rej-F1 | G-mIoU@1 | G-mIoU@3 | N-acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Moment-DETR | 2 | 8.47 | 6.68 | 8.45 | +1.77 | 4.49 | 12.92 | 1.53 | 70.24 | 75.00 | 51.91 | 48.96 | 100.00 |
| QD-DETR | 3 | 19.46 | 14.62 | 15.65 | +1.03 | 9.81 | 22.69 | 8.24 | 75.90 | 74.17 | 51.50 | 47.22 | 93.09 |
| CG-DETR | 3 | 21.21 | 18.66 | 18.02 | -0.64 | 11.60 | 25.14 | 11.61 | 76.05 | 69.91 | 45.45 | 40.57 | 77.44 |
| EaTR | 4 | 20.72 | 16.49 | 16.92 | +0.43 | 10.27 | 24.13 | 9.60 | 75.46 | 69.27 | 44.70 | 40.04 | 75.81 |
| FlashVTG | 2 | 29.56 | 25.34 | 26.23 | +0.89 | 16.09 | 36.60 | 20.03 | 74.72 | 72.95 | 50.99 | 45.88 | 88.82 |

[All epochs: Val/Test CSV](../results/all_metrics.csv) · [Recomputed metrics](reproduced_metrics.csv)
