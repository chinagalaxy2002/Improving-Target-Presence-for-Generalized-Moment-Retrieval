# Global Event Query + Hard-RoI: fixed epoch 4

20/20 cells available (5 backbones × 4 configurations). Frozen GMR uses 0 new epochs; all adapted cells use exactly 4 epochs, seed 2026.

LoRA, Gaussian pooling, 6D saliency, candidate presence fusion and synthetic samples are disabled. Metrics are percentages. Primary tau=0.4; tau=0.5/0.6 are fixed sensitivity checks.

## Test, tau=0.4

| Backbone | Configuration | N-acc | AUROC | Rej-F1 | mAP | mR@1 | mR@5 | mR+@5 | G-mIoU@1 | G-mIoU@3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Moment-DETR | GMR | 0.00 | 70.20 | 0.00 | 6.68 | 4.64 | 9.44 | 0.88 | 5.83 | 2.43 |
| Moment-DETR | + Global Event Query | 100.00 | 70.33 | 75.00 | 6.68 | 4.64 | 9.44 | 0.88 | 52.02 | 49.00 |
| Moment-DETR | + Hard-RoI | 0.00 | 70.20 | 0.00 | 5.59 | 3.27 | 7.87 | 1.27 | 6.01 | 2.60 |
| Moment-DETR | + Global Event Query + Hard-RoI | 100.00 | 70.22 | 75.00 | 5.38 | 3.18 | 7.54 | 1.30 | 52.22 | 49.07 |
| QD-DETR | GMR | 56.71 | 77.01 | 61.73 | 14.62 | 9.47 | 20.17 | 7.43 | 36.51 | 31.47 |
| QD-DETR | + Global Event Query | 80.28 | 76.44 | 71.11 | 14.62 | 9.47 | 20.17 | 7.43 | 46.60 | 41.78 |
| QD-DETR | + Hard-RoI | 56.71 | 77.01 | 61.73 | 15.22 | 9.36 | 22.67 | 8.17 | 36.59 | 31.51 |
| QD-DETR | + Global Event Query + Hard-RoI | 81.91 | 76.26 | 71.84 | 14.99 | 9.18 | 21.73 | 7.60 | 47.32 | 42.52 |
| CG-DETR | GMR | 23.17 | 76.40 | 35.46 | 18.66 | 12.65 | 24.90 | 11.98 | 22.84 | 17.07 |
| CG-DETR | + Global Event Query | 89.23 | 76.70 | 73.78 | 18.66 | 12.65 | 24.90 | 11.98 | 50.55 | 45.67 |
| CG-DETR | + Hard-RoI | 23.17 | 76.40 | 35.46 | 19.02 | 12.60 | 25.52 | 12.41 | 22.95 | 17.22 |
| CG-DETR | + Global Event Query + Hard-RoI | 86.18 | 76.73 | 72.60 | 19.05 | 12.66 | 25.49 | 12.88 | 49.25 | 44.38 |
| EaTR | GMR | 48.58 | 76.56 | 57.11 | 16.22 | 10.64 | 21.42 | 9.59 | 33.32 | 28.34 |
| EaTR | + Global Event Query | 82.52 | 75.73 | 71.35 | 16.22 | 10.64 | 21.42 | 9.59 | 47.14 | 42.52 |
| EaTR | + Hard-RoI | 48.58 | 76.56 | 57.11 | 15.86 | 10.35 | 21.28 | 10.48 | 33.28 | 28.28 |
| EaTR | + Global Event Query + Hard-RoI | 82.32 | 75.91 | 71.55 | 16.06 | 10.47 | 21.46 | 10.92 | 47.09 | 42.48 |
| FlashVTG | GMR | 32.72 | 74.30 | 44.41 | 25.43 | 15.17 | 34.94 | 17.86 | 28.10 | 21.75 |
| FlashVTG | + Global Event Query | 88.01 | 75.26 | 73.08 | 25.40 | 15.11 | 34.94 | 17.86 | 50.48 | 45.43 |
| FlashVTG | + Hard-RoI | 32.72 | 74.30 | 44.41 | 23.83 | 13.56 | 33.30 | 17.27 | 27.26 | 21.36 |
| FlashVTG | + Global Event Query + Hard-RoI | 90.65 | 75.29 | 73.72 | 24.20 | 14.01 | 33.62 | 17.08 | 51.60 | 46.57 |

## Val, tau=0.4

| Backbone | Configuration | N-acc | AUROC | Rej-F1 | mAP | mR@1 | mR@5 | mR+@5 | G-mIoU@1 | G-mIoU@3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Moment-DETR | GMR | 0.00 | 69.21 | 0.00 | 7.35 | 4.65 | 10.86 | 2.17 | 6.08 | 2.60 |
| Moment-DETR | + Global Event Query | 100.00 | 70.59 | 70.71 | 7.35 | 4.65 | 10.86 | 2.17 | 48.68 | 46.33 |
| Moment-DETR | + Hard-RoI | 0.00 | 69.21 | 0.00 | 6.71 | 3.97 | 9.84 | 2.33 | 6.22 | 2.69 |
| Moment-DETR | + Global Event Query + Hard-RoI | 100.00 | 70.76 | 70.71 | 6.73 | 3.90 | 9.84 | 2.33 | 48.70 | 46.34 |
| QD-DETR | GMR | 56.19 | 74.15 | 59.60 | 19.60 | 13.30 | 26.09 | 10.22 | 36.21 | 30.79 |
| QD-DETR | + Global Event Query | 81.90 | 73.81 | 68.39 | 19.60 | 13.30 | 26.09 | 10.22 | 45.99 | 40.98 |
| QD-DETR | + Hard-RoI | 56.19 | 74.15 | 59.60 | 19.02 | 12.47 | 26.23 | 9.48 | 36.25 | 30.74 |
| QD-DETR | + Global Event Query + Hard-RoI | 82.38 | 73.56 | 68.11 | 18.82 | 12.20 | 25.64 | 9.48 | 45.99 | 41.09 |
| CG-DETR | GMR | 19.52 | 74.74 | 30.71 | 20.98 | 14.08 | 28.47 | 11.76 | 22.34 | 15.85 |
| CG-DETR | + Global Event Query | 89.52 | 73.79 | 70.54 | 20.98 | 14.08 | 28.47 | 11.76 | 48.46 | 43.92 |
| CG-DETR | + Hard-RoI | 19.52 | 74.74 | 30.71 | 21.56 | 14.65 | 29.03 | 11.69 | 22.48 | 15.90 |
| CG-DETR | + Global Event Query + Hard-RoI | 87.62 | 73.92 | 69.83 | 21.17 | 14.10 | 28.96 | 12.07 | 47.66 | 43.11 |
| EaTR | GMR | 50.95 | 75.63 | 57.84 | 20.26 | 13.78 | 26.75 | 12.44 | 34.76 | 28.83 |
| EaTR | + Global Event Query | 85.24 | 72.46 | 68.45 | 20.26 | 13.78 | 26.75 | 12.44 | 47.11 | 42.38 |
| EaTR | + Hard-RoI | 50.95 | 75.63 | 57.84 | 19.01 | 13.17 | 25.27 | 10.91 | 34.46 | 28.72 |
| EaTR | + Global Event Query + Hard-RoI | 84.29 | 72.68 | 68.08 | 19.53 | 13.52 | 26.12 | 11.69 | 46.63 | 41.87 |
| FlashVTG | GMR | 36.67 | 74.89 | 48.12 | 29.16 | 18.77 | 38.28 | 14.74 | 31.02 | 23.61 |
| FlashVTG | + Global Event Query | 88.10 | 72.23 | 69.68 | 28.96 | 18.46 | 38.28 | 14.74 | 49.56 | 43.86 |
| FlashVTG | + Hard-RoI | 36.67 | 74.89 | 48.12 | 27.08 | 17.47 | 36.51 | 13.91 | 30.47 | 23.22 |
| FlashVTG | + Global Event Query + Hard-RoI | 89.05 | 72.09 | 69.39 | 27.23 | 17.73 | 36.63 | 14.24 | 49.95 | 44.13 |

The baseline and Global-only cells reuse verified matching historical runs; the 10 Hard-only / Global+Hard cells are newly trained. All cells start independently from the same pretrained GMR checkpoint within each backbone.

Hard-only retains native GMR presence. Original GMR tensors stay frozen. Hard-only all-negative batches have no boundary target and perform no optimizer update. No test-based checkpoint or threshold selection; these are exploratory single-seed results on previously observed Standard labels.

FlashVTG retains its original soft presence confidence calibration, rounding and NMS. Its final candidate confidence ordering can depend on presence even with identical raw localizer proposals.

[All metrics](all_metrics.csv) · [GMR-relative gains, pp](gains_vs_gmr_pp.csv) · [Conditional effects and interaction](factorial_effects_pp.csv)
