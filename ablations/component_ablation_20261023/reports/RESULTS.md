# Formal component ablation: 5 backbones x 6 configurations

30 / 30 formal experiments completed. Only configurations 1--6 enter this table and the formal CSV. Metrics are percentages; primary tau=0.4; 0.5/0.6 are sensitivity analyses. Fixed epoch 4, seed 2026; frozen baseline.

| Backbone | Configuration | Split | N-acc | Rej-F1 | mAP | mR+@5 | G-mIoU@1 |
|---|---|---|---:|---:|---:|---:|---:|
| cg_detr | Coupled GMR Adapter | val | 19.52 | 30.71 | 20.98 | 11.76 | 22.34 |
| cg_detr | Coupled GMR Adapter | test | 23.17 | 35.46 | 18.66 | 11.98 | 22.84 |
| cg_detr | + Decoupled Global Event Query | val | 89.52 | 70.54 | 20.98 | 11.76 | 48.46 |
| cg_detr | + Decoupled Global Event Query | test | 89.23 | 73.78 | 18.66 | 11.98 | 50.55 |
| cg_detr | + LoRA | val | 89.52 | 70.54 | 20.65 | 12.31 | 48.39 |
| cg_detr | + LoRA | test | 87.80 | 72.97 | 18.34 | 11.88 | 49.69 |
| cg_detr | + Hard-RoI Refiner (without saliency) | val | 89.52 | 70.54 | 21.91 | 14.15 | 48.68 |
| cg_detr | + Hard-RoI Refiner (without saliency) | test | 87.80 | 73.03 | 18.93 | 13.10 | 50.02 |
| cg_detr | + Gaussian-RoI Refiner (without saliency) | val | 84.29 | 69.01 | 18.61 | 10.39 | 45.87 |
| cg_detr | + Gaussian-RoI Refiner (without saliency) | test | 84.35 | 72.11 | 20.36 | 12.36 | 48.84 |
| cg_detr | Full DeRoI-GMR (minusFusion) | val | 90.95 | 71.27 | 21.16 | 12.98 | 49.28 |
| cg_detr | Full DeRoI-GMR (minusFusion) | test | 89.02 | 73.55 | 19.33 | 11.87 | 50.52 |
| eatr | Coupled GMR Adapter | val | 50.95 | 57.84 | 20.26 | 12.44 | 34.76 |
| eatr | Coupled GMR Adapter | test | 48.58 | 57.11 | 16.22 | 9.59 | 33.32 |
| eatr | + Decoupled Global Event Query | val | 85.24 | 68.45 | 20.26 | 12.44 | 47.11 |
| eatr | + Decoupled Global Event Query | test | 82.52 | 71.35 | 16.22 | 9.59 | 47.14 |
| eatr | + LoRA | val | 86.67 | 68.94 | 20.66 | 12.89 | 48.07 |
| eatr | + LoRA | test | 83.94 | 71.76 | 16.45 | 9.37 | 47.78 |
| eatr | + Hard-RoI Refiner (without saliency) | val | 87.14 | 68.93 | 19.32 | 10.44 | 48.05 |
| eatr | + Hard-RoI Refiner (without saliency) | test | 85.98 | 72.12 | 17.16 | 9.55 | 48.71 |
| eatr | + Gaussian-RoI Refiner (without saliency) | val | 86.19 | 68.56 | 19.41 | 10.89 | 47.57 |
| eatr | + Gaussian-RoI Refiner (without saliency) | test | 85.98 | 72.37 | 16.25 | 10.21 | 48.49 |
| eatr | Full DeRoI-GMR (minusFusion) | val | 85.71 | 68.44 | 19.54 | 11.44 | 47.22 |
| eatr | Full DeRoI-GMR (minusFusion) | test | 85.37 | 72.29 | 17.01 | 9.63 | 48.58 |
| flashvtg | Coupled GMR Adapter | val | 36.67 | 48.12 | 29.16 | 14.74 | 31.02 |
| flashvtg | Coupled GMR Adapter | test | 32.72 | 44.41 | 25.43 | 17.86 | 28.10 |
| flashvtg | + Decoupled Global Event Query | val | 88.10 | 69.68 | 28.96 | 14.74 | 49.56 |
| flashvtg | + Decoupled Global Event Query | test | 88.01 | 73.08 | 25.40 | 17.86 | 50.48 |
| flashvtg | + LoRA | val | 88.10 | 69.42 | 28.59 | 14.59 | 49.36 |
| flashvtg | + LoRA | test | 88.82 | 73.45 | 25.42 | 17.54 | 50.80 |
| flashvtg | + Hard-RoI Refiner (without saliency) | val | 89.05 | 69.39 | 27.79 | 13.70 | 49.49 |
| flashvtg | + Hard-RoI Refiner (without saliency) | test | 90.04 | 73.34 | 26.24 | 18.27 | 51.67 |
| flashvtg | + Gaussian-RoI Refiner (without saliency) | val | 89.05 | 69.39 | 27.98 | 15.96 | 49.59 |
| flashvtg | + Gaussian-RoI Refiner (without saliency) | test | 90.45 | 73.55 | 26.29 | 19.05 | 52.01 |
| flashvtg | Full DeRoI-GMR (minusFusion) | val | 89.05 | 69.39 | 28.07 | 15.43 | 49.69 |
| flashvtg | Full DeRoI-GMR (minusFusion) | test | 90.24 | 73.51 | 26.13 | 18.46 | 51.84 |
| moment_detr | Coupled GMR Adapter | val | 0.00 | 0.00 | 7.35 | 2.17 | 6.08 |
| moment_detr | Coupled GMR Adapter | test | 0.00 | 0.00 | 6.68 | 0.88 | 5.83 |
| moment_detr | + Decoupled Global Event Query | val | 100.00 | 70.71 | 7.35 | 2.17 | 48.68 |
| moment_detr | + Decoupled Global Event Query | test | 100.00 | 75.00 | 6.68 | 0.88 | 52.02 |
| moment_detr | + LoRA | val | 100.00 | 70.71 | 7.94 | 2.17 | 48.91 |
| moment_detr | + LoRA | test | 100.00 | 75.00 | 5.87 | 0.91 | 51.78 |
| moment_detr | + Hard-RoI Refiner (without saliency) | val | 100.00 | 70.71 | 6.41 | 0.44 | 48.76 |
| moment_detr | + Hard-RoI Refiner (without saliency) | test | 100.00 | 75.00 | 7.35 | 0.76 | 52.34 |
| moment_detr | + Gaussian-RoI Refiner (without saliency) | val | 100.00 | 70.71 | 7.41 | 0.28 | 48.61 |
| moment_detr | + Gaussian-RoI Refiner (without saliency) | test | 100.00 | 75.00 | 7.98 | 0.79 | 52.32 |
| moment_detr | Full DeRoI-GMR (minusFusion) | val | 100.00 | 70.71 | 7.01 | 0.28 | 48.70 |
| moment_detr | Full DeRoI-GMR (minusFusion) | test | 100.00 | 75.00 | 7.59 | 0.98 | 51.92 |
| qd_detr | Coupled GMR Adapter | val | 56.19 | 59.60 | 19.60 | 10.22 | 36.21 |
| qd_detr | Coupled GMR Adapter | test | 56.71 | 61.73 | 14.62 | 7.43 | 36.51 |
| qd_detr | + Decoupled Global Event Query | val | 81.90 | 68.39 | 19.60 | 10.22 | 45.99 |
| qd_detr | + Decoupled Global Event Query | test | 80.28 | 71.11 | 14.62 | 7.43 | 46.60 |
| qd_detr | + LoRA | val | 81.43 | 68.40 | 17.79 | 11.48 | 45.11 |
| qd_detr | + LoRA | test | 80.49 | 71.22 | 13.71 | 6.54 | 46.00 |
| qd_detr | + Hard-RoI Refiner (without saliency) | val | 85.71 | 69.36 | 18.31 | 9.78 | 47.37 |
| qd_detr | + Hard-RoI Refiner (without saliency) | test | 84.35 | 72.43 | 16.49 | 11.11 | 48.38 |
| qd_detr | + Gaussian-RoI Refiner (without saliency) | val | 85.71 | 69.50 | 17.89 | 6.93 | 47.01 |
| qd_detr | + Gaussian-RoI Refiner (without saliency) | test | 84.35 | 72.49 | 15.83 | 11.82 | 48.27 |
| qd_detr | Full DeRoI-GMR (minusFusion) | val | 83.81 | 69.02 | 18.02 | 7.54 | 45.83 |
| qd_detr | Full DeRoI-GMR (minusFusion) | test | 81.50 | 71.54 | 16.18 | 12.26 | 47.35 |

[Additional fusion diagnostics, outside the formal 30](extra_diagnostics/README.md)
