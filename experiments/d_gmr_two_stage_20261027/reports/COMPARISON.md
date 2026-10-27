# Four-route Standard Test comparison

Each route retains its own recorded Val-selected checkpoint. Original GMR is the fixed pretrained baseline.
Threshold 0.4; positive localization is ungated; one seed (2026).

| Backbone | Route | mAP | G-mIoU@1 | N-acc | AUROC | Rej-F1 | Positive presence recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Moment-DETR | original_gmr | 6.68 | 5.83 | 0.00 | 70.20 | 0.00 | 100.00 |
| Moment-DETR | d_gmr | 9.29 | 5.71 | 0.00 | 70.33 | 0.00 | 100.00 |
| Moment-DETR | detach | 8.66 | 7.06 | 1.63 | 70.40 | 3.17 | 99.08 |
| Moment-DETR | two_stage | 8.45 | 51.91 | 100.00 | 70.24 | 75.00 | 39.71 |
| QD-DETR | original_gmr | 14.62 | 36.51 | 56.71 | 77.01 | 61.73 | 75.55 |
| QD-DETR | d_gmr | 15.21 | 39.37 | 63.82 | 76.84 | 64.48 | 69.12 |
| QD-DETR | detach | 16.57 | 45.11 | 77.64 | 75.87 | 70.22 | 60.66 |
| QD-DETR | two_stage | 15.65 | 51.50 | 93.09 | 75.90 | 74.17 | 47.61 |
| CG-DETR | original_gmr | 18.66 | 22.84 | 23.17 | 76.40 | 35.46 | 93.20 |
| CG-DETR | d_gmr | 18.88 | 27.54 | 34.55 | 76.43 | 46.13 | 86.21 |
| CG-DETR | detach | 19.29 | 27.85 | 35.16 | 76.62 | 46.69 | 86.03 |
| CG-DETR | two_stage | 18.02 | 45.45 | 77.44 | 76.05 | 69.91 | 60.11 |
| EaTR | original_gmr | 16.49 | 33.41 | 48.58 | 76.56 | 57.11 | 80.51 |
| EaTR | d_gmr | 18.30 | 32.49 | 47.76 | 75.04 | 55.49 | 77.94 |
| EaTR | detach | 17.30 | 32.23 | 47.36 | 75.29 | 55.41 | 78.68 |
| EaTR | two_stage | 16.92 | 44.70 | 75.81 | 75.46 | 69.27 | 61.03 |
| FlashVTG | original_gmr | 25.34 | 28.05 | 32.72 | 74.30 | 44.41 | 86.76 |
| FlashVTG | d_gmr | 25.73 | 33.22 | 46.14 | 72.95 | 53.66 | 76.65 |
| FlashVTG | detach | 26.64 | 36.23 | 52.44 | 74.34 | 57.98 | 74.26 |
| FlashVTG | two_stage | 26.23 | 50.99 | 88.82 | 74.72 | 72.95 | 50.55 |
