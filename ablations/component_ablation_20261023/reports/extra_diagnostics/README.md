# Additional fusion diagnostics (not formal component ablations)

5 / 5 additional fusion-enabled runs completed. These row-7 execution artifacts are separate from the formal 5 x 6 = 30 component experiments. Compare them with the already-counted formal full/minusFusion configuration (row 6); the comparison does not add a seventh formal configuration.

| Backbone | Configuration | Split | N-acc | Rej-F1 | mAP | mR+@5 | G-mIoU@1 |
|---|---|---|---:|---:|---:|---:|---:|
| cg_detr | Full + candidate presence fusion control | val | 88.10 | 69.94 | 21.87 | 12.65 | 48.10 |
| cg_detr | Full + candidate presence fusion control | test | 86.79 | 72.56 | 19.27 | 12.05 | 49.63 |
| eatr | Full + candidate presence fusion control | val | 89.05 | 69.00 | 19.11 | 11.11 | 48.06 |
| eatr | Full + candidate presence fusion control | test | 88.01 | 72.96 | 17.49 | 9.72 | 49.65 |
| flashvtg | Full + candidate presence fusion control | val | 87.14 | 68.80 | 27.54 | 16.30 | 48.62 |
| flashvtg | Full + candidate presence fusion control | test | 89.02 | 73.43 | 26.21 | 19.00 | 51.34 |
| moment_detr | Full + candidate presence fusion control | val | 100.00 | 70.71 | 6.90 | 0.39 | 48.21 |
| moment_detr | Full + candidate presence fusion control | test | 100.00 | 75.00 | 7.32 | 0.72 | 51.78 |
| qd_detr | Full + candidate presence fusion control | val | 84.76 | 68.99 | 17.79 | 7.59 | 46.55 |
| qd_detr | Full + candidate presence fusion control | test | 83.13 | 71.88 | 16.10 | 10.37 | 47.75 |

[Fusion-enabled absolute metrics](fusion_enabled_metrics.csv) | [minusFusion minus fused deltas](minusFusion_control_pp.csv)
