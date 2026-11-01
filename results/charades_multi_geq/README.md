# Charades GMR + Multi + GEQ (seed 2023)

**Actual completion: 16/24 training tasks; 30/48 formal test evaluations.** This release remains incomplete until the full audit and remote verification pass.

Test-ID is the primary result. Test-OOD is supplementary and keeps all 7,440 rows, including 2,170 duplicate extra records. Tests never select checkpoints or thresholds. All local positive examples have one GT: mR+ and multi-GT coverage are N/A. Flash Multi uses the **Multi-TQAux structural adaptation**, not the same native query decoder as the other backbones.

[Reproduction instructions](../../shared/charades_multi_geq/README.md) · [Protocol](protocol.json) · [Completion audit](audit/completion_audit.json)

## Current Val-selected checkpoints

G-mIoU@1 is the mean at presence thresholds 0.4/0.5/0.6, on a 0–1 scale. mAP is ungated, positive-query localization mAP in percent. In-progress rows are provisional and are not final Test results.

| Backbone | Method | Status | Val mAP % | Val G-mIoU@1 mean |
|---|---|---|---:|---:|
| moment_detr | GMR | Val selection finished | 30.96 | 0.6042 |
| moment_detr | GMR + Multi | Val selection finished | 32.63 | 0.6121 |
| moment_detr | GMR + GEQ Frozen | Val selection finished | 30.96 | 0.6206 |
| moment_detr | GMR + GEQ Full | Val selection finished | 30.42 | 0.6268 |
| moment_detr | GMR + Multi + GEQ Frozen | Val selection finished | 32.63 | 0.6233 |
| moment_detr | GMR + Multi + GEQ Full | Val selection finished | 33.48 | 0.6340 |
| eatr | GMR | Val selection finished | 35.78 | 0.6364 |
| eatr | GMR + Multi | In progress | 27.17 | 0.5024 |
| eatr | GMR + GEQ Frozen | Val selection finished | 35.78 | 0.6364 |
| eatr | GMR + GEQ Full | Val selection finished | 35.59 | 0.6390 |
| eatr | GMR + Multi + GEQ Frozen | Pending | — | — |
| eatr | GMR + Multi + GEQ Full | Pending | — | — |
| qd_detr | GMR | Val selection finished | 35.72 | 0.6465 |
| qd_detr | GMR + Multi | In progress | 34.83 | 0.6296 |
| qd_detr | GMR + GEQ Frozen | Val selection finished | 35.72 | 0.6399 |
| qd_detr | GMR + GEQ Full | Val selection finished | 35.49 | 0.6563 |
| qd_detr | GMR + Multi + GEQ Frozen | Pending | — | — |
| qd_detr | GMR + Multi + GEQ Full | Pending | — | — |
| flashvtg | GMR | Val selection finished | 42.73 | 0.6471 |
| flashvtg | GMR + Multi-TQAux | Val selection finished | 41.85 | 0.6515 |
| flashvtg | GMR + GEQ Frozen | Val selection finished | 42.73 | 0.6577 |
| flashvtg | GMR + GEQ Full | Val selection finished | 42.31 | 0.6598 |
| flashvtg | GMR + Multi-TQAux + GEQ Frozen | In progress | 41.85 | 0.6533 |
| flashvtg | GMR + Multi-TQAux + GEQ Full | In progress | 41.89 | 0.6602 |

## Additional EaTR positive-only ablation

This independently resumed Stage A run is additional to the main24 training tasks and48 formal Test evaluations. It has separate checkpoints and Val history. Its training must finish before final acceptance.

Val-selected mAP: 23.78%; training in progress.


## moment_detr

[All figures, CSV files and offline snapshots](moment_detr/README.md)

![moment_detr Test-ID Frozen](moment_detr/test_id/frozen.png)


## eatr

[All figures, CSV files and offline snapshots](eatr/README.md)

![eatr Val selection](eatr/val/selection.png)


## qd_detr

[All figures, CSV files and offline snapshots](qd_detr/README.md)

![qd_detr Val selection](qd_detr/val/selection.png)


## flashvtg

[All figures, CSV files and offline snapshots](flashvtg/README.md)

![flashvtg Val selection](flashvtg/val/selection.png)

