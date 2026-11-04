# 11-backbone aux-only reuse and evaluation protocol

## Conclusion: no new training is required

The method previously labelled **GMR baseline** in the 11-backbone results is
already the requested **aux-only** ablation:

`native GMR loss + E3 query-aware clip supervision`

where E3 is Balanced BCE plus `0.2 × Dice`, with total auxiliary coefficient
`0.1`. This stage contains no GEQ module, no GEQ BCE, no GEQ logit, and no GEQ
inference branch. Its checkpoints were trained from scratch with seed 2023 and
selected by Standard Val positive-query mAP. Starting another aux-only
fine-tuning stage would be continued auxiliary training, not the requested
ablation.

Therefore the completed GMR `best.pt` files and their existing fixed Test
artifacts must be reused. No completed experiment is retrained and no
multi-seed queue is launched.

## Existing training and Test flow

For each backbone, the completed aux-only flow is:

1. Standard Train from random initialization.
2. Optimize the backbone's native GMR objective plus E3 auxiliary clip loss.
3. Evaluate Standard Val after every epoch.
4. Select `best.pt` by Standard Val positive-query mAP with patience 20 and a
   maximum of 200 epochs.
5. Freeze the selected checkpoint before formal Test.
6. Evaluate the same fixed `best.pt` on Standard Test and Full-Decodable Test.
7. Store predictions, metrics, checkpoint/source/data hashes, and completion
   markers. Test values never select or modify a checkpoint.

The eleven backbones are Moment-DETR, EaTR, QD-DETR, CG-DETR, Sim-DETR,
BM-DETR, BAM-DETR, UVCOM, TR-DETR, TaskWeave, and LD-DETR. Moment-DETR reuses
its already-completed historical E3 checkpoint with its original provenance;
the other ten use the common frozen 11-backbone source.

## Direct evaluation artifacts

The original run root is:

`runs/e3_eleven/formal/<backbone>/gmr/`

The fixed Test artifacts are:

`runs/e3_eleven/formal/<backbone>/gmr_test/test_standard/`

`runs/e3_eleven/formal/<backbone>/gmr_test/test_full/`

The publication copy is under `results/e3_eleven/`, including:

- `METRICS.csv`: all 11 backbones and both stages/splits;
- `predictions/<backbone>/gmr/`: fixed aux-only Test predictions and metrics;
- `provenance/<backbone>/gmr/`: Val history, selection, and completion records;
- `training_val_panel.png/.pdf`: completed training/Val progress;
- `STANDARD_TEST.*` and `FULL_TEST.*`: full threshold curves;
- `RETENTION86_METRICS.csv`: the 86% Val-retention scan.

In current figures, the grey series named **GMR baseline** is this aux-only
method. Adding a second “aux-only” series would duplicate the same values.

## Required direct-test audit

Before reporting the existing results again, perform read-only checks rather
than new training:

1. Verify every `best.pt` hash against its GMR completion marker.
2. Verify source identity, configuration identity, and Train/Val asset hashes.
3. Confirm every checkpoint stage is `gmr` and its model contains the
   query-aware auxiliary head but no GEQ parameters.
4. Verify Standard Test has 1,036 rows and Full-Decodable Test has 2,970 rows.
5. Recompute saved metrics from the hashed predictions and ground truth.
6. Recompute G-mIoU@1/@3 threshold curves without model inference.
7. Confirm the selected checkpoint predates Test evaluation and
   `test_used_for_selection=false`.
8. Reuse the completed training progress panel; no live monitor is necessary
   because all eleven training histories are final.

If a Test artifact is missing or corrupt, rerun **Test inference only** from
the already-selected `best.pt`. Training may be relaunched only if a selected
checkpoint itself fails the hash/provenance audit.

## Interpretation relative to other methods

The intended method names are:

- **GMR no-aux baseline**: native GMR only;
- **Aux only**: the current grey series, native GMR plus E3 supervision;
- **GEQ only**: no-aux GMR followed by GEQ;
- **GEQ+aux**: auxiliary-trained GMR followed by GEQ with auxiliary loss.

The currently imported GEQ-only reference does not share the same Stage-1
parent/protocol for every backbone, so its difference from aux-only is an
outcome-level reference rather than a fully controlled causal ablation. A
strict four-way causal comparison would require a separately frozen no-aux GMR
parent and matched second-stage branches; it must not be fabricated by
renaming or retraining the already-completed aux-only checkpoints.
