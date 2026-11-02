# SoccerGMR Standard → Standard / Full Test handoff

Active branch: `soccergmr-standard-geq-full`, based on `3858d7d9f49265f09bfa6c1dd1edbab6f35866a3`.

The user chose to finish refactoring and functional reproduction verification first, archive old results, and defer a full formal retraining campaign. No new training queue or wake-up timer is required for this completed refactor. Do not restart the prior Charades queue or create other agents.

The root README documents only the supported SoccerGMR GMR → GEQ Full pipeline. Charades is available through `archive/charades/README.md`. The exact original README and 2,487 original tracked files were verified against their Git blobs and preserved under `archive/legacy_20261102/`.

Current supported code is in `src/soccergmr`. Five backbones passed real-data mixed/null backward, two-stage training, epoch-boundary interruption/resume versus uninterrupted tensor equality, and Standard/Full Test entry checks in a clean copy without archive. Six regression tests pass. The wheel includes native source; all five CPU GEQ forward checks pass from the wheel. Evidence is in `reports/verification`; review and limits are in `docs/REVIEW.md`.

These are bounded functional tests (7 Train / 8 Val, 8 Standard Test / 4 Full Test), not converged benchmark results. The complete asset audit separately covers 4,138 / 465 / 1,036 / 2,970 queries. Full-Decodable covers 2,970 of 2,986 raw Full queries; report the subset explicitly. New protocol numerical quality and reproduction of archived strong-baseline scores are not yet verified by formal retraining.

Fixed selection: Standard Val positive-query mAP for GMR; mean Standard Val G-mIoU@1 at existence thresholds 0.4/0.5/0.6 for GEQ Full; ten consecutive non-improving epochs for GEQ, ties count. Never select or tune on Test. Old six-patience results retain original labels in archive.

Future formal experiments must use explicit immutable labels/features and fresh output directories. Do not resume legacy checkpoints as new-protocol `last.pt`. Use the README commands and record new metrics only after complete Train/Val-selected runs. The protected local untracked `smoke_build.py` is unrelated and must not be staged.


## Paired historical Test figures added

`results/test_thresholds` now contains five-backbone Standard and Full Test figures comparing GMR with GEQ Full. Twenty complete Test runs use identical checkpoint hashes across the two splits; all ten Standard runs reproduce the historical metrics exactly, and the matching Full runs are newly evaluated. The old example Full figure used different initializations and must not supply values for this paired comparison. These remain historical six-patience checkpoints with legacy 75-token inputs. Full-Decodable coverage is 2970/2986. No training or Test selection occurred. Source scripts, predictions, official metrics, 101-point CSV and oracle diagnostics accompany the figures. The current ten-patience training protocol still has no new converged benchmark results.
