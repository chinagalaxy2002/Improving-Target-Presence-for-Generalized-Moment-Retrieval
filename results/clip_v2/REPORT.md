# Clip V2: E1–E6

All runs: seed 2023; GMR-only auxiliary loss; unchanged head-free GEQ. All Val selections frozen before any formal Test.

| Method | Split | GMR mAP | GEQ mAP | GEQ G1 | GEQ G3 |
|---|---|---:|---:|---:|---:|
| E1 | test_standard | 6.71 | 6.86 | 52.49 | 49.16 |
| E1 | test_full | 6.00 | 6.38 | 48.83 | 45.81 |
| E2 | test_standard | 5.91 | 6.63 | 52.39 | 49.12 |
| E2 | test_full | 5.58 | 6.00 | 48.87 | 45.77 |
| E3 | test_standard | 5.49 | 6.33 | 52.45 | 49.14 |
| E3 | test_full | 5.33 | 5.81 | 48.95 | 45.89 |
| E4 | test_standard | 7.02 | 6.54 | 52.61 | 49.20 |
| E4 | test_full | 6.25 | 5.49 | 48.88 | 45.88 |
| E5 | test_standard | 6.74 | 6.18 | 52.64 | 49.21 |
| E5 | test_full | 6.09 | 5.50 | 48.83 | 45.76 |
| E6 | test_standard | 7.87 | 6.03 | 52.65 | 49.21 |
| E6 | test_full | 6.43 | 5.35 | 49.00 | 45.92 |

Old C is a historical reference (Standard/Full GEQ mAP 6.82/6.01; G1 52.60/48.91). Per user instruction it was not retrained under this source. Cross-source/single-seed differences do not establish robust gains.

Clip diagnostics, per-tIoU mAP and short/multi-GT subsets are in DETAILS.json. AUPRC and probability gaps are diagnostics, never checkpoint selection criteria. Short means every GT window ≤10 seconds. Full-Decodable covers 2970/2986 queries; Standard Test covers 1036. The two Test sets overlap.
