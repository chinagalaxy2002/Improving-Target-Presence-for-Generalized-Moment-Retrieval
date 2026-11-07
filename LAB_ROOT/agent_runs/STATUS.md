# SoccerGMR SI-Bind Research Queue Status
> **Last Updated**: `2026-11-07 14:34:13` | **Active Revision**: `rev_0001` (`061e2a11d2`)

## 1. GPU Slot Allocation & Active Running Jobs
| Slot | Physical GPU | Active Run | Epoch | Stale/Patience | Best Val mAP | Elapsed | Est. Finish |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|
| GPU0-S1 | 0 | *IDLE* | - | - | - | - | - |
| GPU0-S2 | 0 | *IDLE* | - | - | - | - | - |
| GPU0-S3 | 0 | *IDLE* | - | - | - | - | - |
| GPU1-S1 | 1 | `rev_0001_uvcom_E6` | 25 | 3/20 | 347.00% | 88.3 min | ~60.1 min |
| GPU1-S2 | 1 | `rev_0001_uvcom_E4` | 47 | 1/20 | 787.00% | 159.7 min | ~64.5 min |
| GPU1-S3 | 1 | `rev_0001_uvcom_E3` | 46 | 11/20 | 513.00% | 159.7 min | ~31.2 min |

## 2. Completed Experiments & Pareto Frontier (18 done)
| Run ID | Backbone | Variant | Best Epoch | Val mAP | H_all | H@5 | MultiGT Full@5 | Top2 Top1% | Decision |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `rev_0001_moment_detr_E5` | moment_detr | E5 | 7 | 810.00% | 180 | 113 | 4 | 100.0% | **PROMOTE** |
| `rev_0001_moment_detr_E2` | moment_detr | E2 | 52 | 1304.00% | 196 | 145 | 2 | 67.5% | **PROMOTE** |
| `rev_0001_moment_detr_E1` | moment_detr | E1 | 69 | 1357.00% | 190 | 167 | 14 | 74.5% | **CONTROL** |
| `rev_0001_moment_detr_E4` | moment_detr | E4 | 102 | 1446.00% | 218 | 177 | 14 | 54.5% | **INCONCLUSIVE** |
| `rev_0001_moment_detr_E3` | moment_detr | E3 | 73 | 1272.00% | 220 | 167 | 14 | 59.2% | **INCONCLUSIVE** |
| `rev_0001_moment_detr_E6` | moment_detr | E6 | 78 | 1296.00% | 222 | 190 | 19 | 60.0% | **PROMOTE** |
| `rev_0001_moment_detr_E7` | moment_detr | E7 | 78 | 1296.00% | 222 | 190 | 19 | 60.0% | **CONTROL** |
| `rev_0001_uvcom_E1` | uvcom | E1 | 87 | 1708.00% | 290 | 183 | 17 | 57.6% | **CONTROL** |
| `rev_0001_bm_detr_E2` | bm_detr | E2 | 42 | 865.00% | 149 | 84 | 1 | 100.0% | **PROMOTE** |
| `rev_0001_bm_detr_E5` | bm_detr | E5 | 53 | 781.00% | 165 | 107 | 3 | 100.0% | **PROMOTE** |
| `rev_0001_cg_detr_E2` | cg_detr | E2 | 54 | 1400.00% | 237 | 208 | 18 | 60.8% | **PROMOTE** |
| `rev_0001_bm_detr_E4` | bm_detr | E4 | 42 | 666.00% | 171 | 99 | 2 | 100.0% | **INCONCLUSIVE** |
| `rev_0001_bm_detr_E6` | bm_detr | E6 | 70 | 908.00% | 167 | 104 | 4 | 100.0% | **PROMOTE** |
| `rev_0001_bm_detr_E3` | bm_detr | E3 | 83 | 733.00% | 180 | 130 | 6 | 94.5% | **INCONCLUSIVE** |
| `rev_0001_uvcom_E2` | uvcom | E2 | 175 | 2030.00% | 316 | 219 | 30 | 68.2% | **PROMOTE** |
| `rev_0001_uvcom_E5` | uvcom | E5 | 180 | 2193.00% | 329 | 224 | 24 | 61.2% | **PROMOTE** |
| `rev_0001_cg_detr_E5` | cg_detr | E5 | 68 | 1608.00% | 236 | 214 | 29 | 53.3% | **PROMOTE** |
| `rev_0001_bm_detr_E7` | bm_detr | E7 | 70 | 908.00% | 167 | 104 | 4 | 100.0% | **CONTROL** |

## 3. Pending Queue (3 waiting)
1. **cg_detr / E4** (Preferred GPU 1) - H3 Control: Uniform Refresh matching E5 rate
2. **cg_detr / E3** (Preferred GPU 1) - H4: Slot-Private Innovation
3. **cg_detr / E6** (Preferred GPU 1) - H4+H3: E2 + Innovation + Dominance

## 4. Resource & Verification Audit
- **Original Repo & Checkpoint Invariance**: Verified (`audit_integrity.py PASS`)
- **Gate-I Single-Root Write Scope**: Verified (`check_write_scope.py PASS`)
- **Gate-0 Engineering Assertions**: 10/10 Passed (`test_interlayer_bind_gate0.py PASS`)
- **Test Set Partition**: Strictly Frozen (Never used for selection)
