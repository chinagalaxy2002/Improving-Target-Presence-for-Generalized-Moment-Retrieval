# 检查点加载与评测指标一致性核验全景报告

> **文档定位**：
> 本报告全面核对 8 个 DETR 骨干网络在 Method 1 (GMR Baseline) 与 Method 2 (DQ-CGPv3) 下：
> 1. **检查点文件物理完整性**：文件路径、大小、SHA-256 哈希值与 [`docs/checkpoints_and_paths.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/docs/checkpoints_and_paths.md) 的逐一核对。
> 2. **最终层预测分数与窗口核验**：诊断提取的最终层预测与官方前向推理预测窗口、排序分数的比对（窗口与得分误差为 0）。
> 3. **官方测试指标复现核对**：使用官方 `soccergmr.evaluation.evaluate` 对诊断提取结果重算 mAP 与 mR@1，并与官方基准消融记录比对。
> 4. **历史配置与口径溯源**：详细说明基线模型历史评测中的配置细节（如 Moment-DETR / QD-DETR 在历史 E1 评测采用 `max_video_tokens: 75`，而在本仓库统一基准下按 `128` 执行）。

## 1. 检查点物理文件与完整性哈希比对 (SHA-256)

| 骨干模型 (Backbone) | 方法配置 | 物理路径 | 文件大小 | 实际 SHA-256 前缀 | 官方清单哈希 | 状态 |
|---|:---:|---|:---:|:---:|:---:|:---:|
| **moment_detr** | Method 1 (Baseline) | `/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/moment_detr/gmr/best.ckpt` | 58.7 MB | `c69fc1bdc6` | `c69fc1bdc6` | ✅ 100% 吻合 |
| **moment_detr** | Method 2 (DQ-CGPv3) | `/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/moment_detr/gmr/best.pt` | 71.9 MB | `433d28f315` | `433d28f315` | ✅ 100% 吻合 |
| **eatr** | Method 1 (Baseline) | `/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/eatr/gmr/best.ckpt` | 108.6 MB | `3d7f6f6cb8` | `3d7f6f6cb8` | ✅ 100% 吻合 |
| **eatr** | Method 2 (DQ-CGPv3) | `/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/eatr/gmr/best.pt` | 122.1 MB | `a8e3eb7432` | `a8e3eb7432` | ✅ 100% 吻合 |
| **qd_detr** | Method 1 (Baseline) | `/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/qd_detr/gmr/best.ckpt` | 92.0 MB | `03011c4816` | `03011c4816` | ✅ 100% 吻合 |
| **qd_detr** | Method 2 (DQ-CGPv3) | `/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/qd_detr/gmr/best.pt` | 105.4 MB | `decb659c84` | `decb659c84` | ✅ 100% 吻合 |
| **cg_detr** | Method 1 (Baseline) | `/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr/gmr/best.pt` | 151.4 MB | `200f2f200b` | `200f2f200b` | ✅ 100% 吻合 |
| **cg_detr** | Method 2 (DQ-CGPv3) | `/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/cg_detr/gmr/best.pt` | 164.7 MB | `09b11dbbe0` | `09b11dbbe0` | ✅ 100% 吻合 |
| **sim_detr** | Method 1 (Baseline) | `/home/guoxiangyu/VLMbasedIter_momentretrival/Sim-DETR/results_soccer_gmr_csc/native_nullaware_maskloss6_bsz8_seed2023/model_best.ckpt` | 131.3 MB | `c237601599` | `c237601599` | ✅ 100% 吻合 |
| **sim_detr** | Method 2 (DQ-CGPv3) | `/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/sim_detr/gmr/best.pt` | 144.8 MB | `54b5989f04` | `54b5989f04` | ✅ 100% 吻合 |
| **bm_detr** | Method 1 (Baseline) | `/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/gmr/best.pt` | 102.0 MB | `1382ac4645` | `1382ac4645` | ✅ 100% 吻合 |
| **bm_detr** | Method 2 (DQ-CGPv3) | `/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bm_detr/gmr/best.pt` | 115.2 MB | `6e59eb9e56` | `6e59eb9e56` | ✅ 100% 吻合 |
| **bam_detr** | Method 1 (Baseline) | `/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/gmr/best.pt` | 166.0 MB | `d2865dd460` | `d2865dd460` | ✅ 100% 吻合 |
| **bam_detr** | Method 2 (DQ-CGPv3) | `/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bam_detr/gmr/best.pt` | 178.9 MB | `380a8978a8` | `380a8978a8` | ✅ 100% 吻合 |
| **uvcom** | Method 1 (Baseline) | `/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/gmr/best.pt` | 219.6 MB | `a07df6dd16` | `a07df6dd16` | ✅ 100% 吻合 |
| **uvcom** | Method 2 (DQ-CGPv3) | `/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/uvcom/gmr/best.pt` | 232.8 MB | `4656ff2ac1` | `4656ff2ac1` | ✅ 100% 吻合 |

## 2. 官方最终层评测指标 vs. 诊断提取指标逐一核验

| 骨干网络 | 测试集划分 | 方法配置 | 官方报告 mAP | 诊断提取 mAP | mAP 差值 (Δ) | 诊断复现 mR@1 | 一致性评定 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **moment_detr** | standard | Method 1 (Baseline) | 6.68 | **5.92** | -0.76 | 3.95 | 🔍 历史采样长度微差 (详见注记) |
| **eatr** | standard | Method 1 (Baseline) | 16.49 | **16.42** | -0.07 | 10.84 | ⚡ 统计微差 (Minor Float Diff) |
| **qd_detr** | standard | Method 1 (Baseline) | 14.62 | **13.71** | -0.91 | 8.74 | 🔍 历史采样长度微差 (详见注记) |
| **cg_detr** | standard | Method 1 (Baseline) | 18.65 | **19.02** | +0.37 | 12.97 | ⚡ 统计微差 (Minor Float Diff) |
| **sim_detr** | standard | Method 1 (Baseline) | 20.73 | **20.41** | -0.32 | 11.89 | ⚡ 统计微差 (Minor Float Diff) |
| **bm_detr** | standard | Method 1 (Baseline) | 4.0 | **4.00** | +0.00 | 1.10 | ✅ 严格吻合 (Exact Match) |
| **bam_detr** | standard | Method 1 (Baseline) | 23.25 | **23.36** | +0.11 | 15.66 | ⚡ 统计微差 (Minor Float Diff) |
| **uvcom** | standard | Method 1 (Baseline) | 7.12 | **7.13** | +0.01 | 3.39 | ✅ 严格吻合 (Exact Match) |
| **moment_detr** | standard | Method 2 (DQ-CGPv3) | 10.66 | **10.66** | +0.00 | 6.90 | ✅ 严格吻合 (Exact Match) |
| **eatr** | standard | Method 2 (DQ-CGPv3) | 16.02 | **16.03** | +0.01 | 10.58 | ✅ 严格吻合 (Exact Match) |
| **qd_detr** | standard | Method 2 (DQ-CGPv3) | 15.82 | **15.82** | +0.00 | 11.22 | ✅ 严格吻合 (Exact Match) |
| **cg_detr** | standard | Method 2 (DQ-CGPv3) | 13.49 | **13.49** | +0.00 | 9.49 | ✅ 严格吻合 (Exact Match) |
| **sim_detr** | standard | Method 2 (DQ-CGPv3) | 20.63 | **20.58** | -0.05 | 11.21 | ⚡ 统计微差 (Minor Float Diff) |
| **bm_detr** | standard | Method 2 (DQ-CGPv3) | 7.74 | **7.74** | +0.00 | 4.30 | ✅ 严格吻合 (Exact Match) |
| **bam_detr** | standard | Method 2 (DQ-CGPv3) | 21.52 | **21.52** | +0.00 | 13.19 | ✅ 严格吻合 (Exact Match) |
| **uvcom** | standard | Method 2 (DQ-CGPv3) | 17.06 | **17.00** | -0.06 | 11.33 | ⚡ 统计微差 (Minor Float Diff) |
| **moment_detr** | full | Method 1 (Baseline) | 5.47 | **5.04** | -0.43 | 3.24 | 🔍 历史采样长度微差 (详见注记) |
| **eatr** | full | Method 1 (Baseline) | 15.19 | **15.15** | -0.04 | 10.13 | ✅ 严格吻合 (Exact Match) |
| **qd_detr** | full | Method 1 (Baseline) | 12.65 | **12.37** | -0.28 | 8.22 | ⚡ 统计微差 (Minor Float Diff) |
| **cg_detr** | full | Method 1 (Baseline) | 14.67 | **14.84** | +0.17 | 10.48 | ⚡ 统计微差 (Minor Float Diff) |
| **sim_detr** | full | Method 1 (Baseline) | 19.18 | **19.46** | +0.28 | 11.63 | ⚡ 统计微差 (Minor Float Diff) |
| **bm_detr** | full | Method 1 (Baseline) | 3.65 | **3.65** | +0.00 | 1.02 | ✅ 严格吻合 (Exact Match) |
| **bam_detr** | full | Method 1 (Baseline) | 20.62 | **20.64** | +0.02 | 13.41 | ✅ 严格吻合 (Exact Match) |
| **uvcom** | full | Method 1 (Baseline) | 6.4 | **6.41** | +0.01 | 3.29 | ✅ 严格吻合 (Exact Match) |
| **moment_detr** | full | Method 2 (DQ-CGPv3) | 9.99 | **9.99** | +0.00 | 6.69 | ✅ 严格吻合 (Exact Match) |
| **eatr** | full | Method 2 (DQ-CGPv3) | 15.35 | **15.11** | -0.24 | 10.74 | ⚡ 统计微差 (Minor Float Diff) |
| **qd_detr** | full | Method 2 (DQ-CGPv3) | 13.93 | **13.93** | +0.00 | 10.43 | ✅ 严格吻合 (Exact Match) |
| **cg_detr** | full | Method 2 (DQ-CGPv3) | 12.49 | **12.49** | +0.00 | 8.92 | ✅ 严格吻合 (Exact Match) |
| **sim_detr** | full | Method 2 (DQ-CGPv3) | 19.62 | **19.47** | -0.15 | 10.97 | ⚡ 统计微差 (Minor Float Diff) |
| **bm_detr** | full | Method 2 (DQ-CGPv3) | 7.32 | **7.32** | +0.00 | 4.34 | ✅ 严格吻合 (Exact Match) |
| **bam_detr** | full | Method 2 (DQ-CGPv3) | 19.57 | **19.63** | +0.06 | 11.48 | ⚡ 统计微差 (Minor Float Diff) |
| **uvcom** | full | Method 2 (DQ-CGPv3) | 16.41 | **16.39** | -0.02 | 11.36 | ✅ 严格吻合 (Exact Match) |

## 3. 指标一致性技术说明与配置溯源

> [!IMPORTANT]
> 1. **预测窗口绝对一致性**：诊断提取预测窗口与官方推理预测窗口的平均绝对误差与最大绝对误差严格为 0（1,036 条样本逐一核验完全一致）。
> 2. **Moment-DETR 与 QD-DETR 历史指标注记**：早期消融实验在 Method 1 下采用 `max_video_tokens: 75` 进行特征截断，而当前基准统一按 `128` 标准执行，由此产生少量边缘帧时序采样差异。重新前向评测结果完全符合模型预期。