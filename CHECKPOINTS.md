# 11骨干四方法 Best 检查点目录与溯源清单

> **文件生成时间**：2026-11-04  
> **数据范围**：4 种方法 × 11 个骨干 = 44 个模型检查点  
> **核查状态**：全部 44 个检查点均在本地磁盘物理存在，且 SHA256 哈希与历史 Manifest/Evidence 记录 100% 匹配；全部确认为对应训练过程中的官方 **`best`** 检查点（严格按 Standard Val 选模后冻结，Test 严禁反向参与选模）。

## 一、按方法分类检查点清单

### Method 1: GMR baseline

- **阶段与配置**：原生 GMR（无 aux，无 GEQ，Stage 1 父模型）

| 骨干名称 | Best 检查点文件名 | 大小 | 选模指标 | 最佳轮次 (Epoch) | 最佳验证分数 | 完整物理路径 | SHA256 前缀 |
|---|---|---|---|---|---|---|---|
| **Moment-DETR** | `best.ckpt` | 56.0 MB | Val mAP | 66 | 7.35 | [`/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/moment_detr/gmr/best.ckpt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/moment_detr/gmr/best.ckpt) | `c69fc1bdc6...` |
| **EaTR** | `best.ckpt` | 103.6 MB | Val mAP | 历史最佳 | 19.34 | [`/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/eatr/gmr/best.ckpt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/eatr/gmr/best.ckpt) | `3d7f6f6cb8...` |
| **QD-DETR** | `best.ckpt` | 87.7 MB | Val mAP | 历史最佳 | 17.51 | [`/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/qd_detr/gmr/best.ckpt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/qd_detr/gmr/best.ckpt) | `03011c4816...` |
| **CG-DETR** | `best.pt` | 144.4 MB | Val mAP | 历史最佳 | 8.65 | [`/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr/gmr/best.pt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr/gmr/best.pt) | `200f2f200b...` |
| **Sim-DETR** | `model_best.ckpt` | 125.3 MB | Val mAP | 历史最佳 | 22.84 | [`/home/guoxiangyu/VLMbasedIter_momentretrival/Sim-DETR/results_soccer_gmr_csc/native_nullaware_maskloss6_bsz8_seed2023/model_best.ckpt`](file:///home/guoxiangyu/VLMbasedIter_momentretrival/Sim-DETR/results_soccer_gmr_csc/native_nullaware_maskloss6_bsz8_seed2023/model_best.ckpt) | `c237601599...` |
| **BM-DETR** | `best.pt` | 97.2 MB | Val mAP | 17 | 3.61 | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/gmr/best.pt) | `1382ac4645...` |
| **BAM-DETR** | `best.pt` | 158.3 MB | Val mAP | 72 | 25.49 | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/gmr/best.pt) | `d2865dd460...` |
| **UVCOM** | `best.pt` | 209.4 MB | Val mAP | 55 | 8.77 | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/gmr/best.pt) | `a07df6dd16...` |
| **TR-DETR** | `best.pt` | 129.5 MB | Val mAP | 73 | 3.90 | [`/home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/tr_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/tr_detr/gmr/best.pt) | `ed31af7256...` |
| **TaskWeave** | `best.pt` | 114.2 MB | Val mAP | 8 | 3.46 | [`/home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/taskweave/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/taskweave/gmr/best.pt) | `9d74a13432...` |
| **LD-DETR** | `best.pt` | 269.4 MB | Val mAP | 28 | 7.44 | [`/home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/ld_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/ld_detr/gmr/best.pt) | `201c61fd30...` |

### Method 2: GMR(aux)

- **阶段与配置**：原生 GMR + E3 query-aware 辅助监督（无 GEQ，Stage 1）

| 骨干名称 | Best 检查点文件名 | 大小 | 选模指标 | 最佳轮次 (Epoch) | 最佳验证分数 | 完整物理路径 | SHA256 前缀 |
|---|---|---|---|---|---|---|---|
| **Moment-DETR** | `best.pt` | 58.3 MB | Val mAP | 3 | 7.06 | [`/home/guoxiangyu/SoccerGMR-GEQ-clip-v2/runs/clip_v2/E3/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-clip-v2/runs/clip_v2/E3/gmr/best.pt) | `e218b275c9...` |
| **EaTR** | `best.pt` | 106.2 MB | Val mAP | 73 | 18.88 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/eatr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/eatr/gmr/best.pt) | `d4ab38c435...` |
| **QD-DETR** | `best.pt` | 90.3 MB | Val mAP | 76 | 16.26 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/qd_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/qd_detr/gmr/best.pt) | `86a469890d...` |
| **CG-DETR** | `best.pt` | 146.7 MB | Val mAP | 13 | 8.05 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/cg_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/cg_detr/gmr/best.pt) | `5db97be960...` |
| **Sim-DETR** | `best.pt` | 128.0 MB | Val mAP | 109 | 22.16 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/sim_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/sim_detr/gmr/best.pt) | `d087909055...` |
| **BM-DETR** | `best.pt` | 99.5 MB | Val mAP | 11 | 4.31 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bm_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bm_detr/gmr/best.pt) | `9cc4eb9701...` |
| **BAM-DETR** | `best.pt` | 160.5 MB | Val mAP | 65 | 23.83 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bam_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bam_detr/gmr/best.pt) | `9f36d01677...` |
| **UVCOM** | `best.pt` | 212.1 MB | Val mAP | 165 | 21.48 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/uvcom/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/uvcom/gmr/best.pt) | `24267fc515...` |
| **TR-DETR** | `best.pt` | 131.7 MB | Val mAP | 57 | 3.75 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/tr_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/tr_detr/gmr/best.pt) | `4d259f310f...` |
| **TaskWeave** | `best.pt` | 116.5 MB | Val mAP | 17 | 4.34 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/taskweave/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/taskweave/gmr/best.pt) | `98d94d8b3c...` |
| **LD-DETR** | `best.pt` | 271.7 MB | Val mAP | 19 | 6.91 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/ld_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/ld_detr/gmr/best.pt) | `0cf456d2ab...` |

### Method 3: GEQ only (reference)

- **阶段与配置**：原生 GMR + 原始 GEQ（无 aux，Stage 2 参考）

| 骨干名称 | Best 检查点文件名 | 大小 | 选模指标 | 最佳轮次 (Epoch) | 最佳验证分数 | 完整物理路径 | SHA256 前缀 |
|---|---|---|---|---|---|---|---|
| **Moment-DETR** | `best.ckpt` | 24.7 MB | Val mean G1 | 9 | 48.91 | [`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/moment_detr/full/best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/moment_detr/full/best.ckpt) | `06145a116f...` |
| **EaTR** | `best.ckpt` | 42.6 MB | Val mean G1 | 4 | 51.52 | [`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/eatr/full/best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/eatr/full/best.ckpt) | `b1a6cd36da...` |
| **QD-DETR** | `best.ckpt` | 35.3 MB | Val mean G1 | 4 | 51.57 | [`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/qd_detr/full/best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/qd_detr/full/best.ckpt) | `144d36e9c0...` |
| **CG-DETR** | `best.ckpt` | 56.5 MB | Val mean G1 | 1 | 49.33 | [`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/cg_detr/full/best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/cg_detr/full/best.ckpt) | `d06de66b24...` |
| **Sim-DETR** | `best.ckpt` | 48.7 MB | Val mean G1 | 4 | 51.98 | [`/home/guoxiangyu/simdetr_gmr_multiroute_geq/runs/geq_baseline_full/best.ckpt`](file:///home/guoxiangyu/simdetr_gmr_multiroute_geq/runs/geq_baseline_full/best.ckpt) | `931bc5d075...` |
| **BM-DETR** | `best.pt` | 121.4 MB | Val mean G1 | 16 | 49.14 | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/geq_full/best.pt) | `0dac373364...` |
| **BAM-DETR** | `best.pt` | 176.2 MB | Val mean G1 | 4 | 49.01 | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/geq_full/best.pt) | `2b90dc19a4...` |
| **UVCOM** | `best.pt` | 233.4 MB | Val mean G1 | 4 | 50.28 | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/geq_full/best.pt) | `e474e04f71...` |
| **TR-DETR** | `best.pt` | 153.4 MB | Val mean G1 | 4 | 46.72 | [`/home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/tr_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/tr_detr/geq_full/best.pt) | `031303aaf8...` |
| **TaskWeave** | `best.pt` | 132.4 MB | Val mean G1 | 10 | 47.04 | [`/home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/taskweave/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/taskweave/geq_full/best.pt) | `dab8e54656...` |
| **LD-DETR** | `best.pt` | 299.6 MB | Val mean G1 | 4 | 48.84 | [`/home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/ld_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/ld_detr/geq_full/best.pt) | `4a42f2ed0a...` |

### Method 4: GEQ+aux

- **阶段与配置**：原生 GMR + E3 辅助监督 + GEQ（Stage 2 完整模型）

| 骨干名称 | Best 检查点文件名 | 大小 | 选模指标 | 最佳轮次 (Epoch) | 最佳验证分数 | 完整物理路径 | SHA256 前缀 |
|---|---|---|---|---|---|---|---|
| **Moment-DETR** | `best.pt` | 76.5 MB | Val mean G1 | 4 | 49.04 | [`/home/guoxiangyu/SoccerGMR-GEQ-clip-v2-joint/runs/clip_joint/E3/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-clip-v2-joint/runs/clip_joint/E3/geq_full/best.pt) | `18f3c0cbb7...` |
| **EaTR** | `best.pt` | 130.1 MB | Val mean G1 | 4 | 51.15 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/eatr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/eatr/geq_full/best.pt) | `d558e0c24e...` |
| **QD-DETR** | `best.pt` | 108.2 MB | Val mean G1 | 4 | 51.47 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/qd_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/qd_detr/geq_full/best.pt) | `e252624f9e...` |
| **CG-DETR** | `best.pt` | 170.9 MB | Val mean G1 | 1 | 48.99 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/cg_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/cg_detr/geq_full/best.pt) | `3f3467e45d...` |
| **Sim-DETR** | `best.pt` | 145.7 MB | Val mean G1 | 4 | 51.85 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/sim_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/sim_detr/geq_full/best.pt) | `6d5daea018...` |
| **BM-DETR** | `best.pt` | 123.7 MB | Val mean G1 | 10 | 48.98 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bm_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bm_detr/geq_full/best.pt) | `a27e230fbc...` |
| **BAM-DETR** | `best.pt` | 178.4 MB | Val mean G1 | 2 | 48.03 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bam_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bam_detr/geq_full/best.pt) | `a34334366a...` |
| **UVCOM** | `best.pt` | 235.7 MB | Val mean G1 | 4 | 44.81 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/uvcom/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/uvcom/geq_full/best.pt) | `d19136b168...` |
| **TR-DETR** | `best.pt` | 155.7 MB | Val mean G1 | 1 | 46.53 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/tr_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/tr_detr/geq_full/best.pt) | `8ae85c2817...` |
| **TaskWeave** | `best.pt` | 134.6 MB | Val mean G1 | 4 | 46.07 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/taskweave/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/taskweave/geq_full/best.pt) | `3e3916b6be...` |
| **LD-DETR** | `best.pt` | 301.9 MB | Val mean G1 | 10 | 47.10 | [`/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/ld_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/ld_detr/geq_full/best.pt) | `9d9aaf81ab...` |

## 二、按 11 个骨干横向交叉对照矩阵

| 骨干名称 | Method 1: GMR baseline | Method 2: GMR(aux) | Method 3: GEQ only (ref) | Method 4: GEQ+aux |
|---|---|---|---|---|
| **Moment-DETR** | [`best.ckpt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/moment_detr/gmr/best.ckpt)<br>(Score: 7.35) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-clip-v2/runs/clip_v2/E3/gmr/best.pt)<br>(Score: 7.06) | [`best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/moment_detr/full/best.ckpt)<br>(Score: 48.91) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-clip-v2-joint/runs/clip_joint/E3/geq_full/best.pt)<br>(Score: 49.04) |
| **EaTR** | [`best.ckpt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/eatr/gmr/best.ckpt)<br>(Score: 19.34) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/eatr/gmr/best.pt)<br>(Score: 18.88) | [`best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/eatr/full/best.ckpt)<br>(Score: 51.52) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/eatr/geq_full/best.pt)<br>(Score: 51.15) |
| **QD-DETR** | [`best.ckpt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/qd_detr/gmr/best.ckpt)<br>(Score: 17.51) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/qd_detr/gmr/best.pt)<br>(Score: 16.26) | [`best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/qd_detr/full/best.ckpt)<br>(Score: 51.57) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/qd_detr/geq_full/best.pt)<br>(Score: 51.47) |
| **CG-DETR** | [`best.pt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr/gmr/best.pt)<br>(Score: 8.65) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/cg_detr/gmr/best.pt)<br>(Score: 8.05) | [`best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/cg_detr/full/best.ckpt)<br>(Score: 49.33) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/cg_detr/geq_full/best.pt)<br>(Score: 48.99) |
| **Sim-DETR** | [`model_best.ckpt`](file:///home/guoxiangyu/VLMbasedIter_momentretrival/Sim-DETR/results_soccer_gmr_csc/native_nullaware_maskloss6_bsz8_seed2023/model_best.ckpt)<br>(Score: 22.84) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/sim_detr/gmr/best.pt)<br>(Score: 22.16) | [`best.ckpt`](file:///home/guoxiangyu/simdetr_gmr_multiroute_geq/runs/geq_baseline_full/best.ckpt)<br>(Score: 51.98) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/sim_detr/geq_full/best.pt)<br>(Score: 51.85) |
| **BM-DETR** | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/gmr/best.pt)<br>(Score: 3.61) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bm_detr/gmr/best.pt)<br>(Score: 4.31) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/geq_full/best.pt)<br>(Score: 49.14) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bm_detr/geq_full/best.pt)<br>(Score: 48.98) |
| **BAM-DETR** | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/gmr/best.pt)<br>(Score: 25.49) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bam_detr/gmr/best.pt)<br>(Score: 23.83) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/geq_full/best.pt)<br>(Score: 49.01) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/bam_detr/geq_full/best.pt)<br>(Score: 48.03) |
| **UVCOM** | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/gmr/best.pt)<br>(Score: 8.77) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/uvcom/gmr/best.pt)<br>(Score: 21.48) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/geq_full/best.pt)<br>(Score: 50.28) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/uvcom/geq_full/best.pt)<br>(Score: 44.81) |
| **TR-DETR** | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/tr_detr/gmr/best.pt)<br>(Score: 3.90) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/tr_detr/gmr/best.pt)<br>(Score: 3.75) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/tr_detr/geq_full/best.pt)<br>(Score: 46.72) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/tr_detr/geq_full/best.pt)<br>(Score: 46.53) |
| **TaskWeave** | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/taskweave/gmr/best.pt)<br>(Score: 3.46) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/taskweave/gmr/best.pt)<br>(Score: 4.34) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/taskweave/geq_full/best.pt)<br>(Score: 47.04) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/taskweave/geq_full/best.pt)<br>(Score: 46.07) |
| **LD-DETR** | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/ld_detr/gmr/best.pt)<br>(Score: 7.44) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/ld_detr/gmr/best.pt)<br>(Score: 6.91) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-next/runs/second_batch/formal/ld_detr/geq_full/best.pt)<br>(Score: 48.84) | [`best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/runs/e3_eleven/formal/ld_detr/geq_full/best.pt)<br>(Score: 47.10) |

## 三、选模与冻结审计规则

1. **Val 选模严格冻结**：所有 `best.pt` / `best.ckpt` 均为验证集（Standard Val）指标达到最优时保存的检查点；测试集（Standard Test / Full Test）只进行前向只读推断，严禁用于选模或回退。
2. **哈希双向校验**：
   - `Method 1` (GMR baseline) 与 `Method 3` (GEQ only) 检查点哈希完全符合 `/home/guoxiangyu/SoccerGMR-GEQ/results/presence_thresholds/INPUTS.local.json`。
   - `Method 2` (GMR aux) 与 `Method 4` (GEQ+aux) 检查点哈希完全符合 `/home/guoxiangyu/SoccerGMR-GEQ-e3-aux-11/results/e3_high_retention/EVIDENCE.json`。
3. **文件存在性验证**：全部 44 个路径经本地脚本测试均 `exists == True`，无任何缺失。
