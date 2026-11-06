# SoccerGMR 各方法模型训练检查点本地路径全景清单 (9 骨干 × 4 配置)

> **文档说明**：本清单全面收录 SoccerGMR 实验框架中 **全部 9 个时序骨干模型**（`Moment-DETR`、`EaTR`、`QD-DETR`、`CG-DETR`、`Sim-DETR`、`BM-DETR`、`BAM-DETR`、`UVCOM`、`FlashVTG`）在 **4 种核心方法配置** 下训练产出的最佳检查点物理路径、SHA-256 完整性哈希、选模轮次与验证集得分。
> 
> 所有路径均经过本地物理文件存在性与 SHA-256 哈希核验（核验时间：2026-11-06）。

---

## 📖 目录
1. [Method 1: GMR Baseline (原生基线模型检查点)](#-method-1-gmr-baseline-原生基线模型检查点)
2. [Method 2: GMR + DQ-CGPv3 (Stage 1 连续梯度路由检查点)](#-method-2-gmr--dq-cgpv3-stage-1-连续梯度路由检查点)
3. [Method 3: GMR + DQ-CGPv3 + GEQ (Stage 2 Val Mean G1 选模 - 官方推荐)](#-method-3-gmr--dq-cgpv3--geq-stage-2-val-mean-g1-选模---官方推荐)
4. [Method 4: GMR + DQ-CGPv3 + GEQ (Stage 2 Val mAP 选模 - 选模消融)](#-method-4-gmr--dq-cgpv3--geq-stage-2-val-map-选模---选模消融)
5. [对照组参考: Historical GEQ only (无 DQ-CGPv3 历史检查点)](#-对照组参考-historical-geq-only-无-dq-cgpv3-历史检查点)
6. [快速加载与推理评测指令](#-快速加载与推理评测指令)

---

## 🏛️ Method 1: GMR Baseline (原生基线模型检查点)

- **阶段与定位**：原生视频时序动作检索基线模型（无 DQ-CGPv3 增强，无 GEQ 存在性分类头，Stage 1 父模型）。
- **选模协议**：在 Standard Val 集上以 **Val Localization mAP (threshold=0.4)** 选取全局最优轮次。

| 骨干模型 (Backbone) | Checkpoint 文件 | 大小 | 选模指标 | 最佳轮次 (Epoch) | 最佳验证分数 | 完整物理路径 | SHA-256 前缀 |
|---|---|:---:|:---:|:---:|:---:|---|:---:|
| **Moment-DETR** | `best.ckpt` | 56.0 MB | Val mAP | 66 | 7.35% | [`/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/moment_detr/gmr/best.ckpt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/moment_detr/gmr/best.ckpt) | `c69fc1bdc6` |
| **EaTR** | `best.ckpt` | 103.6 MB | Val mAP | 历史最佳 | 19.34% | [`/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/eatr/gmr/best.ckpt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/eatr/gmr/best.ckpt) | `3d7f6f6cb8` |
| **QD-DETR** | `best.ckpt` | 87.7 MB | Val mAP | 历史最佳 | 17.51% | [`/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/qd_detr/gmr/best.ckpt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/qd_detr/gmr/best.ckpt) | `03011c4816` |
| **CG-DETR** | `best.pt` | 144.4 MB | Val mAP | 历史最佳 | 8.65% | [`/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr/gmr/best.pt`](file:///home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/backbonefolder/cg_detr/gmr/best.pt) | `200f2f200b` |
| **Sim-DETR** | `model_best.ckpt` | 125.3 MB | Val mAP | 历史最佳 | 22.84% | [`/home/guoxiangyu/VLMbasedIter_momentretrival/Sim-DETR/results_soccer_gmr_csc/native_nullaware_maskloss6_bsz8_seed2023/model_best.ckpt`](file:///home/guoxiangyu/VLMbasedIter_momentretrival/Sim-DETR/results_soccer_gmr_csc/native_nullaware_maskloss6_bsz8_seed2023/model_best.ckpt) | `c237601599` |
| **BM-DETR** | `best.pt` | 97.2 MB | Val mAP | 17 | 3.61% | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/gmr/best.pt) | `1382ac4645` |
| **BAM-DETR** | `best.pt` | 158.3 MB | Val mAP | 72 | 25.49% | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/gmr/best.pt) | `d2865dd460` |
| **UVCOM** | `best.pt` | 209.4 MB | Val mAP | 55 | 8.77% | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/gmr/best.pt) | `a07df6dd16` |
| **FlashVTG** | `flashvtg_dq_cgp_gmr_best_epoch32.ckpt` | 134.4 MB | Val mAP | 32 | 25.43% | [`/home/guoxiangyu/VLMbasedIter_momentretrival/Falshvtg-gmr-DQ-CGP/checkpoints/flashvtg_dq_cgp_gmr_best_epoch32.ckpt`](file:///home/guoxiangyu/VLMbasedIter_momentretrival/Falshvtg-gmr-DQ-CGP/checkpoints/flashvtg_dq_cgp_gmr_best_epoch32.ckpt) | `0a56745085` |

---

## ⚡ Method 2: GMR + DQ-CGPv3 (Stage 1 连续梯度路由检查点)

- **阶段与定位**：在原生时序骨干内集成 DQ-CGPv3 模块（双查询交叉绑定、连续梯度路由路径、多尺度动作边界细化）。
- **选模协议**：在 Standard Val 集上以 **Val mAP** 选取 Stage 1 最优模型。

| 骨干模型 (Backbone) | Checkpoint 文件 | 大小 | 选模指标 | 最佳轮次 (Epoch) | 最佳验证分数 (mAP) | 完整物理路径 | SHA-256 前缀 |
|---|---|:---:|:---:|:---:|:---:|---|:---:|
| **Moment-DETR** | `best.pt` | 68.6 MB | Val mAP | 48 | 14.49% | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/moment_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/moment_detr/gmr/best.pt) | `433d28f315` |
| **EaTR** | `best.pt` | 116.4 MB | Val mAP | 95 | 20.99% | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/eatr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/eatr/gmr/best.pt) | `a8e3eb7432` |
| **QD-DETR** | `best.pt` | 100.6 MB | Val mAP | 118 | 20.15% | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/qd_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/qd_detr/gmr/best.pt) | `decb659c84` |
| **CG-DETR** | `best.pt` | 157.1 MB | Val mAP | 75 | 17.39% | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/cg_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/cg_detr/gmr/best.pt) | `09b11dbbe0` |
| **Sim-DETR** | `best.pt` | 138.1 MB | Val mAP | 110 | 23.58% | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/sim_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/sim_detr/gmr/best.pt) | `54b5989f04` |
| **BM-DETR** | `best.pt` | 109.9 MB | Val mAP | 86 | 11.37% | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bm_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bm_detr/gmr/best.pt) | `6e59eb9e56` |
| **BAM-DETR** | `best.pt` | 170.6 MB | Val mAP | 54 | 23.02% | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bam_detr/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bam_detr/gmr/best.pt) | `380a8978a8` |
| **UVCOM** | `best.pt` | 222.1 MB | Val mAP | 115 | 19.10% | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/uvcom/gmr/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/uvcom/gmr/best.pt) | `4656ff2ac1` |
| **FlashVTG** | `model_best.ckpt` | 139.0 MB | Val mAP | 140 | 27.91% | [`/home/guoxiangyu/VLMbasedIter_momentretrival/Falshvtg-gmr-DQ-CGP/outputs/flashvtg_dq_cgp_v3_topk4_seed2024/hl-video_tef-soccer_gmr-2026-08-27-00-21-13/model_best.ckpt`](file:///home/guoxiangyu/VLMbasedIter_momentretrival/Falshvtg-gmr-DQ-CGP/outputs/flashvtg_dq_cgp_v3_topk4_seed2024/hl-video_tef-soccer_gmr-2026-08-27-00-21-13/model_best.ckpt) | `053cb68f57` |

---

## 🌟 Method 3: GMR + DQ-CGPv3 + GEQ (Stage 2 Val Mean G1 选模 - 官方推荐)

- **阶段与定位**：在 Method 2 (DQ-CGPv3 Stage 1) 父模型基础上接入 GEQ 全局事件查询头进行第二阶段全量微调。
- **选模协议**：严格遵循官方标准，以 **Standard Val 集在三个分类阈值 $\tau \in \{0.4, 0.5, 0.6\}$ 下的平均 $\text{G-mIoU@1}$** 作为选模评分（兼顾动作定位与空事件拒识）。
- **软链接映射**：各骨干根目录下的 `best_checkpoint_gmiou.pt` 软链接均指向本目录下的 `best.pt`。

| 骨干模型 (Backbone) | Checkpoint 文件 | 大小 | 选模指标 | 最佳轮次 (Epoch) | 最佳验证分数 (Mean G1 / mAP) | 完整物理路径 | SHA-256 前缀 |
|---|---|:---:|:---:|:---:|:---:|---|:---:|
| **Moment-DETR** | `best.pt` | 86.6 MB | Val Mean G1 | 4 | **51.10%** (mAP 14.61%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/moment_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/moment_detr/geq_full/best.pt) | `5dd43107f9` |
| **EaTR** | `best.pt` | 140.3 MB | Val Mean G1 | 4 | **50.53%** (mAP 19.20%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/eatr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/eatr/geq_full/best.pt) | `4790d98bdb` |
| **QD-DETR** | `best.pt` | 118.3 MB | Val Mean G1 | 10 | **47.65%** (mAP 16.81%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/qd_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/qd_detr/geq_full/best.pt) | `0e9c116009` |
| **CG-DETR** | `best.pt` | 181.0 MB | Val Mean G1 | 4 | **51.84%** (mAP 14.93%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/cg_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/cg_detr/geq_full/best.pt) | `52ffe1e4a2` |
| **Sim-DETR** | `best.pt` | 155.9 MB | Val Mean G1 | 10 | **51.86%** (mAP 21.91%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/sim_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/sim_detr/geq_full/best.pt) | `9cd77923b6` |
| **BM-DETR** | `best.pt` | 133.8 MB | Val Mean G1 | 4 | **52.49%** (mAP 10.39%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bm_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bm_detr/geq_full/best.pt) | `ee19178ce9` |
| **BAM-DETR** | `best.pt` | 188.6 MB | Val Mean G1 | 10 | **48.83%** (mAP 21.43%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bam_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bam_detr/geq_full/best.pt) | `1716d848cd` |
| **UVCOM** | `best.pt` | 245.9 MB | Val Mean G1 | 10 | **45.58%** (mAP 18.70%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/uvcom/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/uvcom/geq_full/best.pt) | `c7b089d60a` |
| **FlashVTG** | `epoch_004.ckpt` | 144.7 MB | Val Mean G1 | 4 | **51.43%** (mAP 28.15%) | [`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/stage_1_geq/runs/flashvtg/E1/epoch_004.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/stage_1_geq/runs/flashvtg/E1/epoch_004.ckpt) | `3c364f93fd` |

---

## 🎯 Method 4: GMR + DQ-CGPv3 + GEQ (Stage 2 Val mAP 选模 - 选模消融)

- **阶段与定位**：与 Method 3 相同的 Stage 2 GEQ 微调训练轨迹，但改用 **最高 Val Localization mAP** 作为选模准则。
- **消融目的**：实证揭示“过分追求验证集定位 mAP 会导致负样本拒识能力崩塌与测试集泛化退化”的机制现象。
- **软链接映射**：各骨干根目录下的 `best_checkpoint_map.pt` 软链接均指向本目录下的 `best.pt`。

| 骨干模型 (Backbone) | Checkpoint 文件 | 大小 | 选模指标 | 最佳轮次 (Epoch) | 最佳验证分数 (mAP / Mean G1) | 完整物理路径 | SHA-256 前缀 |
|---|---|:---:|:---:|:---:|:---:|---|:---:|
| **Moment-DETR** | `best.pt` | 86.6 MB | Val mAP | 6 | **14.77%** (G1 49.80%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/moment_detr/geq_map/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/moment_detr/geq_map/best.pt) | `ff16e932b0` |
| **EaTR** | `best.pt` | 140.3 MB | Val mAP | 2 | **19.82%** (G1 48.90%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/eatr/geq_map/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/eatr/geq_map/best.pt) | `8c018fab69` |
| **QD-DETR** | `best.pt` | 118.4 MB | Val mAP | 31 | **18.89%** (G1 45.20%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/qd_detr/geq_map/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/qd_detr/geq_map/best.pt) | `bd9cdc1280` |
| **CG-DETR** | `best.pt` | 181.0 MB | Val mAP | 8 | **17.77%** (G1 47.90%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/cg_detr/geq_map/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/cg_detr/geq_map/best.pt) | `afede5fe70` |
| **Sim-DETR** | `best.pt` | 155.9 MB | Val mAP | 17 | **23.27%** (G1 49.30%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/sim_detr/geq_map/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/sim_detr/geq_map/best.pt) | `6c7b534351` |
| **BM-DETR** | `best.pt` | 133.8 MB | Val mAP | 9 | **10.68%** (G1 48.50%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bm_detr/geq_map/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bm_detr/geq_map/best.pt) | `64b047756b` |
| **BAM-DETR** | `best.pt` | 188.6 MB | Val mAP | 14 | **23.90%** (G1 46.10%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bam_detr/geq_map/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/bam_detr/geq_map/best.pt) | `119f3b0f7e` |
| **UVCOM** | `best.pt` | 245.9 MB | Val mAP | 14 | **21.26%** (G1 43.10%) | [`/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/uvcom/geq_map/best.pt`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/uvcom/geq_map/best.pt) | `31175a6b60` |
| **FlashVTG** | `epoch_002.ckpt` | 144.7 MB | Val mAP | 2 | **28.66%** (G1 51.12%) | [`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/stage_1_geq/runs/flashvtg/E1/epoch_002.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/stage_1_geq/runs/flashvtg/E1/epoch_002.ckpt) | `e0abbb88f1` |

---

## 📜 对照组参考: Historical GEQ only (无 DQ-CGPv3 历史检查点)

- **阶段与定位**：早期探索阶段直接在原生 Baseline GMR（Method 1）上微调 GEQ 的历史对照组模型。

| 骨干模型 (Backbone) | Checkpoint 文件 | 大小 | 选模指标 | 最佳轮次 (Epoch) | 最佳验证分数 (Mean G1) | 完整物理路径 | SHA-256 前缀 |
|---|---|:---:|:---:|:---:|:---:|---|:---:|
| **Moment-DETR** | `best.ckpt` | 24.7 MB | Val mean G1 | 9 | 48.91% | [`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/moment_detr/full/best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/moment_detr/full/best.ckpt) | `06145a116f` |
| **EaTR** | `best.ckpt` | 42.6 MB | Val mean G1 | 4 | 51.52% | [`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/eatr/full/best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/eatr/full/best.ckpt) | `b1a6cd36da` |
| **QD-DETR** | `best.ckpt` | 35.3 MB | Val mean G1 | 4 | 51.57% | [`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/qd_detr/full/best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/qd_detr/full/best.ckpt) | `144d36e9c0` |
| **CG-DETR** | `best.ckpt` | 56.5 MB | Val mean G1 | 1 | 49.33% | [`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/cg_detr/full/best.ckpt`](file:///home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_epoch_study/20261031/cg_detr/full/best.ckpt) | `d06de66b24` |
| **Sim-DETR** | `best.ckpt` | 48.7 MB | Val mean G1 | 4 | 51.98% | [`/home/guoxiangyu/simdetr_gmr_multiroute_geq/runs/geq_baseline_full/best.ckpt`](file:///home/guoxiangyu/simdetr_gmr_multiroute_geq/runs/geq_baseline_full/best.ckpt) | `931bc5d075` |
| **BM-DETR** | `best.pt` | 121.4 MB | Val mean G1 | 16 | 49.14% | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bm_detr/geq_full/best.pt) | `0dac373364` |
| **BAM-DETR** | `best.pt` | 176.2 MB | Val mean G1 | 4 | 49.01% | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/bam_detr/geq_full/best.pt) | `2b90dc19a4` |
| **UVCOM** | `best.pt` | 233.4 MB | Val mean G1 | 4 | 50.28% | [`/home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/geq_full/best.pt`](file:///home/guoxiangyu/SoccerGMR-GEQ/runs/backbone_expansion/formal/uvcom/geq_full/best.pt) | `e474e04f71` |

---

## 🚀 快速加载与推理评测指令

### 1. 使用官方评测引擎评测任意 DETR 骨干检查点

在 `/home/guoxiangyu/SoccerGMR-DQ-CGPv3` 目录下，使用统一测试脚本评测 Stage 2 最优模型：

```bash
cd /home/guoxiangyu/SoccerGMR-DQ-CGPv3

# 评测 UVCOM 官方推荐检查点 (Method 3: Val Mean G1 选模)
python -m soccergmr test \
  --config configs/experiments/uvcom.json \
  --checkpoint runs/dq_cgp_v3/uvcom/geq_full/best.pt \
  --output runs/eval/uvcom_eval \
  --device cuda:0

# 评测 Moment-DETR 的 Method 4 (Val mAP 选模消融)
python -m soccergmr test \
  --config configs/experiments/moment_detr.json \
  --checkpoint runs/dq_cgp_v3/moment_detr/geq_map/best.pt \
  --output runs/eval/moment_detr_map_eval \
  --device cuda:0
```

### 2. 评测 FlashVTG 非 DETR 骨干检查点

FlashVTG 采用原生交叉注意力与 Highlight 特征投影，可直接通过推理模块评测：

```bash
cd /home/guoxiangyu/VLMbasedIter_momentretrival/Falshvtg-gmr-DQ-CGP

# 评测 FlashVTG + DQ-CGPv3 + GEQ 最优模型 (Epoch 4, Val G1 选模)
python -m training.flash_vtg_gmr.inference \
  --config configs/flashvtg_dq_cgp_v3_eval.json \
  --resume /home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/stage_1_geq/runs/flashvtg/E1/epoch_004.ckpt \
  --eval_split test
```

### 3. Python 代码中直接加载 State Dict

```python
import torch

# 示例: 加载 Moment-DETR Stage 2 (DQ-CGPv3 + GEQ G1 选模) 检查点
ckpt_path = '/home/guoxiangyu/SoccerGMR-DQ-CGPv3/runs/dq_cgp_v3/moment_detr/geq_full/best.pt'
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# 获取模型权重字典
if 'model' in checkpoint:
    state_dict = checkpoint['model']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

print('Successfully loaded checkpoint! Keys count:', len(state_dict))
```
