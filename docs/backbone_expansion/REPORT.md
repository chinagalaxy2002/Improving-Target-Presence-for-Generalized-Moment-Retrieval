# SoccerGMR GMR → GEQ Full：新增 DETR 骨干调研

核验日期：2026-11-02。范围：作者/维护者公开仓库、论文页面、README、实际模型与训练源码。此次没有安装候选模型环境、移植代码或启动训练；“已核验”指代码及入口存在，不代表已在 SoccerGMR 跑通。当前工作仓库仍保持干净。

## 建议与研究目标

先做 **BM-DETR、BAM-DETR、UVCOM**，再做 **TR-DETR、TaskWeave**。第一批三种分别覆盖背景建模、边界导向解码、多粒度跨模态融合，结合已有五个骨干，可将检验范围扩到八个骨干。难度为基于源码的工程判断，不是已测工时或成功保证。

目标是检验：大量存在性误判造成的 G-mIoU 损失，能否通过统一的“先训练 GMR、再 GEQ Full 微调”范式在不同结构上缓解。更多骨干可以提供更广的有效性证据，不能证明所有模型都有效；同一家族的增强版本应单列，不能当作完全独立的架构重复计数。

## 候选清单

| 候选 | 论文/年份 | 结构与实验价值 | 源码入口和主要适配点 | 建议 |
|---|---|---|---|---|
| [BM-DETR](https://github.com/minjoong507/BM-DETR) | WACV 2025 | 背景感知、前景概率建模；可检查已有较强背景建模后，GEQ 是否仍有收益 | `bm_detr/model.py`、`bm_detr/scripts/train_qv.sh`。原生 `vid_logits` 是逐帧概率，不可直接当作视频级 GMR 存在性 logit；数据加载器使用 `relevant_windows[0]`，要处理真实空查询、多窗口及额外负查询/时间移位 | 首批；中等 |
| [BAM-DETR](https://github.com/Pilhyeon/BAM-DETR) | ECCV 2024 | anchor/边界双路径与质量排序，扩展解码结构差异 | `bam_detr/model.py`、`bam_detr/scripts/train.sh`。内部三参数边界表示，输出已转换为中心/宽度；保留原生 quality 排序，检查 encoder 的正负配对多次调用及 GMR head 输入维度 | 首批；中等偏高 |
| [UVCOM](https://github.com/EasonXiao-888/UVCOM) | CVPR 2024 | 多粒度局部/全局跨模态融合 | `uvcom/model.py`、`uvcom/CIM.py`、`scripts/train_QV_scratch.sh`。GEQ 的候选读取点为 `base.CIM.GKA_Module.layers`，不是通用的 `base.transformer.encoder.layers`；核验全局 token 偏移、正负分支与 epoch 控制 | 首批；中等偏高 |
| [TR-DETR](https://github.com/mingyao1120/TR-DETR) | AAAI 2024 | MR/HD 相互细化 | `tr_detr/model.py`、`tr_detr/transformer.py`、`tr_detr/scripts/train.sh`。`HD2MR` 重复调用 encoder；特征读取必须声明取哪次调用。原生 MR/HD 的监督不能无说明移除；Soccer 只有窗口标注时要明确派生 saliency 的规则 | 第二批；中等 |
| [TaskWeave](https://github.com/EdenGabriel/TaskWeave) | CVPR 2024 | 任务解耦与跨任务反馈，覆盖不同特征流 | `taskweave/model.py`、`taskweave/transformer.py`、`taskweave/scripts/train.sh`。`forward(epoch_i, ...)`；反馈分支有 epoch 门槛、MR/HD 分支和多次 transformer 调用。GEQ 阶段不能无意重置原生训练时钟；原生联合损失权重需保留 | 第二批；中等偏高 |
| [LD-DETR](https://github.com/qingchen239/ld-detr) | arXiv 2025，本次未确认会议归属 | Loop Decoder、卷积融合、动量蒸馏；结构差异有价值 | `ld_detr/model.py`、`ld_detr/methods.py`、`ld_detr/scripts/train_qvhighlight.sh`。没有当前通用的 `input_txt_proj`/单一 transformer 布局；有 momentum encoder 和队列状态，恢复时均需保存，读取点要显式适配 | 后续；高 |
| [MESM](https://github.com/lntzm/MESM) | AAAI 2024 | 词—帧/句—段的语义重建，DETR 系模型 | `model/model.py`、`model/transformer.py`、根目录 `train.py`。原生接口含 token id、词权重、文本编码/MLM 和 HDF5 特征；现有缓存的 CLIP hidden state 不一定足以保留其完整方法。应先做监督/输入可用性审计 | 储备；高 |
| [LA-DETR](https://github.com/sjpark5800/LA-DETR) | WACV 2026，正式论文题名含 MomentMix；仓库仍保留早期 arXiv 题名 | 长度感知解码/匹配与 MomentMix 增强 | `la_qd_detr/`、`la_tr_detr/`、`la_uvcom/` 都有训练入口。若采用应明确基于哪一个父骨干，保留长度匹配和增强协议 | 作为变体/鲁棒性扩展，不能按三个独立全新骨干计算 |

以上八个候选/变体的许可文件均已读取，均含 MIT 许可。继承源码时保留各自文件及上游通知，不把 GitHub API 对拼接 LICENSE 的 `NOASSERTION` 误当成没有许可。

## 不放入首批的两项

- [MS-DETR](https://github.com/snailma0229/MS-DETR)：2025 年论文/公开实现，有 `ms_detr/model.py` 和 `scripts/train.sh`，有 SlowFast/CLIP 分支和辅助数据需求。README 表示辅助 JSON/特征生成还在陆续发布；本次递归树未见 LICENSE/COPYING 文件。因此标为“公开代码、开源许可未确认”，不与已确认 MIT 的候选等同，也不默认复制进将发布的主仓库。
- [UMT](https://github.com/TencentARC/UMT)：CVPR 2022，公开训练框架和许可可读，但其原生 `BoundaryHead` 是逐 clip 的 center/window/offset 预测与 top-k 解码，使用 Gaussian focal 等损失。不能仅因使用 Transformer/query 就按本项目的固定候选、匹配损失 DETR 骨干计数。适合作为将来的非同构外延实验，主表单列；另有 `nncore`、模态和特征维度差异。

同理，本轮不把通用图像目标检测的 DAB/DINO 模型直接算作已适配的视频检索骨干，也不将 FlashVTG 的多尺度预测头直接算成另一种标准 DETR decoder。Flash 的解码形式可由[论文方法段](https://openaccess.thecvf.com/content/WACV2025/papers/Cao_FlashVTG_Feature_Layering_and_Adaptive_Score_Handling_Network_for_Video_WACV_2025_paper.pdf)核验。

## 首批必须先补的工程接口

当前仓库：`/home/guoxiangyu/SoccerGMR-GEQ`，分支 `soccergmr-standard-geq-full`，调研开始时 HEAD `9e3ba902119af8ef7bc5f268aece29dbac8e7687`。

当前 `encoder_root()` 只有对 Sim 的特殊处理，GEQ 假定其余模型都含 `input_txt_proj` 和 `transformer.encoder.layers`，并使用固定宽度 256。当前预测器也默认 foreground channel 0 的 softmax，只有 Sim 的质量分数特判。这些是已有五骨干的合同，不是新八种方法已经适配的证明。

建议新增显式 adapter 能力：

1. 文本读取模块、encoder 视频读取模块列表、布局/偏移、有效 mask、正分支选择和重复调用策略；避免为了兼容而修改 GEQ 的核心读出算法。
2. 原生 decoder 状态/池化入口和存在性 head。上游只有 VMR 时，要明确这是新构建的 GMR baseline：采用预先固定的一致 head 规则，处理维度差异，并同时用于该骨干的 GMR 和 GEQ Full。逐帧 saliency/FPM 不能未经验证直接当成视频级存在性概率。
3. 原生损失 callback，保留方法核心损失/原生增强；对真实 null、多窗口、全 null、全前景/无背景和 padding 做明确的数值安全处理。合成负配对不能未经核验当作真实空查询标签。
4. 原生候选导出/排序 callback；尤其 BAM 的质量排序必须保留，不能用现有统一 foreground softmax 偷换。foreground 通道以源码训练标签和导出实现为准，不能只看注释。两阶段保持相同后处理。
5. epoch/阶段路由与 buffer 状态。TaskWeave 等不能直接把微调 epoch 重置成原生早期模式；LD 的 momentum/queue 必须可恢复。
6. 首批全部适配并 debug 后冻结源码与配置再跑正式队列：现有 checkpoint identity 绑定整个 `src`，训练中继续改其它骨干代码也可能使身份校验变化。不得关闭 identity/strict 校验来续训。

判断难度仅为源码推断；未给出未经实测的 GPU 时间或显存保证。官方多仓库原环境是 PyTorch 1.9/1.13，不保证在当前 2.5.1 环境直接运行。当前可复用环境为 `/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python`（此前五骨干验证：Python 3.10、torch 2.5.1+cu121、torchvision 0.20.1+cu121）；新依赖应隔离安装，不覆盖全局环境。

## 实验建议

首批固定 `seed=2023` 做三个新骨干，共 **6 个正式训练阶段、12 组 Test**：每个骨干 GMR → GEQ Full，两阶段 best 各测 Standard/Full。这属于拟议执行方案，本轮未启动。若用户之后要求更强统计证据，再对预先选定骨干做多 seed 和等预算无 GEQ 续训对照；单 seed、更多模型仍不足以排除额外训练预算的影响。

除 mAP/G-mIoU 外，按 .4/.5/.6 分别报告 FP/FN/TN/TP、空查询误保留率、正查询误拒率及正查询保留率。给出固定候选的 oracle-presence 差距和全部拒识参考值，避免把单纯偏向拒绝解释成存在性能力普遍改善。Test 只用于最终报告，不用于改变阈值、模型或超参数。

## 数据与协议锚点

- Standard labels：`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/{train,val,test}.jsonl`，4,138 / 465 / 1,036 条。
- Standard features：`/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr`。
- Full-Decodable labels：`/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/experiments/gmr-Global-Eventquery/data/full_decodable/Full/test.jsonl`，2,970/2,986 条。
- Full features：`/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr/Full_features`。必须使用 Full 自己的 text qid 特征。
- 新协议：Standard-only 训练；GMR 用 Val mAP 选 best，默认 patience 20；GEQ Full 从该 best 初始化，用 Val G1(.4/.5/.6) 均值选 best，严格十轮未提升早停；每 epoch 完整 Val；两阶段上限各 200；同分计未提升。
- 新协议最大视频 128 tokens，不默默截到 75；Full 的 94/104-token 视频要能前向。当前 README 图是历史六轮早停、75-token checkpoint 的评测，不能当成新协议权重或成绩。

## 主要原始来源

每一项的 README、许可、模型和训练入口、查阅时 commit 已保存在 [SOURCES.json](SOURCES.json)。以下是关键结构核验入口：

- [BM 模型](https://github.com/minjoong507/BM-DETR/blob/master/bm_detr/model.py)与[数据加载](https://github.com/minjoong507/BM-DETR/blob/master/bm_detr/start_end_dataset.py)。
- [BAM 模型](https://github.com/Pilhyeon/BAM-DETR/blob/main/bam_detr/model.py)与[encoder/decoder](https://github.com/Pilhyeon/BAM-DETR/blob/main/bam_detr/transformer.py)。
- [UVCOM 模型](https://github.com/EasonXiao-888/UVCOM/blob/main/uvcom/model.py)与[CIM](https://github.com/EasonXiao-888/UVCOM/blob/main/uvcom/CIM.py)。
- [TR 重复 encoder](https://github.com/mingyao1120/TR-DETR/blob/master/tr_detr/transformer.py)。
- [TaskWeave epoch/反馈分支](https://github.com/EdenGabriel/TaskWeave/blob/master/taskweave/model.py)。
- [LD 模型/动量状态](https://github.com/qingchen239/ld-detr/blob/main/ld_detr/model.py)。
- [MESM token/MLM 数据要求](https://github.com/lntzm/MESM#data-preparation)。
- [LA 训练与三种父骨干](https://github.com/sjpark5800/LA-DETR#-training)与[WACV 2026 正式论文](https://openaccess.thecvf.com/content/WACV2026/html/Park_MomentMix_Augmentation_with_Length-Aware_DETR_for_Temporally_Robust_Moment_Retrieval_WACV_2026_paper.html)。
- [UMT BoundaryHead](https://github.com/TencentARC/UMT/blob/main/models/blocks/head.py)。
- [Lighthouse 支持的模型和特征](https://github.com/line/lighthouse#supported-models-datasets-and-features)：可辅助检查 UVCOM/TR/TaskWeave 的实现，但不能无说明替换作者算法版本；它本身不计作新增骨干。
