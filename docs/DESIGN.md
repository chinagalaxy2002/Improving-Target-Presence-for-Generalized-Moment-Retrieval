# GMR + GEQ：Query-aware Clip Relevance Head 设计参考

**适用分支：** [`soccergmr-standard-geq-full`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/tree/soccergmr-standard-geq-full)
**文档性质：** 研究与实现参考方案；不是已完成的实现或已验证的性能结论。
**设计目标：** 在保留 GMR 原生定位和存在性训练、GEQ 全局存在性训练的前提下，引入“文本查询条件化的逐 clip 相关性预测”，为主干提供比 batch 级 GAP 更直接的时序定位监督。优先检查正查询 mAP 是否提高或至少不下降，同时观察 GEQ 的拒识能力及 G-mIoU。

## 1. 基于指定分支的现状核查

该分支已有两阶段训练：

- **Stage 1 — GMR baseline**：从对应骨干初始化，训练原生 GMR 定位与存在性目标，按 **Standard Val 正查询 mAP** 选出 `best.pt`。
- **Stage 2 — GEQ Full**：从正式 GMR `best.pt` 初始化，增加 GEQ，并**联合微调原生骨干与 GEQ**。训练损失为 `L_GMR + BCEWithLogits(z_GEQ, y_exist)`，按 **Standard Val 在存在性阈值 0.4、0.5、0.6 下 G-mIoU@1 的均值**选模。此处“Full”指骨干一起微调，不是使用 Full Test 训练。

最重要的代码事实：[`src/soccergmr/data.py`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-standard-geq-full/src/soccergmr/data.py) **已经从 `relevant_windows` 构建逐 clip 标签**，包括 `saliency_all_labels`、`relevant_clips`、`src_pos_mask`、每个 GT 区间的 `mask_labels`，以及 `src_vid_mask` 有效位置标记。对于空查询，`saliency_all_labels` 全零。

当前 GMR 的部分骨干已经有 saliency 排序损失（如 Moment-DETR 的正负 clip 配对 margin），但它不等同于以下新目标：**为每一个有效 clip 预测一个与文本查询有关的二分类 logit，并对所有正/空查询采用密集监督**。应当明确记录新增损失相对于原生 saliency 机制的增量，避免将已有组件换个名称重新计作创新。

## 2. 方法定义

给定视频特征 `V ∈ R^(B×T×Dv)`、文本 token 特征 `Q ∈ R^(B×L×Dt)`，以及已有骨干编码器的视频表示 `H ∈ R^(B×T×d)`：

1. 对文本 token 做 **带 `src_txt_mask` 的池化**，得到查询向量 `q ∈ R^(B×d)`；或者用一个轻量 cross-attention 查询文本。
2. 将视频 token `h_t` 与 `q` 显式融合，得到查询条件化特征 `u_t`。一个低成本起点：`u_t = LayerNorm(Wv h_t + Wq q + (Wv h_t) ⊙ (Wq q))`。
3. 对每个 `u_t` 应用共享的 MLP/Linear，输出 `z_clip ∈ R^(B×T)`；`sigmoid(z_clip[b,t])` 表示第 `t` 个 clip 属于查询所描述任一真实区间的概率。
4. 这个 head **仅作为辅助监督**。首轮实验不使用它裁剪边界、不改变候选排序、不改变 GEQ 存在性门控，也不使用 clip 概率取代原生 saliency。

可复用接口约定：`ClipHead(video_encoder_tokens, projected_text_tokens, src_vid_mask, src_txt_mask) -> clip_logits[B,T]`。内部表示维度通过投影适配，不假定某一个骨干的隐藏维度；视频特征必须严格对应原始视频 token，排除额外 global token 和 padding。

**建议先使用最后一层 encoder 视频 token + 当前文本投影特征**；需要更强模型时再比较多层特征融合或 GEQ 状态融合。避免首次实验一次增加太多模块。

## 3. GT 定义与损失

**默认直接复用当前分支的 `targets['saliency_all_labels']`**：对于每个视频，把所有 `relevant_windows` 转成 clip 区间后取并集；属于任一 GT 的 clip 标签为 1，其余有效 clip 为 0；空查询整行全 0。

这里必须沿用当前 loader 的时间映射：`a = floor(start / duration × T)`、`b = ceil(end / duration × T)`（再做合法范围约束），而不是默认每条视频都恰好 150 秒或直接用 `start / 2`。当前 loader 使用真实 `duration`，支持不同长度的序列。对于刚好在边界处轻微重叠的 clip，可在后续独立消融中测试“时间交集占比”的软标签；首轮不要改变已有标注规则。

**基础密集损失**：

`L_clip(b) = sum_t [valid_bt × BCEWithLogits(z_bt, y_bt)] / max(sum_t valid_bt, 1)`

`L_clip = mean_b L_clip(b)`

这能防止 padding 被当成负例，也避免长视频在批次损失中占过大比重。所有空查询 batch、只有一个 GT clip 的视频、多区间视频都必须可以稳定反向传播。首轮采用普通 BCE；若发现正 clip 严重稀缺导致模型预测全零，再通过 **Val** 对比固定正类权重或 Focal BCE，避免一开始叠加复杂损失。不要先 sigmoid 再传入 `BCEWithLogits`。

**Stage 1：** `L_total = L_GMR_native + λ_clip × L_clip`。

**Stage 2：** `L_total = L_GMR_native + L_GEQ_BCE + λ_clip × L_clip`。

建议从 `λ_clip ∈ {0.1, 0.3, 1.0}` 做小规模探索，并保留 `λ_clip=0` 的完全匹配对照。这是探索性超参数建议，不是论文或已有仓库确认的最佳值。确保 `L_clip` 不错误地只对正查询计算；空查询全零目标是此方案的重要监督来源。

## 4. 训练阶段如何接入

### 推荐先做 B：只在 GEQ Full 阶段加入

**流程：** 维持原来的 `GMR baseline → 选出 GMR best.pt → GEQ Full`；只在第二阶段创建新的 Clip Head，并在每个训练 batch 的原有损失上增加 `λ_clip × L_clip`。

**理由：** 不改变 GMR Stage 1 的**算法和损失**，因此能较直接地回答“额外时序相关性监督能否减少 GEQ Full 对 mAP 的损害”。Stage 2 仍是联合微调，Clip Head 的梯度可以进入骨干编码器。**但不等于可以直接加载旧运行的 GMR checkpoint**；本分支会对所有 Python 源码求哈希并校验训练身份，详见下一段。

**严格 checkpoint 兼容性（必须处理）：** [`checkpoint.py`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-standard-geq-full/src/soccergmr/checkpoint.py) 的 `source_identity()` 对 `src/soccergmr/` 下**全部 `.py` 文件**求哈希；[`train.py`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-standard-geq-full/src/soccergmr/train.py) 在 GEQ 初始化时要求父 GMR `best.pt` 的身份与**当前源码身份**匹配。因此添加 Clip Head 或改动训练器后，即使 Stage 1 逻辑没变，也不能指望旧分支生成的 `best.pt` 直接通过校验。推荐在**修改后的同一源码快照**下重新跑无 Clip Loss 的 GMR 基线，令 A/B 两组共享其固定 `best.pt`；或者另行实现经过审计的版本化 checkpoint 迁移，**不得直接跳过哈希/身份校验**。这既是复现安全约束，也有利于公平比较。

**实现位置：** [`src/soccergmr/geq.py`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-standard-geq-full/src/soccergmr/geq.py) 已经通过 hook 捕获**投影后文本特征和每层视频 encoder token**。可在 GEQ Full wrapper 中新增独立 `clip_head`，用最后一层视频 token 与投影文本产生 `clip_logits`；[`src/soccergmr/train.py`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-standard-geq-full/src/soccergmr/train.py) 的 `stage=='geq_full'` 分支中计算辅助损失。保留 `outputs['native_output']` 供原生 criterion 使用。**务必为 `model.clip_head.parameters()` 添加独立 optimizer 参数组**（当前 GEQ Full 优化器仅包含 `model.base` 与 `model.geq`，若 Clip Head 挂在二者之外而不增加参数组，虽然能反传但不会更新权重）；确认 `state_dict`、best/last checkpoint 和恢复逻辑覆盖新增 head。原生 `pred_exist_logits` 的替换、GEQ 推理行为和官方评价不作改变。

### 再做 C：GMR 训练和 GEQ Full 两阶段都加入

第一阶段给原生骨干套一个通用的可复用 `GMRWithClipHead` wrapper，训练 `GMR + L_clip`；第二阶段把 Stage 1 的主干和 Clip Head 权重一起继承，再新增 GEQ，继续训练 `L_GMR + L_GEQ + L_clip`。

**关键工程风险：** 目前 GEQ 阶段执行了严格的 GMR `best.pt` 状态加载。新增 wrapper 后必须显式定义两阶段权重键如何映射、Clip Head 是否继承及 optimizer 参数组，否则会出现 `strict=True` 加载失败，或悄悄重新初始化头部。修改的配置/源码也会改变训练 identity；必须使用新的实验目录、不能假装可以原样恢复旧 checkpoint。

### 两阶段 2×2 对照

| 对照组 | Stage 1 Clip Loss | Stage 2 Clip Loss | 研究问题 |
| --- | --- | --- | --- |
| A 原方法 | 关 | 关 | 当前 GMR → GEQ Full 控制组 |
| B 微调期加入 | 关 | 开 | 只增加 GEQ 微调期的时序监督有没有收益？ |
| C 基线期加入 | 开 | 关 | 基线学到的时序表示能否帮助后续 GEQ？ |
| D 两阶段加入 | 开 | 开 | 全流程持续时序监督是否最好？ |

C 组在 Stage 2 关闭 Clip Loss 时需明确是否保留 Clip Head 权重但不参与推理；四组的预算、seed、原生损失、Val 选模规则及模型比较口径需保持可解释的一致性。优先完成 A/B，再扩展 C/D。

## 5. 验收指标和必须避免的误判

- **主指标：** 正查询未经过存在性门控的 mAP、mAP 的各 tIoU 阈值分量，以及短事件和多区间子集。新增 Clip Head 的主要假设是提高查询相关证据的时间辨识能力，因此只报告 G-mIoU 改善不足以证明成功。
- **联合指标：** G-mIoU@1/@3、AUROC、Rej-F1、正例保留率、负例误保留率。GEQ 阶段沿用 Standard Val 的三阈值 G-mIoU@1 均值选模；不要为了新方法使用 Test 最优阈值。
- **辅助诊断：** clip 级 masked BCE、精确率/召回率、预测概率热力图（与 GT 对齐），特别检查 **同一视频替换不同 query 后，clip 分数是否按语义改变**。否则 head 有可能退化成不使用文本的视觉显著性检测。
- **梯度与数值：** 空查询整行零标签、全空 batch、多 GT、单个短 GT、可变视频长度、padding mask、额外 global token offset、encoder 多次调用，都应保证维度与梯度正确。特别验证增加 `L_clip` 后原生编码器参数确实得到非零梯度，且原有定位损失数值保持一致。
- **数据协议：** 仅用 Standard Train/Val 训练与选模；已选定的 checkpoint 在 Standard Test、Full-Decodable Test 上固定评估；不得根据 Test 调整 `λ_clip` 或 checkpoint。
- **控制对照：** 如果 mAP 上升，需要排除“单纯多训练一段时间”的解释；如果 G-mIoU 上升但 mAP 下降，需要区分存在性改进与定位退化。

## 6. 推荐阅读顺序与参考代码

以下为可直接查看的真实代码/论文入口，**它们用于借鉴思想，不代表在当前分支内已经实现此方案**。

| 优先级 | 链接 | 参考重点 |
| --- | --- | --- |
| 1 | [本分支 `data.py`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-standard-geq-full/src/soccergmr/data.py) | `relevant_windows` → `mask_labels`、`saliency_all_labels`，正/空查询标签和有效长度 |
| 2 | [本分支 `geq.py`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-standard-geq-full/src/soccergmr/geq.py) | 投影文本、encoder 视频 token 的 hook；最容易添加 Stage 2 Clip Head 的位置 |
| 3 | [本分支 `train.py`](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-standard-geq-full/src/soccergmr/train.py) | 原生损失 + GEQ BCE 的接入点、两阶段参数组及选模流程 |
| 4 | [本分支原生损失适配](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-standard-geq-full/src/soccergmr/backbones/loss.py) | 正/空查询分流及已有 GMR BCE，避免错误替换原有监督 |
| 5 | [UniVTG 论文](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_UniVTG_Towards_Unified_Video-Language_Temporal_Grounding_ICCV_2023_paper.html) · [模型代码](https://github.com/showlab/UniVTG/blob/main/model/univtg.py) | 最贴近该思路的 **clip-level foreground indicator**，以及与边界/显著性分开的监督 |
| 6 | [Moment-DETR 模型与 loss](https://github.com/jayleicn/moment_detr/blob/main/moment_detr/model.py) | 原生 `saliency_scores`、正负 clip pair-ranking loss；用于明确新 dense BCE 与现有方法的区别 |
| 7 | [FAANTRA 模型](https://github.com/MohamadDalal/FAANTRA/blob/main/model/futr.py) | “主任务 + 逐帧辅助监督”的多任务训练思路；它是未来动作预判，不是可直接复用的文本条件 GMR 分割 |
| 8 | [本分支算法协议](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/soccergmr-standard-geq-full/docs/ALGORITHM.md) | 原有研究目标、GEQ 行为、mAP 与 G-mIoU 的口径及选择规则 |

## 7. 最小可执行研究结论

**第一轮只需做一个可验证的改动：** 在 GEQ Full 的现有文本/视频 encoder 表示上增加一个共享的 Query-aware Clip Relevance Head，对已有 `saliency_all_labels` 计算忽略 padding 的 BCE，损失权重单独可配；其预测结果暂不进入推理。先在**同一新源码身份**下完成无 Clip Loss 的 GMR Stage 1，再让 A/B 两组共享该 GMR best，进行相同训练预算的 GEQ 微调实验。若 B 在 Standard Val 上改善了 clip 相关性并维持/提升正查询 mAP，再考虑两阶段联合训练、Dice/Focal、显式多尺度和利用 clip 分数进行候选排序。与只增加 GAP 相比，此处研究的是**查询相关的时间证据定位**，而不是仅拉开视频级正负分数的平均差距。
