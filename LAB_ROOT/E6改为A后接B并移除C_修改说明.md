# E6 改为 A 后接 B 并移除 C 的修改说明

本文用于后续实现与论文表述修订。本次只编写说明，不修改模型代码、训练脚本或配置，也不运行训练。

依据分支 `DQ_CGP_SoccerGMR_SI_Bind_IsolatedLab`，核对提交 `34cfe40e55514dd52703ca3ae13d7dd3c9d9b57e`。目标是先完成机制 A 的层间绑定与残差更新，再将更新后的槽状态送入机制 B，移除机制 C 的支配度刷新及其训练状态依赖。

该分支已拉取至 `D:\大四上作业\ICME投稿辅导\Improving-Target-Presence-for-Generalized-Moment-Retrieval`，本地 HEAD 与上述提交一致，交付前确认 Git 工作区干净。说明文件保存在仓库目录外。

## 1 方案选择

建议保留当前代码中 B 的定义，即对各槽的时间上下文与槽均值上下文之差进行共享线性映射。将“每槽独立正交矩阵”作为另一项研究假设，不纳入本次改动。

理由如下。

1. 本次需要回答的是移除 C 和调整 A、B 顺序是否有效。若同时替换 B 的参数化方式，实验收益将混合顺序变化、参数量变化和正交约束变化，难以归因。
2. 当前 B 已经利用各槽自身的时间上下文形成差异化更新。这里的“槽私有”体现在输入上下文随槽变化，参数矩阵本身由所有槽共享。
3. 每个矩阵满足自身正交，不代表不同槽的输出互相正交，更不代表它们覆盖不同时间区域。即使使用不同的正交变换，后续注意力仍可能指向同一事件。
4. 对方阵而言，正交变换主要保持长度与内积结构，不会自动实现时间区域分工。若采用每槽独立矩阵，还必须明确正交约束如何持续成立；仅做正交初始化不足以保证训练后的正交性。
5. 以隐藏维度 256、槽数 10 为例，共享矩阵含 65,536 个权重，每槽独立矩阵共含 655,360 个权重，不含偏置。更大参数量本身也是额外实验变量。

因此，选择方案 1 更符合当前目标。方案 2 可以另行开展，并需要与参数量匹配的非正交方案比较。

## 2 当前代码与原文字描述的差异

核心实现位于 `src/soccergmr/modules/interlayer_bind.py` 的 `InterLayerBind` 与 `build_interlayer_adapter`。

当前 A 先根据槽状态、文本语义和视频记忆计算时间注意力，再得到各槽的时间上下文。残差网络使用原始槽状态、时间上下文和文本语义的拼接输入，产生更新量。

当前 B 只有一个 `innovation_projection = nn.Linear(d, d)`。它处理的是各槽上下文减去停止梯度的槽均值上下文，没有每槽独立矩阵，也没有正交约束。

当前 C 在训练阶段、预热结束后，根据正样本 Top-1 槽分布的 EMA 计算刷新概率并进行 Bernoulli 采样。具体概率由超过均匀占比 `1/K` 的部分缩放得到，最高概率由 `p_max` 控制，并非原文字中单一固定阈值判断。刷新作用于候选槽状态，不能据此推导“保证没有假阳性”。

设原始槽为 S，绑定得到的上下文为 c，文本语义为 s，A 的残差为 r，B 的创新为 n，C 刷新后的状态为 S_C。当前 E6 的主要计算关系是：

\[
r=R([S;c;s]),\qquad
n=W_{nov}\big(c-\operatorname{sg}(\operatorname{mean}_{slot}(c))\big),
\]

\[
S_{out}=S_C+\beta\big(r+\lambda_{nov}n\big).
\]

其中 `sg` 表示停止梯度。A 和 B 的更新量均使用由原始槽 S 得到的上下文，随后一起相加。仅把 C 关闭，所得结构仍然是两个更新量的合并，不能据此声称 B 读取了 A 更新后的状态。

以 QD-DETR 的 E6 架构配置为例，实际系数为：

- `si_bind_beta = 0.05`，控制残差更新幅度。
- `si_bind_lambda_nov = 0.2`，控制创新项的相对权重。
- `si_bind_loss_coef = 0.2`，控制绑定监督损失权重。
- `si_bind_alpha = 0.25`，控制 C 的刷新混合比例。

原文字中“A 的绑定系数 α = 0.2”与这些参数含义不一致。此次应沿用代码中的命名和系数，移除 C 后不再用刷新参数 α 描述 A。

## 3 严格串行的目标数据流

目标路径为：

**第 l 层解码器输出 S → A 的时间绑定与残差更新 → 中间槽状态 H_A → B 的上下文差异创新 → 下一层解码器输入。**

为保留 B 原有“时间上下文差异”的含义，同时让 B 真正依赖 A 的输出，本说明采用以下设计。

第一步，A 按现有方式计算时间上下文 c，并完成残差更新：

\[
c=\mathcal T(S,V,s),\qquad
r=R([S;c;s]),\qquad
H_A=S+\beta r.
\]

这里 \(\mathcal T\) 表示现有时间注意力及上下文汇聚运算，V 为视频记忆，s 为文本语义。

第二步，B 使用 H_A 重新计算时间上下文：

\[
c_A=\mathcal T(H_A,V,s),\qquad
\bar c_A=\operatorname{sg}(\operatorname{mean}_{slot}(c_A)).
\]

该步骤复用 A 的时间绑定投影参数；视频键和值可复用本次前向计算结果。必须重新计算依赖槽状态的查询、注意力和上下文，不能直接复用由原始 S 得到的 c。

第三步，使用现有共享创新映射，完成第二次残差更新：

\[
n_A=W_{nov}(c_A-\bar c_A),\qquad
S_{out}=H_A+\beta\lambda_{nov}n_A.
\]

该设计保留了 β = 0.05 和 λ_nov = 0.2 的尺度关系，创新项外部系数仍为 0.01。两次残差更新分别作用于 S 和 H_A，避免重复叠加原始槽。

B 的梯度应能通过 c_A 和 H_A 回传到 A。只对槽均值上下文停止梯度，不对 H_A 或 c_A 整体停止梯度。输入、输出仍保持 `[K, batch, d]`，内部上下文保持 `[batch, K, d]`。

该方案复用参数，但增加一次由更新后槽状态发起的时间注意力计算，需要记录计算量与推理耗时。不能直接宣称串行化没有计算开销。它是待实现的设计，不能用当前 E6 的结果作为其性能证据。

## 4 C 的移除范围

移除 C 应覆盖实际训练路径，而不只是把刷新概率设为零。

- 新方案中不执行 dominance 或 uniform 刷新，不进行 Bernoulli 采样，不使用刷新 token 混合候选状态。
- 新方案不需要 `refresh_fc`、`refresh_norm`，不依赖 `top1_ema`、刷新预热轮次或刷新专用的 `current_epoch`。
- 在 `src/soccergmr/train.py` 中，新方案应跳过 `extract_top1_slots` 和 `update_top1_ema` 这条刷新统计更新路径。当前训练循环仅判断方法是否存在，因此只改 `refresh_mode` 仍可能继续更新 EMA。
- 新方案的日志应明确“刷新机制不适用”，不能用零刷新率冒充刷新效果。槽使用分布可作为独立评估诊断保留，但不再反馈控制训练。
- C 专用参数 `si_bind_p_max`、`si_bind_alpha`、`si_bind_warmup`、`si_bind_ema_decay` 对新方案不再生效。

历史 E4、E5、E6、E7 仍需可复现。因此以上移除针对新方案的执行路径；后续实现不应通过删除共享模块中的全部 C 代码破坏历史实验。

## 5 后续实现涉及的位置

`src/soccergmr/modules/interlayer_bind.py`：增加严格串行路径，将 A 更新后的 H_A 作为 B 的时间查询输入，并在新方案中禁用 C。维护工厂函数中的方案选择，避免 E6 默认的 `dominance` 或显式配置覆盖导致 C 被重新启用。

`configs/interlayer_architectures/<backbone>/E6.json`：以现有参数为参照建立新方案配置。建议新方案使用明确名称 `AB_serial_noC`，保留旧 E6 的含义。该名称是建议新增标识，当前提交尚不支持直接运行。仅设置 `si_bind_refresh_mode = "none"` 不能完成严格串行化。

`configs/experiments/interlayer/<backbone>/E6.json`：后续为新方案建立对应实验入口，并确保它指向新架构配置。当前部分配置包含原训练服务器的绝对路径，迁移运行时需要核对，不能只修改未被实际加载的本地 JSON。

`src/soccergmr/train.py`：区分启用 C 的历史方案与无 C 新方案，分别处理 EMA 更新、刷新日志及状态保存。

`tests/test_interlayer_bind_gate0.py`：后续实现时扩展已有机制检查，保留历史方案检查，新增串行依赖与无 C 路径检查。本次仅列出验收要求，不增加或执行测试代码。

`README.md`：修订机制说明和数据流图，把旧的三分支融合改为新方案的 A → B 串行流程；保留历史 E6 的结果及名称，避免把旧结果归到新结构下。删除新方案中的正交矩阵、刷新 EMA 和零假阳性保证等不适用表述。

若新方案以后接入 GEQ，应从新方案自己的检查点初始化，单独命名。此次不改变 GEQ 或原有存在性判别头。

## 6 监督与检查点处理

保持原有定位、分类、存在性及绑定监督的定义和权重。建议绑定监督继续使用 A 第一次计算的时间注意力，B 的第二次注意力不额外增加监督，以免引入新的实验变量。若修改诊断输出结构，应明确区分 A 和 B 的注意力，避免现有损失误读成 B 的输出。

新的训练对比优先从同一初始化协议重新训练。旧 E6 检查点包含刷新专用参数和状态，且其权重是在 C 存在时训练得到的；即使能够加载，也不能把结果视为新结构从头训练的消融结果。

如确需迁移旧权重，应只显式排除已移除的 C 专用键，检查其余键是否完整匹配，并记录为初始化迁移实验。不要用未检查的非严格加载掩盖其他参数缺失。

## 7 验证实验与验收条件

最小对照组为旧 E6、现有 E3、严格串行的 `AB_serial_noC`。当前工厂定义中 E3 为 A 加现有 B、无 C；在其他配置一致的前提下，E6 与 E3 的比较用于观察 C 的作用，E3 与新方案的比较用于观察严格串行化的作用。A 单独运行的 E2 可作为补充对照。不同方案保持数据、种子、训练预算和检查点选择标准一致。

实现验收需要确认以下几点。

1. B 的注意力查询实际来自 H_A；只把两个加法拆成两行不算完成串行化。
2. 新方案的训练和评估均不触发 C，不读取或更新支配度 EMA，也不进行刷新随机采样。
3. 当 λ_nov = 0 时，输出退化为 A；当 β = 0 时，输出槽保持原样。B 对 A 的梯度路径可用，均值上下文的停止梯度行为保持一致。
4. 槽数、批次布局、视频 padding 和全 padding 情况仍被正确处理，输出维度不变。验证相同输入确定性时，应在评估模式下进行，避免把骨干中的 dropout 与 C 混淆。
5. 记录正样本定位 mAP，并单独报告多事件样本的召回或覆盖情况；同时报告拒绝指标、端到端 G-mIoU 及阈值曲线。空集样本上的假阳性率应由预测统计得到，不能由机制描述推断。

本次未实现新结构，未运行上述实验，也没有新方案性能结果。

## 8 可替换原机制介绍的文字

拟议的 SI-Bind 串行版本由层间绑定与槽差异创新两个阶段组成。机制 A 根据解码器槽状态、文本语义和视频记忆建立时间对应关系，提取槽相关的时间上下文，并通过轻量残差更新保留原有槽表示。机制 B 接收 A 更新后的槽状态，重新汇聚对应的视频上下文，再将每个槽相对平均槽上下文的差异输入共享线性映射，形成创新残差并送入下一层解码器。该设计旨在先稳定候选的时间信息，再增强槽之间的上下文差异，以改善多事件检索。新版本移除支配度 EMA、自适应刷新及相关随机替换路径，其定位收益和拒绝性能需要通过与原 E6、E3 的受控实验验证。

## 9 核对来源

- [指定分支](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/tree/DQ_CGP_SoccerGMR_SI_Bind_IsolatedLab)
- [核心绑定与创新实现](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/34cfe40e55514dd52703ca3ae13d7dd3c9d9b57e/src/soccergmr/modules/interlayer_bind.py)
- [训练入口](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/34cfe40e55514dd52703ca3ae13d7dd3c9d9b57e/src/soccergmr/train.py)
- [QD-DETR 的 E6 架构配置](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/34cfe40e55514dd52703ca3ae13d7dd3c9d9b57e/configs/interlayer_architectures/qd_detr/E6.json)
- [已有机制检查](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/34cfe40e55514dd52703ca3ae13d7dd3c9d9b57e/tests/test_interlayer_bind_gate0.py)

说明日期为 2026 年 11 月 8 日。
