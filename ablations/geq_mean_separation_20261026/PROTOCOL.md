# 固定研究协议

## 1. 假设和解释边界

H-GAP：显式正负类均分间隔约束能改善均分分离，并可能改善排序或拒识。
H-ENDPOINT：分别限制正均分下限与负均分上限，能抑制两类分数一起上移。

均分更分离不保证 AUROC、AUPRC、困难正例召回或定位改善。两组结果无论方向如何均保留。单 seed、固定预算不支持统计显著性或跨种子稳定性结论，不预设方法新颖性。

## 2. 损失

令 z 为附着计算图的 GEQ 视频级 logit，s=sigmoid(z)，y∈{0,1}。混合 batch 中：

mu_pos = mean(s[y==1]); mu_neg = mean(s[y==0])。

沿用 E1 的原始损失 L_base = L_GMR_native + BCEWithLogits(z,y)。native 包含原生视频级 existence BCE、候选分类与定位回归；native 与新增 GEQ 监督路径保持 E1 原状。

### GAP

L_gap = relu(m - (mu_pos - mu_neg))²；m=0.5。
L_total = L_base + 1.0 * L_gap。

它在均分间隔不足时同时提升正均分、降低负均分，间隔达标后停止施压。它允许不同端点组合满足同一间隔。

### ENDPOINT

L_endpoint = 0.5 * [relu(a-mu_pos)² + relu(mu_neg-b)²]；a=0.75，b=0.25。
L_total = L_base + 1.0 * L_endpoint。

两类等权。某一类达标后，该类新增梯度停止，另一类仍可优化。

### 固定值的依据

0.75/0.25 是概率区间上关于0.5对称的先验目标，二者间距为0.5，与 GAP 的 m 对齐；lambda=1 是预先固定的探索协议。没有根据现有 Test 均分反推这些目标，也不搜索权重。

相同 lambda 不表示梯度强度相同：例如两类同时活动且对称偏离时，按上述定义 GAP 与 ENDPOINT 的梯度量级不同。这是两种固定目标的整体比较，不能把结果差异全部归因于目标几何。记录原始损失与梯度量级用于解释，不据此自动调参或扩展消融。

### 全正或全负 batch

两个新增损失都返回 z.sum()*0.0；只计算存在的单类均分供日志使用，不伪造另一类、不引入跨 batch 缓存。native 与 GEQ BCE 继续训练。新增监督只对混合 batch 激活，记录激活比例和每批正负数。沿用原始 shuffle，不改为类平衡 sampler。

## 3. 训练与初始化

- 仅 FlashVTG 两次新训练；seed=2026，4 epochs，batch size=8，shuffle=True，drop_last=False。
- Train/Val/Test=4138/465/1036，复用 E1 划分、预处理与离线 CLIP/SlowFast/文本特征。外部特征提取器不在计算图中。
- 一个新 AdamW：原始 GMR lr=1e-5，GEQ lr=3e-5，weight decay=1e-4；constant LambdaLR，clip=0.1，AMP=False。
- GMR 与 GEQ 全量联合更新；不 detach 融合特征，不在 backbone forward 包裹 no_grad，不强制 base.eval() 冻结训练行为。
- 不恢复旧 optimizer/scheduler；不 early stop；只正式报告第4轮。
- 原始 checkpoint 路径与哈希见各 JSON；执行前检查文件存在、SHA256、strict load 与初始 tensor 一致性。
- 同一 seed 下 GAP、ENDPOINT、E1 的共享初始张量与 loader 顺序一致。损失无随机操作；不应因新增分支消耗额外模型 RNG。
- 保留原生 localization Head、匹配/point target、分类及存在性监督。无损失路径的原生禁用分支如实记录，不宣称全部参数都有非零梯度。

## 4. 正式评估和诊断

唯一模型选择规则为第4轮。每轮可记录 Val loss 与分数均值，但不根据 Val 或 Test 更换 epoch/阈值/权重。正式运行官方 Val/Test，并独立复算。

对各 split 的全部原始预测计算：

D_pos = mean(1-s | y=1)
D_neg = mean(s | y=0)
D_balanced = (D_pos+D_neg)/2
S = 100*(1-D_balanced)
mean_gap = mu_pos-mu_neg

每类在全 split 内求均值，不平均各 batch 的均值，避免 batch 大小及类数影响。S=50*(1+mean_gap)，因此 S 与 mean_gap 是同一信息的不同刻度，不作为两项独立证据。完美区分且到达端点 S=100；常数预测 S=50。该指标不能完整描述分布重叠或校准，也不替代 AUROC/AUPRC。两类端点误差分别保存。

存在性：AUROC、positive AUPRC、Rej-F1、TPR/Positive Presence Recall、TNR/N-acc。主阈值0.4；0.5/0.6仅固定敏感性报告。
定位：mAP、mR@1、mR@5、mR+@5、正查询定位 Recall@5/IoU0.5、G-mIoU@1/@3。原始正例定位独立于 existence hard gating；G-mIoU 与纯定位分别解释。
诊断：正负概率/logit分布、各类均值/分位数、每轮 train/Val 分项损失、混合 batch 比例、新损失激活比例与梯度范数、相对 E1 的正例存在性 lost/restored 和定位 lost/restored。

## 5. 结论问题

1. GAP、ENDPOINT 是否提高均分间隔及 S？是正类上升、负类下降还是两者共同导致？
2. AUROC/AUPRC 是否改善，或仅分数尺度发生变化？
3. 在固定0.4下是否改善 TPR，同时造成更多负查询误接收？
4. 原始定位 mAP、正查询召回及多时刻召回是否受损？
5. 新增损失是否实际激活并传到共享 Transformer；若未改善，是否存在损失过早达标、混合 batch 太少、饱和或与原生损失竞争等迹象？

不因为结果未提升临时延长训练、追加种子、调权重、加入模块或进行阈值选择。

