# 查询条件化的逐 clip 辅助监督

## 方法是什么？

以“球员射门”查询为例：Moment-DETR 负责预测射门发生的时间区间，GEQ 判断视频里是否确实存在该事件。新增的 Clip Head 则在训练时逐段回答“这一小段是否属于查询描述的事件”。

例如，真实事件在 60～65 秒：属于该时间区间的片段标 1，其余片段标 0；查询事件不存在时全部标 0。多段真实事件取并集，padding 不参与损失。时间映射使用每个视频真实 duration 和序列长度，而不是假设固定视频时长。

## 新增模块

文本投影 token 经过有效位置掩码池化得到查询向量 q；最后一层 encoder 的视频 token 为 h_t。分别投影后融合：

```text
q = masked_mean(projected_text_tokens)
v_t = Wv(h_t)
q' = Wq(q)
u_t = LayerNorm(v_t + q' + v_t * q')
z_t = MLP(u_t)
```

对每个有效片段计算 BCEWithLogits；先按每个视频的有效 clip 数取平均，再对 batch 取平均：

```text
L_clip = mean_b [sum_t valid_bt * BCEWithLogits(z_bt, y_bt)
                / max(sum_t valid_bt, 1)]
Stage 1: L = L_GMR_native + lambda * L_clip
Stage 2: L = L_GMR_native + L_GEQ_BCE + lambda * L_clip
```

Clip Head 只是训练辅助。推理仍由 Moment-DETR 输出边界与候选分数，GEQ 决定保留还是返回空结果；Clip 分数不用于边界修正、排序或存在性门控。

## A/B/C/D 的区别

| 组别 | Stage 1 GMR | Stage 2 GEQ |
|---|---|---|
| A | 原生 GMR 损失 | 原生 GMR + GEQ BCE |
| B | 同 A | 原生 GMR + GEQ BCE + Clip Loss |
| C | 原生 GMR + Clip Loss | 原生 GMR + GEQ BCE；Clip Head 冻结 |
| D | 原生 GMR + Clip Loss | 原生 GMR + GEQ BCE + Clip Loss |

两种 GMR 父模型均从头训练。A/B 共享无 Clip 的父模型；C/D 共享有 Clip 的父模型，同时继承主干和 Clip Head。

## 实验协议

使用 Standard Train/Val，seed=2023，batch=8，最大各 200 epoch。GMR 按 Val 正查询 mAP 选模、patience=20；GEQ 按 Val 三阈值（0.4/0.5/0.6）G-mIoU@1 均值选模、patience=10。原生损失和阈值定义保持一致。

B 搜索 lambda=0.1/0.3/1.0，其最优 Val G1 均值分别为 48.80/48.58/48.77，最终选择 0.1，C/D 使用相同权重。另两个有 Clip 的 GMR 父模型提前训练，仅为并行准备，不改变基于 B Val 的权重选择。Test 不参与选权重、checkpoint 或阈值。

## 正式结果

所有指标以 0～100 报告。mAP 使用正查询的未门控候选；G1/G3 为三阈值均值。

| Test | 组别 | mAP | 相对 A | G1 均值 | 相对 A | G3 均值 | AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| Standard | A | 6.05 | — | 52.77 | — | 49.25 | 68.96 |
| Standard | B | 6.83 | +0.78 | 52.74 | −0.03 | 49.24 | 69.16 |
| Standard | C | 6.82 | +0.77 | 52.60 | −0.17 | 49.17 | 69.44 |
| Standard | D | 6.69 | +0.64 | 51.06 | −1.71 | 47.74 | 68.93 |
| Full-Decodable | A | 5.26 | — | 49.12 | — | 45.94 | 78.37 |
| Full-Decodable | B | 5.28 | +0.02 | 49.09 | −0.03 | 45.93 | 78.34 |
| Full-Decodable | C | 6.01 | +0.75 | 48.91 | −0.21 | 45.77 | 78.46 |
| Full-Decodable | D | 5.89 | +0.63 | 47.49 | −1.63 | 44.45 | 78.21 |

Standard 覆盖 1036 条，Full-Decodable 覆盖 2970/2986 条。两者存在样本重叠。

## Clip Head 自身的表现

固定概率阈值 0.5，Standard Val 上：

| 组别 | masked BCE | Precision | Recall |
|---|---:|---:|---:|
| B | 0.1761 | 0.00% | 0.00% |
| C | 0.1649 | 80.00% | 0.82% |
| D | 0.1627 | 63.16% | 0.82% |

Clip Head 严重漏检，倾向低概率输出。C 的高 precision 仅对应极少量正预测，不能据此认定效果良好。替换同一视频的不同 query 后分数有变化，但这只能说明使用了 query，不证明语义正确。

## 结论与限制

B 在 Standard 的 mAP 增益最大，C 在 Full 的 mAP 增益最大；但三组的 G1 均值均未超过 A，D 下降更明显。可以报告观察到 mAP 提升，尚不能宣称整体检索能力提高，也不能证明增益来自可靠的逐段语义识别。

本轮为单 seed，各组实际早停轮次不同，不能完全排除随机性和训练时长影响。Plain BCE 的低召回问题值得进一步研究，但本轮没有为改善 Test 分数追加权重/阈值搜索。

[训练图](../results/clipseg/training_val_panel.png) · [逐项指标](../results/clipseg/DETAILS.json) · [完整预测和训练记录](../results/clipseg/execution)
