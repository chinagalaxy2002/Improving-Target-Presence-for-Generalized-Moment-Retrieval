# Moment-DETR Clip Relevance 实验结果

**A/B/C/D 已完成。** B 在 Standard Val 选出的 λ=0.1；正式 Test 固定使用该权重，未使用 Test 选参。

| 数据集 | 组别 | 正查询 mAP | 相对 A | 三阈值 G1 均值 | 相对 A |
|---|---|---:|---:|---:|---:|
| Standard Test | A | 6.05 | +0.00 | 52.77 | +0.00 |
| Full-Decodable Test | A | 5.26 | +0.00 | 49.12 | +0.00 |
| Standard Test | B | 6.83 | +0.78 | 52.74 | -0.03 |
| Full-Decodable Test | B | 5.28 | +0.02 | 49.09 | -0.03 |
| Standard Test | C | 6.82 | +0.77 | 52.60 | -0.17 |
| Full-Decodable Test | C | 6.01 | +0.75 | 48.91 | -0.21 |
| Standard Test | D | 6.69 | +0.64 | 51.06 | -1.71 |
| Full-Decodable Test | D | 5.89 | +0.63 | 47.49 | -1.63 |

A 为无 Clip Loss 的 GMR→GEQ 控制组；B 只在 GEQ 加 Clip Loss；C 只在 GMR 加；D 两阶段均加。C/D 严格继承同一 λ=0.1 的有 Clip Loss GMR best。

本次单 seed 结果：B 的 Standard Test mAP 增益最大（+0.78 个百分点），C 的 Full Test mAP 增益最大（+0.75）。三组 G1 均值均未超过 A，D 下降更明显，因此不应声称综合检索性能改善。

Clip Head 的 Val 诊断显示明显低召回：以固定 0.5 阈值统计，B precision/recall 均为 0，C recall=0.00820，D recall=0.00820。普通 BCE 下存在偏向低概率输出的问题；当前证据不足以证明语义时序辨识能力获得有效提升。未为改善 Test 成绩追加权重或阈值搜索。

Standard Test 覆盖 1036 条；Full-Decodable Test 覆盖 2970/2986 条。指标按 0–100 报告，G1 为 0.4/0.5/0.6 三点算术均值。mAP 使用未经过存在性门控的正查询候选。两个 Test 集存在重叠，不能视为独立复现。

A 的 Stage 1 GMR Test mAP 为 Standard 6.10、Full 5.19；C/D 共享的 Stage 1 GMR 为 Standard 6.76、Full 6.33。GEQ 阶段的 A 为 6.05/5.26，C 为 6.82/6.01，D 为 6.69/5.89。上表统一比较相同阶段的 A/B/C/D。

各组使用 seed=2023 和相同原生损失、最大轮数、早停规则；实际停止轮次不同。单 seed、不同早停时长不能完全排除训练时长和随机性影响。

用户后续授权同时使用两张 GPU、每卡最多 3 个任务，并要求提前开始 C/D。由此提前训练了 λ=0.1/0.3/1.0 三个 C/D 共享 GMR 父模型；C_10 从 GPU0 完整 epoch checkpoint 迁移到 GPU1，其余两组未中断。最终 C/D 的 λ 仍取 B 的 Val 选择，不依据额外 GMR 或 Test 成绩改变。

共完成 10 个训练阶段（含额外两个 GMR 候选）和 12 组固定 Test 评估。全部阶段完成标记、best/last 哈希、父 checkpoint、早停、C 头冻结、D 头更新及 Test 指标重算已审计。原只认 GPU1 的审计失败记录保留；后续双 GPU 审计通过。

- [完整指标与各 tIoU、短事件、多区间子集](DETAILS.json)
- [全部指标 CSV](METRICS.csv)
- [主审计](COMPLETION_AUDIT.json) 与 [额外候选及迁移审计](SUPPLEMENTARY_AUDIT.json)
- [训练总图](training_val_panel.png)
- [固定 Test 对比图](COMPARISON.png)
- [Val 轨迹](VAL_CURVES.png)

训练、评测和诊断已完成，无需继续后台训练。
