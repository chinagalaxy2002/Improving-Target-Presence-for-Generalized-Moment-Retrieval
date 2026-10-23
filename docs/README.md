# 仓库研究文档

新增正式30组组件消融及独立额外融合诊断，见 [新增消融实验](../ablations/component_ablation_20261023/README.md)；本页原有主结果数值保持历史口径。

本文档仅依据本 GitHub 仓库已经发布的代码、配置、Standard 标签、冻结预测和结果表撰写。研究范围为五个主干的 Baseline、GMR 和 GMR + S4_minus_Fusion 对比；使用原始数据，不包含数据增强实验。

| 文档 | 内容 |
| --- | --- |
| [数据说明](DATA.md) | 标签结构、特征接口、数据构成及划分检查 |
| [算法说明](ALGORITHM.md) | 全局存在性查询、Gaussian RoI、边界精修、LoRA 和训练损失 |
| [实验说明](EXPERIMENTS.md) | 14 组本地实验、三个阈值、五主干增益和结论边界 |
| [复现说明](REPRODUCIBILITY.md) | 仅依赖当前仓库的 CPU 审计与已有图表复现入口 |
| [论文段落](PAPER_NOTES.md) | 基于本仓库证据的英文摘要、方法描述和结果表述 |
| [已有代码映射](CODE_MAP.md) | 各主干的模型、训练代码、配置与输出位置 |

审计结果保存在 [audit/](audit/)：包含从标签重新统计的数据、14 组预测在三个阈值下的指标、相对同主干 GMR 的差值以及输入文件 SHA256。使用 [audit_documentation.py](../tools/audit_documentation.py) 可在本仓库直接重算，不需要另一个源工作区。

仓库已有表格中的六组论文 MLLM 参考行属于外部论文转录，不计入本地 14 组实验。本次文档没有引入新的实验、外部文献检索结果或本地工作区的其他实验材料。
