# 清空上下文后从这里开始

本目录保存新增 DETR 骨干的调研和完整工作提示词。当前只完成文档，没有移植新骨干、启动训练、微调或定时队列。

## 文档

- [REPORT.md：调研结论与适配难点](REPORT.md)
- [SOURCES.json：官方仓库、核验 commit、源码及许可文件记录](SOURCES.json)
- [NEXT_SESSION_PROMPT.md：完整可执行工作提示词](NEXT_SESSION_PROMPT.md)

调研日期为 2026-11-02。首批建议 **BM-DETR、BAM-DETR、UVCOM**；后续再考虑 TR-DETR、TaskWeave 等。完整清单包含优先级、输入/监督要求和不计入独立 DETR 骨干的情况。此次只做静态来源与代码接口核验，所有新骨干仍须真实 debug 和正式训练。

## 在新会话复制这段话

```text
请在 /home/guoxiangyu/SoccerGMR-GEQ 仓库、soccergmr-standard-geq-full 分支工作。
先读取 docs/backbone_expansion/START_HERE.md、REPORT.md、SOURCES.json 和 NEXT_SESSION_PROMPT.md。
我现在授权你按 NEXT_SESSION_PROMPT.md 执行首批 BM-DETR、BAM-DETR、UVCOM 的适配、逐个debug、正式GMR训练、GEQ Full微调及Standard/Full测试。
先把首批全部debug通过并冻结协议，再启动正式队列；禁止按Test调参，不启动其他代理，不覆盖现有结果。
请实际执行，不要只回复计划。
```

若只想阅读和讨论，在新会话说明“只读调研，不执行训练”即可。完整提示词中的具体数据路径、选模、早停、恢复、并发和验收规则已写明，无需依赖旧聊天记录。

## 必须保留的上下文

研究问题：现有 GMR 的大量存在性误判造成 G-mIoU 损失，检验“先训练 GMR、再 GEQ Full 微调”能否在更多结构上缓解损失。更多骨干是扩大有效性证据，不应预设所有指标都提高。

训练只用 SoccerGMR Standard Train；GMR 按 Standard Val mAP 选 best；GEQ Full 按 Val G-mIoU@1 在存在性阈值 .4/.5/.6 下的均值选 best，连续十个 epoch 不提升早停。默认首批 seed2023，不自动增加其它骨干或多 seed。正式任务为六个训练阶段、十二组 Test。

两阶段固定 best 分别测试 Standard 和 Full-Decodable 2970/2986，使用相同 checkpoint 配对；Full 不参与训练或调参。新协议保留完整加载至 128 video tokens，不恢复成静默截75步。当前 README 图是历史六轮早停、75-token 检查点的正式测试，并非新协议收敛结果。

只有本次明确要执行完整提示词时，才进入新训练任务；以前 HANDOFF 对“上一阶段暂不正式重训”的说明不应被误读为禁止用户在新会话授权的新任务。现有五骨干、历史产物和其它 GPU 任务保持保护。
