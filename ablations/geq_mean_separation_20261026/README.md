# FlashVTG + GEQ：正负均分分离监督实验方案

本目录只新增两组 FlashVTG 实验：GAP 与 ENDPOINT，各微调4轮、seed=2026。现已实现训练与自动评估入口，GAP/ENDPOINT 均通过 preflight 并正常后台训练。第4轮结束自动进行官方 Val/Test、独立复算及汇总绘图；最终性能尚待实测。

目标是检验：在 E1 的原有逐样本存在性监督之外，增加类别均分约束，能否拉大正负查询的存在性分数均值间隔，并保持存在性排序与定位能力。

## 实验定义

| 配置 | 初始化 | 原有损失 | 新增损失 | 训练预算 |
|---|---|---|---|---|
| E0 参考 | 已训练原始 GMR | 复用已有结果 | 无 | 不新增训练 |
| E1 参考 | 原始 GMR checkpoint + 随机 GEQ | Native GMR + GEQ BCE | 无 | 复用已有4轮结果 |
| GAP | 与 E1 相同 | 与 E1 相同 | 均分间隔平方 hinge | 4轮 |
| ENDPOINT | 与 E1 相同 | 与 E1 相同 | 正负均分分别平方 hinge | 4轮 |

GAP、ENDPOINT 独立从原始预训练 GMR checkpoint 初始化，不从 E1 第4轮或任何 E2/E3/E4 权重续训。GEQ 结构和初始参数必须与 E1 一致。保留 E1 的 native presence BCE、GEQ BCE 和原生定位损失；它们与 scratch E3 移除 native presence BCE 的方案不同。

不加入 PV、LoRA、边界 refiner、新 saliency、teacher、合成负样本、额外数据或存在性融合。新增均分损失只用于训练，不改变官方推理路径。

详细定义见 [PROTOCOL.md](PROTOCOL.md)，实施与验收见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。固定配置位于 configs/，排队意图位于 QUEUE_PLAN.json；实际任务进程见 JOB_GAP.json、JOB_ENDPOINT.json 与 runs/*/STATUS.json。

## 交付与状态

- `configs/GAP.json`、`configs/ENDPOINT.json`：训练前固定的两组配置。
- `SOURCE_MANIFEST.json`：拟复用的 E1 实现、初始化记录与数据划分的 SHA256；计划阶段未重哈希大 checkpoint；本次执行已重新验证原始 checkpoint 字节哈希，见 preflights/*/CONFIG.json 与 runs/*/CONFIG.json。
- `STATUS.json`：真实交付状态，当前状态见文件及各 runs/*/STATUS.json，只有完成实际评估后标记 complete。
- `QUEUE_PLAN.json`：主会话后续执行用的队列方案，本次已更新为实际启动记录，进度以 runs/*/STATUS.json 为准。
- `RESULTS_TEMPLATE.md`：仅提供结果字段，没有实测值。

主会话已按 IMPLEMENTATION.md 使用两个工作单元分别完成 GAP/ENDPOINT preflight 与启动；原侧边会话的计划记录保留在协议文件中。

## 运行与复现

使用 `/home/guoxiangyu/miniconda3/envs/gmr/bin/python`。原生训练入口为 `train.py`，默认固定4轮；共享新增损失位于 `separation_losses.py`，来源版本在每组 CONFIG/PREFLIGHT 中冻结。旧实验代码和证据不修改。

```bash
CUDA_VISIBLE_DEVICES=0 /home/guoxiangyu/miniconda3/envs/gmr/bin/python train.py --setting GAP --preflight-only
CUDA_VISIBLE_DEVICES=1 /home/guoxiangyu/miniconda3/envs/gmr/bin/python train.py --setting ENDPOINT --preflight-only
/home/guoxiangyu/miniconda3/envs/gmr/bin/python train.py --setting GAP --launch --gpu 0
/home/guoxiangyu/miniconda3/envs/gmr/bin/python train.py --setting ENDPOINT --launch --gpu 1
```

以上命令在本目录执行；已存在的 preflight/run 不覆盖，重现实验需独立目录。每卡最多两个实际 GPU 进程，当前两组各占一张卡。状态及损失见 `runs/{GAP,ENDPOINT}/STATUS.json`、`history.jsonl`、`batches.jsonl`；后台日志见 `logs/GAP.log`、`logs/ENDPOINT.log`。

第4轮结束严格加载 `epoch_004.ckpt`，保留官方原生推理，自动运行官方 Val/Test 和独立指标复算；`evaluate.py` 计算全 split 端点距离及相对 E1 的 lost/restored；`summarize.py` 使用文件锁生成 E0/E1/GAP/ENDPOINT 报告、CSV及矢量图。若只有一组完成，报告标记 partial，不填补另一组结果。E0 为原始 GMR，E1 为 GEQ 微调4轮，两者独立列出。

新增损失梯度范数在第1步及每25步采样；每批记录类别数、均分与约束激活状态；epoch日志保存混合batch比例、全Val类别均分以及实际loader顺序哈希。验证时保留 RNG 状态；正式初始化不继承 preflight 更新。
