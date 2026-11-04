# 11骨干 Aux-only 结果复用与评测方案

## 结论：不需要重新训练

11骨干结果中此前标为 **GMR baseline** 的方法，实际上已经是本次要求的
**Aux only**：

`原生GMR损失 + E3 query-aware clip辅助监督`

E3为 Balanced BCE 加 `0.2 × Dice`，辅助损失总系数为`0.1`。该阶段没有
GEQ模块、GEQ BCE、GEQ logit或GEQ推理分支。所有checkpoint均使用
`seed=2023`从头训练，并按Standard Val正查询mAP选模。

因此，再启动一个aux-only微调阶段会变成“在已经完成的aux-only模型上继续
训练aux”，不再是所需消融。后续必须复用已经完成的GMR `best.pt`及固定Test
产物；不重训已完成实验，也不安排多seed。

## 已完成的训练与测试流程

每个骨干已经执行了以下Aux-only流程：

1. 在Standard Train上从随机初始化开始训练。
2. 优化骨干原生GMR目标与E3辅助clip损失。
3. 每个epoch在Standard Val上评测。
4. 按Standard Val正查询mAP选择`best.pt`，patience为20，最多200轮。
5. 正式Test前冻结已选checkpoint。
6. 同一个固定`best.pt`分别评测Standard Test与Full-Decodable Test。
7. 保存预测、指标、checkpoint/source/data哈希及完成标记；Test结果不参与
   checkpoint选择，也不反向修改阈值或模型。

覆盖的11个骨干为Moment-DETR、EaTR、QD-DETR、CG-DETR、Sim-DETR、
BM-DETR、BAM-DETR、UVCOM、TR-DETR、TaskWeave和LD-DETR。Moment-DETR
复用已经完成的历史E3 checkpoint并保留原始溯源；其余10个骨干使用统一冻结的
11骨干源码版本。

## 已有产物位置

原始训练目录：

`runs/e3_eleven/formal/<backbone>/gmr/`

固定Test产物：

`runs/e3_eleven/formal/<backbone>/gmr_test/test_standard/`

`runs/e3_eleven/formal/<backbone>/gmr_test/test_full/`

发布结果位于`results/e3_eleven/`，主要包括：

- `METRICS.csv`：11骨干、两个阶段和两个测试集的汇总指标；
- `predictions/<backbone>/gmr/`：Aux-only固定Test预测与指标；
- `provenance/<backbone>/gmr/`：Val历史、选模记录与完成标记；
- `training_val_panel.png/.pdf`：已经完成的训练/Val进度图；
- `STANDARD_TEST.*`与`FULL_TEST.*`：完整存在性阈值曲线；
- `RETENTION86_METRICS.csv`：86% Val正查询保留率扫描。

当前图中灰色、名为 **GMR baseline** 的系列就是Aux only。再增加一条
“Aux only”曲线会重复同一批数值。

## 下一步只读测试与审计

清空上下文后继续时，不启动训练，按以下顺序直接审计既有结果：

1. 核对每个`best.pt`哈希与GMR完成标记一致。
2. 核对源码身份、配置身份和Train/Val资产哈希。
3. 确认checkpoint阶段为`gmr`，模型含query-aware辅助头且不含GEQ参数。
4. 确认Standard Test为1,036条，Full-Decodable Test为2,970条。
5. 从已保存预测和GT重新计算mAP与G-mIoU，并与metrics文件逐项比较。
6. 从固定预测重新计算G-mIoU@1/@3的完整阈值曲线，不重新训练模型。
7. 确认checkpoint在Test之前已经选定，且`test_used_for_selection=false`。
8. 直接复用已完成的训练进度图；11个训练历史都已结束，不需要实时训练监控。

如果某个Test产物缺失或损坏，只从已经选定的`best.pt`重跑对应Test推理。
只有当选定checkpoint本身无法通过哈希或溯源审计时，才讨论重新训练。

## 与其他方法的正确命名关系

- **GMR no-aux baseline**：仅原生GMR；
- **Aux only**：当前灰色系列，即原生GMR加E3辅助监督；
- **GEQ only**：无aux的GMR后接GEQ；
- **GEQ+aux**：辅助训练的GMR后接GEQ，并在第二阶段继续使用aux。

当前导入的GEQ-only参考结果并非所有骨干都共享同一个Stage-1父模型和训练协议，
因此它与Aux-only的差值属于绝对结果参考，不能直接解释为严格的aux因果增益。
若后续需要严格四路受控对比，应另外冻结统一的no-aux GMR父模型，再从同一父模型
建立匹配的第二阶段分支；不能通过重新命名或重复训练现有Aux-only checkpoint来伪造
这一对照。
