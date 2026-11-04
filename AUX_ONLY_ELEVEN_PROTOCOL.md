# 11骨干 GMR(aux) 结果复用与四方法评测方案

## 结论：不需要重新训练

11骨干结果中此前误标为 **GMR baseline** 的方法，实际上是
**GMR(aux)**：

`原生GMR损失 + E3 query-aware clip辅助监督`

E3为 Balanced BCE 加 `0.2 × Dice`，辅助损失总系数为`0.1`。该阶段没有
GEQ模块、GEQ BCE、GEQ logit或GEQ推理分支。所有checkpoint均使用
`seed=2023`从头训练，并按Standard Val正查询mAP选模。

因此，再启动一个GMR(aux)微调阶段会变成“在已经完成的GMR(aux)模型上继续
训练aux”，不再是所需消融。后续必须复用已经完成的GMR `best.pt`及固定Test
产物；不重训已完成实验，也不安排多seed。

## 已完成的训练与测试流程

每个骨干已经执行了以下GMR(aux)流程：

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
- `predictions/<backbone>/gmr/`：GMR(aux)固定Test预测与指标；
- `provenance/<backbone>/gmr/`：Val历史、选模记录与完成标记；
- `training_val_panel.png/.pdf`：已经完成的训练/Val进度图；
- `STANDARD_TEST.*`与`FULL_TEST.*`：完整存在性阈值曲线；
- `RETENTION86_METRICS.csv`：86% Val正查询保留率扫描。

当前图中灰色、原名 **GMR baseline** 的系列就是GMR(aux)，必须更名为
**GMR(aux)**。再增加一条GMR(aux)曲线会重复同一批数值。

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

## 后续测试的固定执行协议

后续不是重新训练GMR(aux)，而是把已有固定checkpoint和预测整理成能够进入目标
汇总图的统一数据。必须严格按以下步骤执行。

### 1. 先固定方法名与数据来源

| 图中方法名 | 实际模型 | 数据来源 | 是否含GEQ |
|---|---|---|---|
| GMR baseline | 仅原生GMR | `soccergmr-standard-geq-full`分支中GEQ训练前的GMR父模型 | 否 |
| GMR(aux) | 当前11骨干结果中原名`gmr`/`GMR baseline`的模型 | 当前E3 11骨干固定结果 | 否 |
| GEQ only（reference） | GMR baseline后接原始GEQ | `soccergmr-standard-geq-full`分支 | 是 |
| GEQ+aux | GMR(aux)后接GEQ，并在第二阶段保留aux | 当前E3 11骨干固定结果 | 是 |

**禁止事项：**不能把当前E3 `gmr`数据同时画成“GMR baseline”和“GMR(aux)”
两条柱；二者是同一批checkpoint和预测。要形成四方法图，必须从参考分支另行
读取真正的原始GMR baseline结果。若原始GMR baseline的Val或Test预测无法完整
审计，则先生成三方法图，只把当前灰色系列改名为`GMR(aux)`，不得用复制数据
补出第四条。

### 2. 对四种方法分别做Standard Val阈值校准

继续沿用当前SUMMARY图的86%工作点，不改变规则：

1. Standard Val共有255条正查询。
2. 候选存在性阈值为`0.00–1.00`，步长`0.01`。
3. 对每个骨干、每种方法独立选择能够保留至少220条正查询的最大阈值；实际
   最低保留率为`220/255=86.27%`。
4. 判定规则固定为`pred_exist_score > threshold`，不是`>=`。
5. 每种方法选出的阈值固定后，同时用于Standard Test和Full-Decodable Test。
6. 禁止在任一Test集合上重新选阈值、微调阈值或重新选checkpoint。

需要保存标准化结果文件，其中每行至少包含：骨干、方法、Val阈值、Val保留正例
数、Val实际保留率、Test split、mAP、G-mIoU@1、G-mIoU@3、Test TPR/FPR、
checkpoint哈希、Val/Test预测哈希及来源提交。

### 3. 固定Test复算内容

对每种方法和每个骨干执行以下只读复算：

- Standard Test：1,036条查询；
- Full-Decodable Test：2,970 / 2,986条可解码查询；
- 正查询未门控mAP；
- 所选86%工作点上的G-mIoU@1和G-mIoU@3；
- 固定存在性阈值`0.3/0.4/0.5/0.6`的G-mIoU@1/@3，供附表与审计使用；
- 完整`0.00–1.00`、步长`0.01`曲线只作为补充数据，不替换已有的全11骨干
  `STANDARD_TEST.*`与`FULL_TEST.*`图。

正式汇总图的主指标仍为86% Val保留率工作点上的 **G-mIoU@1**。mAP不使用
存在性门控，应放在README表格或单独图中，不能与G-mIoU共用同一组横柱以免
误导。

### 4. 固定展示骨干，禁止重新筛选

目标图继续使用当前已经公开的9个骨干及顺序：

1. EaTR
2. QD-DETR
3. CG-DETR
4. Sim-DETR
5. BM-DETR
6. BAM-DETR
7. UVCOM
8. TR-DETR
9. TaskWeave

Moment-DETR和LD-DETR继续留在完整CSV与全11骨干审计中。加入No-aux GMR或
GMR(aux)正确命名后，不得重新查看Test并改变这9个骨干的交集，否则会形成新的
Test后筛选偏差。图注继续明确这是既有探索性9/11子集。

## 目标对比图的精确定义

需要更新的就是以下两张README汇总图：

目标参考图：
[STANDARD_SUMMARY.png](https://github.com/chinagalaxy2002/Improving-Target-Presence-for-Generalized-Moment-Retrieval/blob/moment-detr-clip-supervision/results/e3_eleven/STANDARD_SUMMARY.png)

- `results/e3_eleven/STANDARD_SUMMARY.png/.pdf/.svg`
- `results/e3_eleven/FULL_SUMMARY.png/.pdf/.svg`

并同步更新两页合集：

- `results/e3_eleven/TEST_SUMMARY_BOTH.pdf`

### 四方法图的视觉编码

每个骨干一行，绘制四条横向分组柱，顺序在两张图中完全一致：

| 顺序 | 方法 | 颜色 | 标记 |
|---:|---|---|---|
| 1 | GMR baseline | 浅灰 | 方形 |
| 2 | GMR(aux) | 紫色 | 菱形 |
| 3 | GEQ only（reference） | 橙色 | 三角形 |
| 4 | GEQ+aux | 蓝色 | 圆形 |

Standard与Full必须使用相同骨干顺序、相同0起点和相同横轴上限。每条柱末端标注
实际G-mIoU@1数值。为避免标签含义混乱，差值固定定义为：

- `ΔAux = GMR(aux) − GMR baseline`；
- `ΔGEQ(no-aux) = GEQ only − GMR baseline`；
- `ΔGEQ(aux) = GEQ+aux − GMR(aux)`。

图标题列出9骨干四种方法的等权平均。图注必须写明每种方法独立在Standard Val
校准阈值，并使用同一阈值测试Standard与Full。图内不得再使用含义不明确的
`GMR baseline`与`GMR(aux)`不得混用。

如果原始GMR baseline数据未通过审计，则目标图暂时保留三条柱：`GMR(aux)`、
`GEQ only (reference)`、`GEQ+aux`；此时只做正确重命名，不制作虚假的四方法图。

### README需要同步表达的内容

图上方说明必须包含：

1. 当前原`GMR baseline`已经更名为`GMR(aux)`；
2. 四种方法的训练组成与父模型来源；
3. 86% Val阈值校准规则与220/255的实际下限；
4. 9骨干子集是此前Test后固定的探索性子集，未因新增方法重新筛选；
5. Standard和Full各自的四方法9骨干均值；
6. 可展开的逐骨干表格，列出四种Val阈值、四个Standard G1和四个Full G1；
7. reference分支与当前分支的父模型/训练协议差异。

## 制图与结果完整性检查

生成图片后必须通过以下自动检查，全部通过后才能替换README图：

1. 四方法完整时共`9 × 4 × 2 = 72`根柱，且72个数值标签逐项来自CSV；
2. 每个方法、每个骨干在Standard和Full使用完全相同的Val阈值；
3. 图中三类差值由未四舍五入值计算，显示时再保留两位小数；
4. Standard与Full横轴范围一致并从0开始；
5. PNG人工检查无文字重叠，PDF/SVG保持可编辑矢量文本；
6. 绘图脚本连续运行两次，PNG/PDF/SVG哈希保持一致；
7. README中的均值和展开表格由同一份标准化CSV生成，禁止手工抄写第二套数值；
8. 原有全11骨干`STANDARD_TEST.*`与`FULL_TEST.*`六个文件哈希保持不变；
9. 审计JSON记录所有输入CSV、预测、脚本、输出图和参考提交哈希；
10. 不进行新训练；只有预测文件缺失或损坏时，才从已经冻结的best checkpoint
    重跑对应Test推理。

## 结果解释边界

四方法图首先是结果层面对比，不自动构成严格的四路因果消融。当前参考分支中
EaTR、QD-DETR、CG-DETR和Sim-DETR沿用历史patience6 / 75-token协议，另外
5个展示骨干使用patience10 / 128-token扩展协议；当前GMR(aux)/GEQ+aux则来自
E3 11骨干流程。

因此，最可靠的同父模型比较是当前分支内部的
`GEQ+aux − GMR(aux)`；参考分支内部的`GEQ only − GMR baseline`也应保持成对解释。
跨分支的`GMR(aux) − GMR baseline`只能标为参考差值，不能单独归因于aux。README
和图注必须保留这一限制，不能使用“aux必然带来X个百分点提升”之类的因果表述。

## 与其他方法的正确命名关系

- **GMR baseline**：仅原生GMR；
- **GMR(aux)**：当前灰色系列，即原生GMR加E3辅助监督；
- **GEQ only**：无aux的GMR后接GEQ；
- **GEQ+aux**：辅助训练的GMR后接GEQ，并在第二阶段继续使用aux。

当前导入的GEQ-only参考结果并非所有骨干都共享同一个Stage-1父模型和训练协议，
因此它与GMR(aux)的差值属于绝对结果参考，不能直接解释为严格的aux因果增益。
若后续需要严格四路受控对比，应另外冻结统一的no-aux GMR父模型，再从同一父模型
建立匹配的第二阶段分支；不能通过重新命名或重复训练现有GMR(aux) checkpoint来伪造
这一对照。
