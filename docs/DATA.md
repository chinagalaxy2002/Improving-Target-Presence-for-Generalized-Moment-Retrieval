# 数据与输入接口

## 原始标签及构成

本仓库发布的是 Soccer-GMR 的 Standard 标签，位于 [shared/labels/](<../Phase2(addbackbone_with_S4_minusFusion)/shared/labels/>)。本研究使用这些原始标签，不进行数据增强。下表由仓库标签逐行重算。

| 划分 | 查询数 | 独立视频 ID | 无目标 | 单目标 | 多目标 | GT 区间总数 | 单查询最多区间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 4,138 | 1,369 | 2,002 | 1,423 | 713 | 3,038 | 6 |
| Val | 465 | 127 | 210 | 165 | 90 | 365 | 4 |
| Test | 1,036 | 461 | 492 | 384 | 160 | 750 | 5 |

Test 的正查询共 544 条。所有已记录 GT 满足 `0 <= start < end <= duration`。完整统计、来源构成和标签哈希见 [dataset.json](audit/dataset.json)。

## 标签字段

每行 JSON 表示一个查询与视频的配对。主要字段如下。

| 字段 | 含义 |
| --- | --- |
| `qid` | 查询实例 ID；预测与标签按此字段关联 |
| `vid` | 视频及特征标识 |
| `query` | 原始自然语言查询 |
| `duration` | 视频时长，单位秒 |
| `relevant_windows` | 完整 GT 区间列表；每个区间为秒单位的 `[start, end]` |
| `dataset_source` | 来源标记：`worldcup2022` 或 `sportsmoments` |
| `match_info`、`action_type`、`moment` | 标签保留的来源及事件信息 |

存在性标签为 `int(len(relevant_windows) > 0)`。`[]` 是有效的无目标标签，不能在 GMR 训练或评估时被删除。多目标定义为至少两个 GT 区间。

Train 有两条查询包含六个 GT。FlashVTG 数据加载器的原生 `max_windows=5` 会限制原生 span-label 张量，但 S4 精修损失读取 metadata 中的完整 `relevant_windows`。不同主干原生标签设置应以各自配置为准，不能把同一限制推广到所有主干。代码依据为 [FlashVTG dataset.py](<../Phase2(addbackbone_with_S4_minusFusion)/flashvtg/baseline_source/training/flash_vtg_gmr/dataset.py>) 和各主干 `configuration/CONFIG.json`。

## 预提取特征

| 输入 | 维度与顺序 |
| --- | --- |
| 视觉特征 | SlowFast 2304 → CLIP 512 → TEF 2，合计 2818 维 |
| 文本特征 | CLIP 文本特征，512 维 |
| 模型内部表示 | 256 维投影特征 |
| 时间网格 | 名义 clip 长度为 2 秒，最多 75 个 clip |

视频和文本使用原有预提取特征。各流按有效长度对齐，视觉流分别归一化后拼接，TEF 表示接收视频自身的归一化时间位置。有效位掩码必须正确传递，注意力不能读取 padding。文本截断长度和文件命名由各主干 loader/config 指定，不应混用其他主干的数据入口。

时长并非全部为 150 秒。模型中的 `/150` 是固定几何归一化尺度，不代表每条视频的实际时长。四个 DETR 适配器使用实际时长转换并约束候选边界；FlashVTG保留其原生时间戳后处理。

GitHub 发布包不含视频、视觉/文本特征数组或二进制检查点。标签、预测、配置和代码可用于 CPU 指标复核；重新训练或模型推理需要另行配置对应资产。原始发布范围见 [PUBLICATION_SCOPE.json](../PUBLICATION_SCOPE.json)。

## 来源构成与划分边界

| 划分/来源 | 无目标 | 单目标 | 多目标 |
| --- | ---: | ---: | ---: |
| Train / WorldCup2022 | 2,002 | 720 | 713 |
| Train / SportsMoments | 0 | 703 | 0 |
| Val / WorldCup2022 | 210 | 84 | 90 |
| Val / SportsMoments | 0 | 81 | 0 |
| Test / WorldCup2022 | 492 | 168 | 160 |
| Test / SportsMoments | 0 | 216 | 0 |

全部无目标和多目标记录来自 WorldCup2022；SportsMoments 只有单目标正例。因此整体存在性指标可能受到来源与类别关联的影响；这并不直接证明模型使用了来源捷径。仅含正例的 SportsMoments 子集不能单独计算二分类 AUROC。

| 划分对 | 重复 qid | 重复 vid | 相同查询字符串 | 相同 match_info |
| --- | ---: | ---: | ---: | ---: |
| Train / Val | 0 | 0 | 320 | 27 |
| Train / Test | 0 | 0 | 599 | 31 |
| Val / Test | 0 | 0 | 138 | 27 |

本划分在已记录的视频 ID 层面互斥，但存在相同比赛元数据，不能称为比赛互斥划分。查询字符串在不同视频中重复本身不等价于样本重复。本审计没有通过文件名推断其他比赛关系，也不支持“未见动作”“未见球队”等额外泛化结论。
