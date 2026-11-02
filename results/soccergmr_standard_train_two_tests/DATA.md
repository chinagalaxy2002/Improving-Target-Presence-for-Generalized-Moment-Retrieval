# 数据、测试协议与覆盖

## 划分

| 数据 | 行数 | 正例 | 负例 | 本目录标签 |
|---|---:|---:|---:|---|
| Standard Train | 4138 | 2136 | 2002 | data/labels/standard_train.jsonl |
| Standard Val | 465 | 255 | 210 | data/labels/standard_val.jsonl |
| Standard Test | 1036 | 544 | 492 | data/labels/standard_test.jsonl |
| Full-Decodable Test | 2970 | 1633 | 1337 | data/labels/full_decodable_test.jsonl |
| 原始Full Test | 2986 | 1638 | 1348 | data/labels/full_original_test.jsonl |

完整身份、原路径、GT SHA256和特征协议见[data/DATASETS.json](data/DATASETS.json)及[evidence/original/DATA_AUDIT.json](evidence/original/DATA_AUDIT.json)。本目录Formal Full结果只覆盖2970条，不能称为原始2986条全覆盖。

历史未覆盖16条（5正例、11负例）在[data/labels/full_original_excluded_16.jsonl](data/labels/full_original_excluded_16.jsonl)，源于两个视频ffprobe失败：`WC2022_3869685_1_2240s_2390s`和`AC_Ajaccio_Lens_QsKULW3B_1_1960s_2110s`。没有新增静默跳过。原始Full标签供核对排除关系，不能拿它与2970条预测直接交集后声称全覆盖。

Full-Decodable的2970条包含1008种原始vid字符串，但去扩展名后为823个特征stem；同一视频可能带/不带扩展名出现。只在查找特征时去扩展名，GT/预测的qid、vid和query身份严格保留。Full特征共有823×2+2970=4616个唯一文件。

## 特征与规范化

Standard原特征根目录为`/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr/`；Moment记录的原目录是项目下`mr_experiments/features/`。Full必须换成`Soccergmr/Full_features/`及Full的独立qid文本，不能复用Standard预测/文本。

Full协议为`soccer_full_lighthouse_standard_bn45_v1`，原manifest SHA256为`a56bea0061a65076b05e9556f15b4496ef5d0839ce3ef02a7195009e43a69a8b`。原manifest/VERIFIED/EXCLUSIONS在[data/feature_protocol/](data/feature_protocol/)。特征本体没有上传；原路径哈希在[data/FULL_FEATURE_SHA256.json](data/FULL_FEATURE_SHA256.json)及[data/REGISTERED_FEATURE_SHA256.json](data/REGISTERED_FEATURE_SHA256.json)。后者登记Canonical Standard训练/验证/测试和Full测试特征，不应将其误认为已经上传了这些数组。

GT直接使用`relevant_windows`，保留自然负例的空数组及多GT；不从`moment`再次扩窗。视频CLIP512/SlowFast2304各自L2归一化并加TEF2；Moment流顺序CLIP→SlowFast，其他三个骨干SlowFast→CLIP。文本保持原生前32 token与mask规则，不因为CLIP输出有padding而自行改变token处理。

Full模型使用原75步上限，两段94/104步视频截断前75步；输出仍按原duration缩放。另3段视频的特征步数×2与GT duration偏差超过2秒。具体视频/qid/长度在DATASETS.json的`full_temporal_caveats`中。Moment另保留原生150秒时间后处理上限。所有查询均完成推理，但这不等于所有视频的全部时间范围都被编码；修复时间处理必须另立协议并统一重评对照。

## 评估定义

定位mAP/mR/mR+/mIoU/mIoU+只在正例未门控候选上计算。存在性AUROC、N-acc、Rej-F1和G-mIoU在完整正负域计算。阈值τ严格按`pred_exist_score > τ`判存在，否则预测集合为空。N-acc=TN/N_neg；Acc是整体准确率，两者不能混用。

G-mIoU用官方集合IoU：空/空为1；只有一方空为0；两集合非空采用原生贪心匹配和集合归一化。官方raw metrics保存百分数，而data/RESULTS和所有图表将G-mIoU转换到0–1一次。其它指标保持百分数；差值只相对同骨干同Test的GMR。

Frozen先逐张量确认原生权重等于父模型，再逐字段复制父模型同一测试域的候选（边界、顺序、置信度及其他字段），仅替换存在性分数。Full候选由父模型在Full上独立推理生成，不能取Standard或GEQ Full候选给Frozen。GEQ Full本身独立推理，保留其全部已微调权重。两个测试域共16项Frozen一致性检查见AUDIT。

oracle存在性上限对每个模型保留相同top-k候选，正例全接收、负例全拒绝，再用官方函数重算。图中每个子图只展示六方法中最高的oracle值，不是阈值扫描最大值，也没有用GT替换定位窗口。固定扫描τ=0.00–1.00、步长0.01仅作描述，正式报告三点始终为.4/.5/.6。

## 重叠与后续使用

原始Full Test与Standard Test共享1036个规范化视频stem+查询对；原始Full Test与Standard Train/Val的规范化video stem初查交集为0。两个Test不能作为相互独立的实验样本。

这些Test及曲线已经看过并参与形成研究问题。后续在Standard Train/Val上优化、明确记录初始化和选模，Full上的改进保持探索性说明；不能重新声称是未接触的Test。若引入Full Train/Val训练、校准或修正时间处理，应另立设置，不继续沿用原“仅Standard训练且协议不变”的标签。
