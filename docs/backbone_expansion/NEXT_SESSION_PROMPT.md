请直接执行下面的新增骨干适配、正式训练、微调和测试任务。这次明确授权训练下面三个新骨干，覆盖仓库 HANDOFF 中“前一阶段暂不正式重训”的旧阶段说明；不重训现有五骨干，不启动其他代理。

【研究目标】
我们发现现有 GMR 的大量事件存在性误判会造成 G-mIoU 性能损失。要检验统一的“先训练 GMR baseline，再用 GEQ Full 微调”范式能否缓解这种损失，并拓展到更多 DETR 时刻检索骨干。需要保留失败和下降结果；不能预设每个模型都应提高，也不能把更多变体数量当作对所有架构有效的证明。

【先读这些文件】
- 独立仓库：/home/guoxiangyu/SoccerGMR-GEQ
- 唯一跟踪分支：soccergmr-standard-geq-full；代码与历史图的基准提交为 9e3ba902119af8ef7bc5f268aece29dbac8e7687，调研文档在后续提交；实际以当前 HEAD 为准。先确认当前 git 状态和远端，保留用户已有改动，不强制重置或强推。
- 阅读仓库 README.md、HANDOFF.md、docs/ALGORITHM.md、docs/EXTENDING.md、docs/REVIEW.md，以及 src/soccergmr/{config.py,train.py,checkpoint.py,geq.py,predict.py,backbones/registry.py,backbones/loss.py}。
- 调研报告：/home/guoxiangyu/SoccerGMR-GEQ/docs/backbone_expansion/REPORT.md
- 上游版本与文件证据：/home/guoxiangyu/SoccerGMR-GEQ/docs/backbone_expansion/SOURCES.json
- 当前五骨干已通过真实小样本功能验证，但这不是新协议的收敛结果。README 中 Standard/Full 正式图是历史六轮早停、75-token 检查点的配套测试，不得冒充新协议成绩。不要导入 archive 的外部路径训练脚本。

【本次目标骨干：仅这三个】
1. bm_detr： https://github.com/minjoong507/BM-DETR ，调研版本 b9afb3d56baa5c4a999e06e1740a744c7a73c2d7
2. bam_detr： https://github.com/Pilhyeon/BAM-DETR ，调研版本 914b3e710afdde813f1ba2979da3c69e28f4c549
3. uvcom： https://github.com/EasonXiao-888/UVCOM ，调研版本 6faab33a821fe045f0ef007e63a8ecc2a4e443bd
TR-DETR、TaskWeave、LD-DETR 是后续储备，本次不自动增加。不要将 LA-DETR 的多个父骨干版本当作独立新骨干，也不要把 UMT/FlashVTG 的 dense head 强行计为标准 DETR decoder。

【数据】
只在 SoccerGMR Standard Train 训练，Standard Val 选模。
- Standard 标注：/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard/{train,val,test}.jsonl
- Standard 特征：/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr
- Full-Decodable Test 标注：/home/guoxiangyu/GMR_Unseen_s4_minus_fusion_baseline_20261019/experiments/gmr-Global-Eventquery/data/full_decodable/Full/test.jsonl
- Full 特征：/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr/Full_features
- Standard Train/Val/Test 应为 4138/465/1036 条；Full 当前为 2970/2986 条，报告时明确是 Full-Decodable，不自动排除新样本、不暗称完整2986条。
- Full 使用独立的 clip_text qid 命名空间。先核验标签/特征 SHA256、覆盖、维度、有限值和视频泄漏。
- 沿用新协议实际 duration 的 target/export 坐标和最大128 video tokens；不得恢复成静默截75步再按整段duration缩放。Full 的94/104步视频只能做无梯度前向检查，不能用于训练或选参。

【统一训练协议】
seed=2023。先完成首批单 seed，不自动扩展多 seed。
A. GMR baseline：保留对应骨干原生时刻检索结构和核心损失。上游若没有 GMR 视频级存在性 head，按预先固定的一致读出规则增加 head（如最终 decoder 查询 max-pool + 两层 MLP，输入维度随原生表示），对所有真实正/null 查询做 BCE；不能把 BM 的逐帧 vid_logits 直接冒充视频级 logit。明确记录为适配后的 backbone-GMR，不声称上游原版已支持 GMR。
   Standard Val 正查询 mAP 最高选 best；每 epoch 完整验证；同分不替换 best；默认连续20轮不提升早停，最多200轮。
B. GEQ Full：只从该骨干本次正式 GMR 的 Val-best 初始化，原生模型与 GEQ 共同更新；损失为原生 GMR loss + GEQ BCE，默认权重1。原生 lr=1e-5，GEQ lr=3e-5；保留原 GMR head 的训练监督，推理使用 GEQ 存在性分数。
   只按 Standard Val 的 G-mIoU@1 在存在性阈值0.4、0.5、0.6的算术均值最高选 best；严格连续10个epoch没有提升就早停，同分计一次，提升清零；最多200轮。达到上限要记 max_epochs，不能称触发早停。
   GMR 默认 AdamW lr=3e-5、weight_decay=1e-4、clip=.1、有效batch=8；沿用配置中的非AMP设置。若原生方法确有训练调度需求，先检查并明确固化协议，不用Test决定；OOM优先调整microbatch/accumulation保持有效batch并记录。
C. 测试：阶段完成并固定 best 后，同一个checkpoint分别测完整 Standard Test 和 Full-Decodable Test。Test不能选模型、选阈值或调超参数。两阶段同骨干必须采用相同、预先固定的原生候选排序/后处理。

【移植与debug顺序】
1. 核验上游版本/许可，收集真正需要的源文件和依赖，保留原LICENSE、commit与文件hash；所有Python训练/推理代码落在本仓库，不从其他本地项目动态导入。
2. 通过registry、配置、损失及导出adapter扩展，避免复制三份trainer。必要时将GEQ的特征读取点/布局/正分支调用策略做成显式接口，保持共享GEQ核心算法。
   - BM：检查真实null、多GT、负查询和原生背景损失；保留其核心方法，区分合成负配对与真实空查询。
   - BAM：保留边界双路径、原生quality候选排序；内部三参数和输出二维span不可混淆，存在性head维度不能盲写256；encoder多次调用只读取预先声明的正分支。
   - UVCOM：入口在CIM/GKA_Module；不得写死transformer.encoder.layers。核验token偏移、负配对及epoch参数，不能将负分支读入GEQ视频证据。
3. 原生MR及相关损失对自然空查询、全null、多窗口、无背景/全前景、padding做数值安全处理；GT派生saliency只用相应训练标签，不能伪造人工HD标注；不得为了跑通而无说明删除方法核心损失。保留每项修改记录。
4. 每个骨干独立进程，先完成模型构建、真实Train混合正/null、多窗口、全null有限损失和梯度、GEQ及骨干非零梯度、一步优化、strict保存/加载、合法last中断恢复、与连续训练一致性、两套Test无梯度前向和精确qid覆盖检查。源码接口或后处理改动须回归既有五骨干；新增测试应验证实际风险而非镜像实现。
5. 三骨干都debug通过后，冻结源代码、配置和资产身份，再拉起正式队列。当前source identity绑定整个src，不能边训练边随意改其它骨干代码；如有修复，使用受控版本/状态迁移，绝不关闭identity或strict检查。

【正式运行与监控】
- 默认共6个训练阶段、12组Test；GEQ任务依赖各自GMR完成，不是依赖某个中途checkpoint。
- 使用持久化队列、目录锁、completion哈希、原子best/last和RNG/optimizer状态，防重复派发；失败从合法last恢复，不拿best当训练恢复点。
- 先检查GPU和其他用户进程，2张3090上最多每卡2个、全机4个任务；显存不足则降低并发，不杀其他任务。不要启动其他代理/同任务会话。
- 把PID、GPU、日志、任务依赖和完成状态写进独立HANDOFF。长时任务正常运行时可以汇报并结束当前轮，但必须确保队列独立存活。若设置约2小时自动唤醒，先核验真实systemd/唤醒链路；只配置本任务，不重启旧Charades定时器，不把普通sleep当成已配置自动唤醒。
- 现有已测试环境：/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python；新增依赖隔离安装，不覆盖全局环境。旧官方代码环境较老，应真实构建验证兼容性。

【验收与报告】
- 保存每阶段best/last、Val曲线、选择依据、源/数据hash、checkpoint lineage、完整Test预测、官方指标及完成标记，不上传大型权重和数据。
- 报告同骨干GEQ Full相对GMR的mAP、G1/G3三阈值和均值增减，保留负值；补充FP/FN/TN/TP、正查询误拒率/保留率、空查询误保留率、固定候选oracle-presence差距及全部拒识参考值，不能把全部G-mIoU收益等同为存在性错误的减少。
- 仿照当前README的两张图，画Standard/Full阈值曲线、mAP和相对GMR增减，绿色正值/红色负值；新10轮协议与历史6轮协议分开标注。先核验再提交当前分支，不能强推或改原main/历史分支。
- 每个骨干逐项区分“已适配”“debug通过”“正式GMR完成”“GEQ完成”“两套Test完成”；任何未完成项如实标注。只有正式收敛训练和最终测试完成，才可以报告本轮实验完成。

现在从读取上述文件和核验工作区开始，先实现并debug，满足条件后继续正式训练，不要只给计划。
