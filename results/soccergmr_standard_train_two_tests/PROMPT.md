# 清空上下文后可直接使用的提示词

请完整读取并执行本目录交接：`results/soccergmr_standard_train_two_tests/`。若在原机器，本目录绝对路径为`/home/guoxiangyu/soccergmr-two-tests-publish-worktree-20261102/results/soccergmr_standard_train_two_tests/`；若新克隆仓库，以仓库中的相对路径为准。

依次读取README.md、HANDOFF.md、TRAINING.md、DATA.md、REPRODUCE.md，以及data/CHECKPOINTS.json、data/RUNS.json、AUDIT.json；先运行`python verify.py --all-files`核对发布身份。只由当前会话实施，不启动其他代理。

范围是SoccerGMR的Moment-DETR、EaTR、QD-DETR、CG-DETR，每骨干六方法GMR、GMR+GEQ Frozen、GMR+GEQ Full、Multi、Multi+GEQ Frozen、Multi+GEQ Full。24个Standard训练best在Standard Test与Full-Decodable Test各评24项，共48正式记录，不是48次独立训练。不要混入Charades、Shared、组件消融、历史main E1或固定四轮GEQ。

先依据本轮用户任务判断：查数值/绘图直接使用附带预测/指标；复查旧模型时按SHA合法复用；新训练/优化只有在用户给出相应任务时实施，在独立工作目录保存新版本。不能因读取提示词自动启动旧训练、旧队列或全24项GPU推理。

模型绝对路径、SHA、父子关系、epoch和配置以清单为准，不另找同名best。GMR/Multi按旧Standard Val mAP选best，GEQ按Val三点G-mIoU@1均值选best；GEQ Full须严格加载全部已微调权重，不得退回父模型权重而称为同一方法。EaTR GMR是seed2026复用例外；batch/预算亦有差异。

Full结果只能称Full-Decodable Test (2970/2986)，保留原16条排除与时间输入限制。Full qid/文本/特征独立，不能取Standard预测。Frozen只替换对应父模型在同一测试域的存在性分数，候选和全部定位指标保持完全一致。所有输入覆盖、模型严格加载及结果哈希都应核验。

定位指标用正例未门控候选，分类与G-mIoU用全正负域。报告G-mIoU@1/@3在τ=.4/.5/.6三点，数值0–1，其他百分数，增减统一相对同骨干同Test的GMR。图例带对应mAP，每个子图一条oracle存在性上限横线；不把oracle当实际可部署性能。

两个Test共享1036个规范化视频stem+查询对，不能当独立样本。Test曲线已看过，优化配置只在Train/Val确定；后续该Test上的提升按探索性说明。若引入Full训练/校准或修正时间处理，另立协议并统一处理对照。

任何新GPU任务都先检查其他作业，每卡最多2项、总计4项；不终止其他任务。保留旧权重、代码、GT、特征、预测与报告。完成后提供实际完成状态、明确数据覆盖、结果/代码/输入哈希和可复查入口，不能把启动或小批成功当成完整完成。
