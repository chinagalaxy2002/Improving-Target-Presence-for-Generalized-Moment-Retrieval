# 新会话交接：四骨干、Standard训练、两套Test

## 工作目标与已完成状态

本目录发布当前四骨干×六方法的24个Standard Train/Val best，在Standard Test和Full-Decodable Test上的48条正式记录。权重不重训、不微调、不重新选best；本次只整理、校验并离线重算指标和图表，没有新模型推理。

清空上下文后先读README.md、TRAINING.md、DATA.md、REPRODUCE.md，再读`data/CHECKPOINTS.json`、`data/RUNS.json`和AUDIT.json。运行`python verify.py --all-files`验证发布快照；要重建数值/图表运行`python rebuild.py`。不需要回忆旧会话，也不应搜索另一个同名best。

## 当前模型身份

四骨干顺序：moment_detr、eatr、qd_detr、cg_detr。每组六方法顺序：gmr、gmr_geq_frozen、gmr_geq_full、multi、multi_geq_frozen、multi_geq_full。

父模型目录：Moment为`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/mr_experiments/{baseline,multi}/best.ckpt`；其余为`/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr/runs/<backbone>/{baseline,multi}/best.ckpt`。GEQ为`/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/geq_protocol_experiments/20261031/<backbone>/<baseline或multi>/<frozen或full>/best.ckpt`。绝对路径和SHA以data/CHECKPOINTS.json为准。

原生GMR/Multi按Standard Val定位mAP选best；GEQ按Standard Val三点G-mIoU@1均值选best。EaTR GMR baseline复用seed2026原模型，不能套用seed2023的Multi训练配置；其他预算差异见TRAINING。GEQ Full要恢复完整包装及已更新的原生权重；从父模型重建随机GEQ是不同初始化。

## 数据与测试必须保留的边界

Standard Train/Val/Test为4138/465/1036。当前Full测试仅覆盖Full-Decodable 2970条，原始2986的16条历史排除、两段75步截断及3段时长偏差已列入DATA。Full与GEQ Full模式必须区分。

Full使用独立qid、文本和Full_features，不能复用Standard预测。GT读取relevant_windows，不重复从moment扩窗。只在查找特征时去vid扩展名；输出身份不变。所有方法相同GT和特征协议，严格state_dict加载，eval/no_grad，无训练辅助路线泄漏。

Frozen按当前测试域复制对应父模型定位候选边界/排序/分数，仅替换存在性分数，候选和全部定位指标一致。两个测试域共16项检查均通过。G-mIoU保持0–1，其他百分数，增减只相对同骨干同Test的GMR。oracle上限只是完美存在性条件下当前候选能恢复的最大值，不是部署成绩。

Standard Test与Full Test共享1036个规范化视频stem+查询对，不是独立样本；Train/Val与原始Full Test规范化video stem初查无交集。测试结果与曲线已看过，未来在这些Test上的优化应说明探索性，不能再声称是未触及的Test。新配置/校准/训练选择在Train/Val完成。

## 原本机位置

```text
原Standard汇总：/home/guoxiangyu/charades-multi-geq-worktree/results/test_all_thresholds/
原Full正式结果：/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_full_test_20261102/
原Full交接：/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_full_test_handoff_20261102/
原CG优化交接：/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/cg_detr_gmiou_handoff_20261102/
本次发布工作树：/home/guoxiangyu/soccergmr-two-tests-publish-worktree-20261102/
本目录：上述工作树/results/soccergmr_standard_train_two_tests/
Python：/home/guoxiangyu/miniconda3/envs/gmr/bin/python
```

原工作仓库有用户未提交的README.md与docs/DATA.md修改；本次通过独立工作树只新增本目录，不把这些修改加入发布。未来同样先检查git状态，不reset、覆盖或夹带其他工作。

## CG-DETR后续优化上下文

此前默认重点是CG的Multi+GEQ Full：checkpoint SHA `535f94493897cb825c2b4c2b502fa1abf2bdaf223df635770ba0abb23157f1f1`，其父Multi SHA `c229ccf56f81131e2d3fa9dafdf3595ef7753b6975c28dbf70c23466605d2a98`。

Full mAP=15.93，相对GMR 13.58为+2.35百分点；G-mIoU@1三点=.3593/.4455/.4929，低于GMR的.5068/.5143/.5150；其oracle上限@1=.5743、@3=.5123。应优先在Standard Val拆解存在性误接收、误拒绝与定位质量差距，再验证具体优化假设，不可直接用Test挑阈值。

专用交接归档：[cg_detr_gmiou_handoff_20261102.zip](handoff/cg_detr_gmiou_handoff_20261102.zip)。解压后内含PROMPT_OPTIMIZE、TRAINING_HANDOFF、TESTING_HANDOFF、核验脚本及隔离原生训练准备工具；文件中的绝对路径仍对应原机器。

## 新任务的执行方式

- 查数值/重画图：使用本目录data和压缩预测离线完成，不重跑模型。
- 严格复查旧结果：只核验清单与完成标记，合法复用，不宣称是新训练或新推理。
- 测试新权重：在独立目录新建显式checkpoint/GT/特征配置的入口，先严格加载/小批，再全量Test，不改旧结果。
- 优化训练：先记录初始化、选模指标、控制变量和预算，在新工作树编辑；Train/Val选配置。阅读本交接本身不代表授权自动启动全套训练或24项GPU队列。
- GPU操作计入其他CUDA作业，每卡最多2项、总计4项，不终止其他作业。当前会话执行，不启动其他代理。

最终报告需区分“已校验/已复用/已重新推理/已训练”，提供模型与输入身份、覆盖、官方指标、日志和哈希。不得把准备脚本、启动队列或少量样本成功当作全部完成。
