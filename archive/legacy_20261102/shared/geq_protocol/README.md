# Multi / Shared：Val 选模、全量与冻结 GEQ

用户指定四个骨干的 `multi` 和 `shared` 各运行 full / frozen，共16组。
另补当前同一族 GMR baseline 的 full / frozen 共8组，保证两套四曲线使用相同选模协议。
旧固定4轮结果和 main E1 历史初始化的8组对照独立保留，不覆盖、不串接。

统一 seed=2023，从原路线 best 独立初始化 GEQ；两模式的初始模型和 GEQ 张量相同。
每轮评估完整 Val 465条，以阈值0.4/0.5/0.6的 G-mIoU@1 均值最大化选 best，
记录三点是否都高于原路线。连续超过5轮（6轮）不提升早停，上限20轮。
完整 Test 1036条只在选模结束后评估，不参与轮次、阈值或方案选择。

- full：保持原 GMR 分类、边界、存在性和 O2M 损失，另加 GEQ BCE；原参数LR=1e-5、GEQ LR=3e-5。
- frozen：原模型和 buffer 冻结，保持eval，只训练 GEQ BCE，LR=3e-5；辅助路线不执行。
- AdamW、wd=1e-4、梯度裁剪0.1、无AMP；Moment batch8，其余batch32；num_workers=2。
- full GEQ 的编码器特征参与反向传播；frozen 特征detach。推理两模式均仅使用主路线。
- frozen Test仅替换原导出候选的存在性概率，时间边界、候选分数、排序及mAP保持相同。

服务器上：

```bash
bash gmr_geq_protocol_kit/launch.sh
tail -f geq_protocol_experiments/20261031/queue.log
python gmr_geq_protocol_kit/report.py
python gmr_geq_protocol_kit/publish.py
```

GitHub整理后的目录为 `shared/geq_protocol/`。复现前设置 `GMR_PROJECT_ROOT`、
`GMR_BACKBONE_REPO`、`GMR_RUNTIME_ROOT` 指向服务器原项目、原仓库和运行目录；
原项目必须包含官方GMR依赖、`gmr_multiroute_kit`、数据、特征和已有路线best。
权重和特征不随GitHub提交。`GMR_PROTOCOL_OUTPUT`、`GMR_PROTOCOL_FIGURES`可以另设输出目录。

队列在tmux中持续运行，计入所有已有CUDA作业，每卡最多两个；每个作业先做真实数据
空集、混合样本和原生batch的前后向检查，检查后恢复初始化再训练。失败从last恢复，
最多3次；永久失败停止新增派发并在QUEUE.json中标记。文件锁防止重复启动。

每完成一组，自动更新状态、Val曲线及已完成的Test对照并上传指定分支。
每骨干两模式各需baseline/shared/multi三组全部完成，才生成相应完整四曲线。
未完成组留空，不拿旧四轮结果填新协议的曲线。PNG、PDF、SVG、101点CSV和快照
按骨干目录保存；G-mIoU图轴与阈值均为0–1，mAP为百分数。

`report.py --baseline-study`专门绘制已完成的main E1同源8组对照；其历史初始化与
本次多路线初始化不同，独立放在 `results/geq_epoch_study/`。

自动发布使用独立工作树 `GMR_PUBLISH_WORKTREE` 和 `GMR_GIT_TOKEN_FILE` 指定的本地
凭据文件。凭据只通过Git askpass读取，不写入仓库、URL或日志。发布失败记录为失败，
不会中断训练；后续完成事件会重试。换机时须自行配置凭据和Git工作树。
