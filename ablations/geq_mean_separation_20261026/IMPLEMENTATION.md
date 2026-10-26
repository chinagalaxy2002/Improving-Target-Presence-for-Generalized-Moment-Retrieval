# 主会话实施与验收计划

## 复用关系

复用 ../end_to_end_geq_pv_20261026 的 adapters.py、model.JointGMR(pv=False)、GlobalEventQuery、evaluation.py 与 independent_metrics.py。先完整阅读实际调用关系，再在本目录实现新入口；不修改运行中的 E3/E4 源码，不覆盖原 E0/E1/E2/E3/E4 结果。对照 SOURCE_MANIFEST.json 检查来源是否变化，变更时记录实际版本。

训练损失必须调用 E1 的 native_loss 与 GEQ BCE 原路径，再加入本目录定义的标量；不使用 ScratchAdapter，因为它移除了 native existence BCE。不要复制整个 backbone 或改写 native criterion。

新增损失的实现示意：

```python
s = logits.sigmoid().reshape(-1)
y = labels.reshape(-1).bool()
if y.any() and (~y).any():
    mu_pos, mu_neg = s[y].mean(), s[~y].mean()
    gap = (0.5 - (mu_pos - mu_neg)).clamp_min(0).square()
    endpoint = 0.5 * ((0.75-mu_pos).clamp_min(0).square()
                      + (mu_neg-0.25).clamp_min(0).square())
else:
    gap = endpoint = logits.sum() * 0.0
```

每次训练只选择其中一个新增损失，不把 GAP 和 ENDPOINT 相加。mean、sigmoid 和 clamp 均保留梯度，不用 .item()/.detach() 参与 loss；日志可 detach。

## Preflight 必须通过

1. 原始 checkpoint 字节哈希与 E1 初始化记录一致；strict load；GMR/GEQ 初始共享张量哈希与 E1 一致。
2. 两组完整 trainable parameter 清单、参数组学习率正确；原生 train/eval 行为正确。
3. 手工构造分数验证两种损失的公式、达标为零和梯度方向；未达标时正类 logit 新增梯度为负、负类为正。达到目标可零梯度。
4. 正、负、混合 batch 新增损失有限；单类 batch 的零项可反传，native/GEQ 保持工作。
5. 混合且约束未达标时，新损失单独回到 GEQ 和实际融合 Transformer；原生定位项单独回到 coordinate Head。
6. 实际正、负、多 GT 和原生 batch8 做 optimizer step，finite check、clip=0.1、参数确有变化。
7. 保留 checkpoint 严格重建加载，概率及官方预测一致；官方评估与独立接口可运行。
8. 检查丢弃 smoke 更新，正式训练重新从相同原始 checkpoint/seed 开始。

## 两个工作单元与 GPU 队列

主会话按用户请求使用两个 agent：A 负责 GAP 的配置检查、preflight、训练与自动评估；B 负责 ENDPOINT 的相同步骤。各 agent 仅写自己的 run/log/preflight；共享训练与评估工具先由主会话统一实现并冻结版本，避免同时编辑公共文件。

A 首选 GPU0，B 首选 GPU1。启动前读取实际 GPU 进程与队列；每卡最多同时两个任务，满额则等待，不终止或改动既有实验。preferred GPU 不是已分配资源，QUEUE_PLAN.json 不是运行状态证明。本侧边会话禁止 agent 调度，因此当前仅形成该计划。

## 预期目录（后续实现时创建）

- losses.py、run.py、evaluate.py：本实验新增损失与统一入口。
- preflights/{GAP,ENDPOINT}/：梯度、单类/混合、多GT、重载检查与状态。
- runs/{GAP,ENDPOINT}/：CONFIG、initial hashes、epoch_001..004、history、STATUS、COMPLETE。
- logs/：独立训练和评估日志；失败保留 traceback，不写 COMPLETE。
- reports/：官方/独立 Val/Test、完整 CSV、结论及 provenance。
- figures/：E0/E1/GAP/ENDPOINT 的指标、ROC/PR、正负分数分布和损失曲线，PDF/SVG/PNG。

训练结束严格加载 epoch_004，自动 Val/Test、独立复算，再产生 summary。正常训练启动后可以按用户偏好结束交互；后台任务保留自动评估。只有实际完成且验证通过的步骤可标记完成。
