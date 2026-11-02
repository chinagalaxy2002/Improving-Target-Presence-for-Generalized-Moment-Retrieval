# Backbone 入口

每个骨干一个目录：

| 目录 | 内容 |
|---|---|
| [moment_detr/](moment_detr/) | 原生源码、六路线训练/测试入口、完整冻结结果包 |
| [eatr/](eatr/) | EaTR 源码与调用共享实现的训练/测试入口 |
| [qd_detr/](qd_detr/) | QD-DETR 源码与调用共享实现的训练/测试入口 |
| [cg_detr/](cg_detr/) | CG-DETR 源码与调用共享实现的训练/测试入口 |

所有目录顶层都有 `smoke.sh`、`train.sh`、`test.sh`，第一个参数为方法名称。
公共实现只在 [`../shared/`](../shared/) 保存一份。
