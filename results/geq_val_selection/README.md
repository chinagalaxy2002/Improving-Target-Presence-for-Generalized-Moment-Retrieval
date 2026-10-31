# Baseline / Multi / Shared：统一 Val 选模协议

Multi与Shared各运行全量和冻结两种方案，共16组。另补同源Baseline的两种微调方案8组，合计24组，用于统一四曲线对照。

各方案从对应已完成的原路线best独立初始化，GEQ seed2023。Val G-mIoU@1在0.4/0.5/0.6的均值选best，连续超过5轮（6轮）无提升早停，上限20轮；Test仅评估选定best。full保留原GMR+O2M损失并加GEQ BCE，原模型lr1e-5、GEQ lr3e-5；frozen仅GEQ BCE，原模型保持eval。

每任务num_workers=2，Moment batch8，其余batch32，无梯度累积，AMP关闭。每GPU至多两个CUDA任务，计入已有训练。旧固定4轮结果单独保留，不冒充新选模结果。

全量和冻结成对初始化相同。每骨干四条曲线分别为原GMR、GMR+GEQ、Multi+GEQ、Shared+GEQ；两种训练方式各一套。原GMR与路线的既有预训练预算仍存在差异，完整方法比较不等于单因素因果消融。

更新时间：2026-10-31T22:51:49+08:00

| Backbone | 路线 | 模式 | 状态 | 完成Epoch | Best Epoch | Best Val均值 | Test mAP (%) | Test G1均值 |
|---|---|---|---|---:|---:|---:|---:|---:|
| Moment-DETR | shared | full | RUNNING | 8 | 6 | 0.4826 | — | — |
| Moment-DETR | shared | frozen | COMPLETED | 9 | 3 | 0.4846 | 7.28 | 0.5128 |
| Moment-DETR | multi | full | QUEUED | — | — | — | — | — |
| Moment-DETR | multi | frozen | QUEUED | — | — | — | — | — |
| Moment-DETR | baseline | full | QUEUED | — | — | — | — | — |
| Moment-DETR | baseline | frozen | QUEUED | — | — | — | — | — |

Moment-DETR full：四曲线等待同源Baseline、Multi、Shared全部完成Test。

Moment-DETR frozen：四曲线等待同源Baseline、Multi、Shared全部完成Test。
| EaTR | shared | full | RUNNING | 4 | 3 | 0.4044 | — | — |
| EaTR | shared | frozen | QUEUED | — | — | — | — | — |
| EaTR | multi | full | QUEUED | — | — | — | — | — |
| EaTR | multi | frozen | QUEUED | — | — | — | — | — |
| EaTR | baseline | full | QUEUED | — | — | — | — | — |
| EaTR | baseline | frozen | QUEUED | — | — | — | — | — |

EaTR full：四曲线等待同源Baseline、Multi、Shared全部完成Test。

EaTR frozen：四曲线等待同源Baseline、Multi、Shared全部完成Test。
| QD-DETR | shared | full | QUEUED | — | — | — | — | — |
| QD-DETR | shared | frozen | QUEUED | — | — | — | — | — |
| QD-DETR | multi | full | QUEUED | — | — | — | — | — |
| QD-DETR | multi | frozen | QUEUED | — | — | — | — | — |
| QD-DETR | baseline | full | QUEUED | — | — | — | — | — |
| QD-DETR | baseline | frozen | QUEUED | — | — | — | — | — |

QD-DETR full：四曲线等待同源Baseline、Multi、Shared全部完成Test。

QD-DETR frozen：四曲线等待同源Baseline、Multi、Shared全部完成Test。
| CG-DETR | shared | full | QUEUED | — | — | — | — | — |
| CG-DETR | shared | frozen | QUEUED | — | — | — | — | — |
| CG-DETR | multi | full | QUEUED | — | — | — | — | — |
| CG-DETR | multi | frozen | QUEUED | — | — | — | — | — |
| CG-DETR | baseline | full | QUEUED | — | — | — | — | — |
| CG-DETR | baseline | frozen | QUEUED | — | — | — | — | — |

CG-DETR full：四曲线等待同源Baseline、Multi、Shared全部完成Test。

CG-DETR frozen：四曲线等待同源Baseline、Multi、Shared全部完成Test。
