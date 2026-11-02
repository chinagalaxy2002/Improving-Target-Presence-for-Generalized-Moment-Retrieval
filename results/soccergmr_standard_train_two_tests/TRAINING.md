# Standard训练来源与模型身份

## 固定身份与方法链

本目录24项模型均使用已经存在的best；训练只来自Standard Train/Val，测试没有参与旧模型的选模。完整机器清单：[data/CHECKPOINTS.json](data/CHECKPOINTS.json)。每条有原本机checkpoint路径、SHA256、原始epoch、父checkpoint、Standard mAP、训练配置和证据目录。[evidence/training/](evidence/training/)包含原配置、日志/历史、Val-best指标和参数键布局，不包含权重数值。

每个骨干的方法链相同：

```text
Standard Train/Val → 原生GMR best
                       ├─ GEQ Frozen → GMR + GEQ Frozen
                       └─ GEQ Full   → GMR + GEQ Full
Standard Train/Val → 原生GMR + Multi训练 → Multi best
                       ├─ GEQ Frozen → GMR + Multi + GEQ Frozen
                       └─ GEQ Full   → GMR + Multi + GEQ Full
```

原生GMR/Multi按其既有**Standard Val正例定位mAP**选best。GEQ按**Standard Val的G-mIoU@1在τ=.4/.5/.6三点均值**选best，最多20轮，每轮验证，连续**超过5轮**不改善才停止；均值相同的处理以`validation_selection.py`为准。GEQ best刚好在第1轮，不表示它是历史main E1或固定一轮实验。

## 实际原生训练差异

| 骨干/路线 | seed | microbatch × 累积 | 有效batch | 学习率 | 来源说明 |
|---|---:|---:|---:|---:|---|
| Moment-DETR GMR | 2023 | 2×4 | 8 | 5e-5 | 既有run_train路线 |
| Moment-DETR Multi | 2023 | 2×4 | 8 | 5e-5 | 既有run_train路线 |
| EaTR GMR | 2026 | 32×1 | 32 | 3e-5 | 复用已完成原生baseline，再包装成当前模型 |
| EaTR Multi | 2023 | 2×4 | 8 | 5e-5 | multiroute路线 |
| QD-DETR GMR | 2023 | 2×4 | 8 | 5e-5 | multiroute路线 |
| QD-DETR Multi | 2023 | 32×1 | 32 | 5e-5 | 后续队列恢复原batch32的路线 |
| CG-DETR GMR | 2023 | 2×4 | 8 | 5e-5 | multiroute路线 |
| CG-DETR Multi | 2023 | 32×1 | 32 | 5e-5 | 后续队列恢复原batch32的路线 |

这些是完整方法的既有实测比较，**不能称为统一seed/预算的消融**。源码中的当前默认值不能替代每个历史检查点的配置，尤其EaTR baseline有独立`reused_baseline.json/source_effective_config.json`。

Moment配置最大400轮，但本次保留的train.log截至GMR第25轮、Multi第24轮，最后Val日志分别到25/20轮；只能报告日志快照与已选定best，不能据上限声称训练到400轮或推断停止原因。其best原始epoch分别4/14，即按0起始记录的第5/15轮。

其余新multiroute路线通常配置最大400轮、每5轮验证、当前轮−best轮≥20时停；AdamW、wd=1e-4、grad clip=.1，scheduler及细节以各配置/源码为准。EaTR既有GMR原路线最大200轮，不能套用后来multiroute配置。

## 代码位置与构造

原环境快照在[source/](source/)，哈希和原路径在PROVENANCE.json：

- Moment：`source/gmr_multiroute_kit/run_train.py`及`multiroute.py`；底层模型/配置在`source/moment_project/`。
- EaTR/QD/CG统一控制：`source/backbonefolder/multiroute_gmr/run.py`的`Experiment`，包装器在同目录`multiroute.py`。
- EaTR：`source/backbonefolder/eatr/deploy.py`及`models/`。
- QD：`source/backbonefolder/qd_detr/train_gmr.py`及`gmr_deployment/`。
- CG：`source/backbonefolder/cg_detr/train_gmr.py`及`gmr_source/`。
- GEQ：`source/gmr_geq_protocol_kit/native_adapter.py`、`geq_model.py`、`run_study.py`、`validation_selection.py`、`study_paths.py`。

GMR/Multi原生best由相应包装器加载。GEQ必须构造`RouteGEQ`后加载完整best，`Full`保留已经更新的原生权重；只把父GMR/Multi加载进新GEQ不等于恢复已有GEQ Full。所有现有checkpoint加载采用`strict=True`，24项Full预检/完成证据在[evidence/full/](evidence/full/)。

Multi在训练期使用Separate-FFN和Instruction等辅助路线，原主query数保留，推理不执行辅助路线。具体loss以源码为准；例如CG的Multi辅助分类交叉熵对负例也可非零，不能把整个Multi loss都描述成只处理正例。

GEQ Frozen保持原生base eval且参数冻结，只训练GEQ BCE；GEQ Full为原生loss加GEQ BCE，更新base与GEQ。Moment的GEQ batch8，其余三个骨干batch32；seed2023、lr_geq=3e-5、Full lr_base=1e-5、wd=1e-4、clip=.1、AMP关闭。每个方法自己的CONFIG.json是最终依据。

## 原本机路径

```text
项目：/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval
原生runtime：/home/guoxiangyu/Improving-Target-Presence-for-Generalized-Moment-Retrieval/backbonefolder/multiroute_gmr
Moment父best：项目/mr_experiments/{baseline,multi}/best.ckpt
其余父best：原生runtime/runs/{eatr,qd_detr,cg_detr}/{baseline,multi}/best.ckpt
GEQ best：项目/geq_protocol_experiments/20261031/<backbone>/<baseline或multi>/<frozen或full>/best.ckpt
Python：/home/guoxiangyu/miniconda3/envs/gmr/bin/python
```

不要另找同名但不同起点的旧best。清单没有列出的seed2026独立CG部署、历史main E1、固定四轮GEQ等不属于当前主表。EaTR GMR的seed2026来源是清单明确记录的例外。

## 未来训练的执行原则

本次发布没有训练。未来从头复现/优化时：先固定目标检查点和数据、复制源码到独立工作树、把输入和输出路径显式适配到新目录，再核验真实正/负/多GT小批、梯度/冻结/严格保存重载，最后执行训练。原始入口可能恢复last、写入固定runs目录或训练完自动测Standard Test；不得在旧目录直接执行。

`run_study.py`训练之后有自动Test块；探索阶段应先改为Train/Val-only流程。校准参数/新loss权重/early-stop只在Train/Val选择，冻结配置后再报告Test。权重/特征没有随报告上传，从头重训也不承诺逐位相同的checkpoint SHA256。

CG的完整隔离准备工具及专门优化交接见[handoff中的归档](handoff/cg_detr_gmiou_handoff_20261102.zip)；其他骨干先按相应源码和配置建立隔离入口，不把CG准备工具当作其他骨干的通用启动器。
