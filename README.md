# SoccerGMR: GMR baselines and GEQ Full

本分支的当前主线是在 **SoccerGMR Standard 上训练**多个 DETR 骨干：先训练 GMR baseline，再从其 Val 最优模型初始化 GEQ Full；两阶段选出的固定模型分别评测 **Standard Test 与 Full Test**。Full 不参与训练、选模或调参。

原始 README 及全部旧实验已保存为[历史归档](archive/README.md)，其中 [README 副本](archive/legacy_20261102/README.md)保留原字节与[哈希记录](archive/legacy_20261102/BACKUP_MANIFEST.json)。Charades 仅通过[文件夹索引](archive/charades/README.md)访问。

## 当前协议

| 阶段 | 初始化 | 训练集 | best 的选择依据 | 早停 |
|---|---|---|---|---|
| GMR | 对应骨干随机初始化，保留其原有 GMR head | Standard Train | **Standard Val 的正查询 mAP 最高** | 默认连续 20 个 epoch 不提升，可在配置中指定 |
| GMR + GEQ Full | 上一阶段正式完成的 GMR `best.pt` | Standard Train | **Standard Val 的 G-mIoU@1 在 τ=0.4、0.5、0.6 下的算术均值最高** | **连续 10 个 epoch 不提升** |

每个 epoch 完整验证一次；相等也计为一次未提升，有严格提升时清零。GEQ Full 联合更新原骨干和 GEQ head，保留原 GMR 损失并增加 GEQ 存在性 BCE。定位候选排序沿用骨干规则，存在性阈值仅用于对应的 GMR 指标。两阶段均设置 200 epoch 的默认上限，到上限会记录 `max_epochs`，不能解释成已触发早停。

新实现的训练与验证入口只加载 Train / Val。Test 命令只接受已完成阶段、经 Val 选择且哈希匹配的 `best.pt`，同一个 checkpoint 依次评测两种 Test。恢复使用 `last.pt` 中完整 epoch 的模型、优化器、随机数、数据打乱状态和早停计数；中断的未完成 epoch 会重做。目录锁阻止同一输出目录被重复派发。

## 骨干和验证状态

| 骨干 | GMR → GEQ Full | Standard / Full Test 入口 | 当前数值状态 |
|---|---|---|---|
| Moment-DETR | 已接入 | 已接入 | 本轮做功能复现验证，正式新协议结果待训练 |
| EaTR | 已接入 | 已接入 | 同上 |
| QD-DETR | 已接入 | 已接入 | 同上 |
| CG-DETR | 已接入 | 已接入 | 同上 |
| Sim-DETR | 已接入 | 已接入 | 同上 |

[代码审查与验证报告](docs/REVIEW.md)记录真实数据验证范围、断点恢复证据和限制。按本次授权，**暂不重训全部正式实验**。旧结果按原协议归档，没有将旧早停结果改名为当前 10 轮协议的成绩；功能验证的少量样本分数也不作为性能结果。

## 安装与数据

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

验证环境为 Python 3.10、PyTorch 2.5.1 + CUDA 12.1、torchvision 0.20.1。GPU 训练需安装与本机 CUDA 驱动匹配的上述 PyTorch 版本。所有训练代码、骨干依赖和官方评测实现都在本仓库；数据、预提取特征不随 Git 分发，也不会从其他本地项目导入 Python 代码。

准备路径并设置环境变量：

```text
$SOCCERGMR_LABEL_ROOT/
  Standard/train.jsonl
  Standard/val.jsonl
  Standard/test.jsonl
  Full/test.jsonl
$SOCCERGMR_STANDARD_FEATURES/{clip,slowfast,clip_text}/*.npz
$SOCCERGMR_FULL_FEATURES/{clip,slowfast,clip_text}/*.npz
```

```bash
export SOCCERGMR_LABEL_ROOT=/path/to/labels
export SOCCERGMR_STANDARD_FEATURES=/path/to/Standard_features
export SOCCERGMR_FULL_FEATURES=/path/to/Full_features
python -m soccergmr backbones
python -m soccergmr audit --config configs/experiments/moment_detr.json --output runs/data_audit.json
```

[数据契约](docs/DATA.md)解释 NPZ 字段、特征顺序、空查询、Full 文本特征独立命名空间及覆盖率。Full 原标注为 2,986 查询；已有可解码特征的子集为 2,970 查询。如果使用子集，必须在配置的 `full_label` 中明确填写 **Full-Decodable 2970/2986**。当前入口不自动剔除缺失样本，缺失或异常特征会报错。

## 训练、微调和测试

以 Moment-DETR 为例，其他骨干替换配置文件名即可。每条命令在独立进程中运行。

```bash
python -m soccergmr train --config configs/experiments/moment_detr.json --stage gmr --output runs/moment_detr/gmr
python -m soccergmr train --config configs/experiments/moment_detr.json --stage geq_full --baseline runs/moment_detr/gmr/best.pt --output runs/moment_detr/geq_full
python -m soccergmr test --config configs/experiments/moment_detr.json --checkpoint runs/moment_detr/gmr/best.pt --output runs/moment_detr/gmr_test
python -m soccergmr test --config configs/experiments/moment_detr.json --checkpoint runs/moment_detr/geq_full/best.pt --output runs/moment_detr/geq_full_test
```

恢复时重用相同命令并添加 `--resume`，GEQ 阶段仍提供相同的 `--baseline`。训练配置、代码或 Train / Val 资产变化会拒绝续训；需要新建实验目录。不同 GPU 的可见设备设置也必须保持一致。Test 缓存绑定 checkpoint、代码、标注、特征及结果文件哈希，覆盖率不完整或文件变化时不会直接复用。

每次 Test 在 `test_standard/` 与 `test_full/` 中保存预测、完整指标和 provenance。mAP / G-mIoU 均沿用官方实现的 **0–100** 单位。报告只比较同一骨干的 GEQ Full 相对 GMR 的增减，不使用 Test 决定任何训练参数。

## 本地验收与扩展

```bash
pytest -q
PYTHONPATH=src python scripts/verify_real_data.py --config configs/experiments/moment_detr.json --output runs/verification/moment_detr
```

真实数据验收只抽取明示的小样本，执行中断恢复、连续训练对照、GEQ 联合更新和两套 Test。它验证流程可运行，不验证收敛性能。

新增骨干从[扩展契约](docs/EXTENDING.md)开始。主入口在 [`src/soccergmr`](src/soccergmr/)，架构配置在 [`configs/backbones`](configs/backbones/)，实验配置在 [`configs/experiments`](configs/experiments/)；原生代码快照有[来源与哈希清单](src/soccergmr/backbones/SOURCE_MANIFEST.json)。
