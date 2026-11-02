# 离线复现与原环境评测

## 在新机器离线重建已有结果

克隆仓库后进入本目录：

```bash
cd results/soccergmr_standard_train_two_tests
python -m pip install -r requirements.txt
python verify.py --all-files
python rebuild.py
```

`verify.py`只读取本目录文件，不访问原本机路径，不加载模型、不初始化CUDA。`--all-files`核对发布快照的所有文件；默认只核对固定输入/源码/文档并验证48条数值身份。

`rebuild.py`从48份gzip逐查询预测与测试GT调用附带官方评估器，重新计算三点正式指标、0–1固定阈值曲线、oracle上限、全部相对GMR的增减，并导出PNG/PDF/SVG/LaTeX/CSV/JSON。无网络、无GPU、无模型权重/特征依赖。所需Python包仅numpy、matplotlib、scikit-learn。

仅重算数值而不渲染图片可运行`python rebuild.py --metrics-only`。重建会改写本目录的派生结果；PDF/SVG编码及官方统计中的耗时可能因库版本/运行环境改变，因此不保证重建后所有二进制字节与发布快照相同。应核对数值与固定输入身份；重新生成后可用默认`verify.py`检查固定输入，`--all-files`仍按原发布字节核验。

## 附带的输入及证据

- `data/RUNS.json`为48条记录的唯一入口，声明测试域、checkpoint SHA、gzip预测、未压缩原预测SHA、GT、原发布期望值和证据路径。
- `data/CHECKPOINTS.json`为24个权重身份入口。同一模型的两个Test引用完全相同的checkpoint SHA。
- `data/predictions/<test>/<backbone>/<method>.jsonl.gz`为原逐查询预测的确定性gzip打包，解压字节与原文件SHA256一致。
- `metrics/`为本包官方重算结果；`evidence/standard/`和`evidence/full/`保留原正式指标及完成证据。
- `evidence/standard_summaries/`保留与此前Standard主表对应的来源摘要；本包重新计算的正式值逐项核对，未用别的best替换。
- `PROVENANCE.json`登记原文件/原检查点SHA及每个复制文件来源；`PACKAGE_SHA256.json`是本目录发布内容的完整指纹。

## 原机器重新运行模型推理

原机器Python为`/home/guoxiangyu/miniconda3/envs/gmr/bin/python`。原Full独立eval入口位于：

```text
/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_full_test_20261102/eval_only.py
```

它与同目录`common.py`、`audit_inputs.py`、`run_all.py`协作。合法完成后按checkpoint/GT/特征/代码/预测身份复用。示例：

```bash
cd /home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/soccergmr_full_test_20261102
CUDA_VISIBLE_DEVICES=0 /home/guoxiangyu/miniconda3/envs/gmr/bin/python -B eval_only.py --index 23 --phase formal
```

这是CG的Multi+GEQ Full；原结果已完成时会输出`REUSED_VALID_COMPLETION`，不是重新进行模型推理。每个方法的索引为清单顺序：Moment 0–5、EaTR 6–11、QD 12–17、CG 18–23；每组按README中的六方法顺序。不要为了查看指标而启动24项模型队列。

对新模型测试必须建新输出、新清单和新CG/骨干入口：显式设置checkpoint/sha、父模型、GT、Full特征、代码版本和输出根，再做逐项严格加载/小批/完整推理。不能改旧清单或完成标记。`source/full_eval/`是原环境快照，里面包含历史绝对路径、`PROJECT=HERE.parent`及全24项依赖，不能搬到新目录后不改配置就直接运行。

## 原机器训练复现/后续优化

读TRAINING.md与HANDOFF.md，核对选定方法原始配置和初始化。源码快照保留原路径语义，不是一键通用启动器。新训练要在隔离工作树重设数据/输出，先验证保存重载和loss/梯度，再运行；不得覆盖旧best、启动旧队列或让旧自动刷图脚本改写历史发布结果。

所有GPU操作前计入已有CUDA作业，最多每卡2项、全机4项，不终止其他任务。训练探索仅用Train/Val；原`run_study.py`训练后自动Test的块需要与探索流程分离。Frozen只替换同测试域父预测的存在性分数。完整best和输出身份以本目录清单及原机校验为准。
