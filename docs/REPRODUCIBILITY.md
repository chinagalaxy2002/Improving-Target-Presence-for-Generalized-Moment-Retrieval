# 当前仓库的复现方法

## 独立数据与指标审计

在当前 GitHub 仓库根目录执行：

```bash
python -m pip install -r requirements-cpu.txt
python tools/audit_documentation.py
```

该命令只读取本仓库的标签、冻结预测、配置和结果表，默认写入 `docs/audit/`。不需要视频、特征、检查点、GPU、其他源工作区或远程下载。依赖安装完成后，审计过程本身不访问网络。

| 输出 | 含义 |
| --- | --- |
| `docs/audit/dataset.json` | 三个划分的原始标签统计、来源构成、划分交集与 SHA256 |
| `docs/audit/metrics.csv` | 14 组本地预测 × 三个阈值，共 42 行 |
| `docs/audit/gains.csv` | 五个主干在三个阈值下相对自身 GMR 的差值 |
| `docs/audit/sources.json` | 全部审计输入均为仓库内相对路径，并记录文件 SHA256 |
| `docs/audit/verification.json` | 计分和发布表一致性检查结果 |

可用 `--output /tmp/gmr_documentation_audit` 指定另一输出目录。审计不会覆盖已有曲线缓存、结果表或原始预测，也不会加载二进制权重。

审计先检查预测文件与各自缓存记录的哈希，再检查 1,036 条 Test 查询覆盖。随后调用本仓库 `shared/official_eval`，在 .4/.5/.6 上重算指标，与三张完整 CSV 的 14 行 `Our experiment` 逐项比较。六行外部论文结果明确排除。共核验 42×8=336 个已发表本地指标值。

## 已有表格和图形入口

仓库原有工具仍可使用：

```bash
python tools/reproduce_results.py
python tools/reproduce_results.py --figures
python tools/reproduce_results.py --recompute-metrics
```

第一个命令重建三张结果表，`--figures` 额外绘制五主干曲线。`--recompute-metrics` 会重算完整指标并更新原有缓存和图表文件；如果只需要检查数值一致性，使用前述 `audit_documentation.py` 即可。

## 训练与模型推理

模型和训练脚本是历史源码快照，路径映射见 [CODE_MAP.md](CODE_MAP.md)。这些脚本可能含有原运行环境的绝对路径，不能直接承诺在发布目录中一键重训。

GitHub 包不含原始预提取特征或二进制权重；原始 `ARCHIVE_INDEX.json` 中的检查点字段主要提供身份与来源记录，不代表该二进制文件已经上传。重新训练或推理需要准备匹配的资产、核对各主干接口并配置路径。文档审计没有执行新的训练、重新推理或性能测量。

## 本次核验范围

本次仅使用此仓库快照中已经存在的研究材料。14 组预测的哈希和查询覆盖均通过，42 行重算结果与原发布表一致。文档没有把可用的辅助代码、其他目录内容或外部论文行认定为本研究完成的额外实验。
