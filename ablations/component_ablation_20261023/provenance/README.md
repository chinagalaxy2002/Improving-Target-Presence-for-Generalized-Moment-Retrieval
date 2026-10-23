# 发布来源与适配

`SOURCE_COPY_MANIFEST.json` 保存从本地完整实验复制时的源文件哈希；预测压缩后另保存压缩与解压字节的哈希。`EXTERNAL_CHECKPOINTS.json` 描述25组正式新增配置及5组额外诊断的权重，基线身份见各正式配置1的 CONFIG.json 与仓库原外部权重记录。模型二进制不包含在Git发布中。

本发布将本地 `runs/<backbone>/row_01..06` 搬到 `results/formal/`，将执行编号 `row_07` 搬到 `extra_diagnostics/results/`；预测仅作确定性gzip压缩，不改变JSONL字节、数值或顺序。省略缓存、旧失败检查过程日志、其他数据集标签、视频和特征。训练日志与科学源码保留。

`LOCAL_EXPERIMENT_README.md`、`LOCAL_EXPERIMENT_REPORT.md` 以及 analysis_snapshot 是原本地目录的历史快照，其中相对/绝对路径按原本地结构解释，不能作为公开CPU复现入口。原始报告中未四舍五入的N-acc差值可能与发布表按显示两位小数相减的增益相差0.01 pp。公开表格遵循仓库原表的后者口径。

当前公开子集的复现入口是仓库根目录 `tools/reproduce_ablation_tables.py`，完整性与数值复核见本目录上一级的 PUBLICATION_MANIFEST.json、PUBLICATION_VERIFICATION.json、TABLE_VERIFICATION.json。完整未发表论文与本地编译器不属于这次实验发布范围。
