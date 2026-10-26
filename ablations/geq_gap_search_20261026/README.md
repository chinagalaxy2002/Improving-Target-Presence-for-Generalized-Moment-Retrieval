# GAP 参数搜索

用户授权6组新训练，两张GPU，每卡最多3组。margin={0.4,0.6} × weight={0.5,1,2}，原GAP(0.5,1)复用对照。各组原始GMR checkpoint独立初始化，GEQ初始张量与E1一致，seed2026，batch8，固定4轮，原生损失+GEQ BCE+weighted GAP。GMR lr1e-5、GEQ lr3e-5、AdamW wd1e-4、clip0.1，constant scheduler，无PV等其他组件。

search_train.py复用原生训练代码，单组preflight先验证公式、单类/混合/多GT、GEQ/Transformer梯度、原生定位与presence梯度、严格重载、官方导出；正式训练重新初始化，逐epoch与原GAP的实际loader qid哈希比对。

仅保存第4轮Val官方结果并独立复算。选择主指标为Val AUROC，依次AUPRC、S、mAP打破并列；六个新配置和原GAP共同参与。选择文件保存后，只有选中配置自动运行Test；非选中模型不测Test。全部Val结果保留，不能把更高均分间隔等价为更高排序或校准能力。原参考Test已经观察，不声称全研究盲测。

使用gmr环境Python执行 `launch.py --launch`，后台驱动器依次衔接preflight、训练、选择、所选Test及报告绘图。STATUS.json/QUEUE.json为真实状态，失败保存日志且不选择。结果见reports/RESULTS.md、val_search.csv、test_selected.csv、SELECTION.json和AUDIT.json；图见figures/。本目录不改写旧实验代码或证据。

## 后续：全部配置 Test 评估已完成

按用户追加要求，六组 epoch004 已全部运行 Test 官方评估与独立复算。最新总表见 [reports/all_test/RESULTS.md](reports/all_test/RESULTS.md)，CSV见 [reports/all_test/test_summary.csv](reports/all_test/test_summary.csv)，完整Val/Test及固定阈值敏感性见 reports/all_test/all_metrics.csv；复核见该目录AUDIT.json。原按Val选定的0.6/2保持不变。旧reports/RESULTS.md和AUDIT.json是首次仅评估所选模型时的历史快照，不代表最新覆盖范围。补测入口为evaluate_all_test.py。
