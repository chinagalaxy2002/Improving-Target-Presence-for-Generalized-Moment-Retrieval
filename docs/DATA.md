# Moment-DETR Clip Supervision 数据契约

```text
$SOCCERGMR_LABEL_ROOT/{Standard/train.jsonl,Standard/val.jsonl,Standard/test.jsonl,Full/test.jsonl}
$SOCCERGMR_STANDARD_FEATURES/{clip,slowfast,clip_text}/*.npz
$SOCCERGMR_FULL_FEATURES/{clip,slowfast,clip_text}/*.npz
```

每条 JSONL 必须含唯一 qid、vid、真实 duration、relevant_windows（二元时间区间列表，空查询为 []），可含 query 文本。四个 split 的视频身份必须满足 Train/Val/Test 隔离；Standard/Full Test 可重叠。Full 的文本特征使用自己的命名空间。

视频 NPZ：文件名为去掉扩展名的 vid，字段 features，clip 宽度 512、slowfast 宽度 2304。文本 NPZ：qid{qid}.npz，字段 last_hidden_state，宽度 512。Moment-DETR 按 clip→slowfast 顺序拼接，添加时间端点特征；视频最大 128 token（超限报错，不静默截断），文本最多 32 token。

片段标签沿用 loader：floor(start/duration*T)、ceil(end/duration*T) 并约束合法范围。多窗口取并集，空查询整行零；padding 在辅助 BCE 中排除。

归档 Full-Decodable 使用明确的 2970/2986 查询清单；请将该清单放在 Full/test.jsonl，并将配置 full_label 设为 Full-Decodable Test 2970/2986。入口不自动剔除缺失特征；未提供资产、非法数组或覆盖不完整会报错。数据、特征不随代码发布。
