# 额外融合诊断：不计入正式30组

这里单列每个主干的融合开启诊断，共 **5组额外实验**（原执行编号 `row_07`）。正式组件消融仍严格为 **5个主干 × 配置1–6 = 30组**。

- 正式Full/配置6采用已有 `global_refine` / minusFusion 路径，存在性不读取候选证据。
- 本目录诊断模型增加参考候选注意力、匹配MLP、局部头和残差融合门；与配置6使用相同初始化、4轮预算和后处理。
- 差值方向：正式配置6（minusFusion）− 本目录融合开启诊断；正值表示关闭融合更高。

[诊断绝对指标（所有三个阈值）](../reports/extra_diagnostics/fusion_enabled_metrics.csv) · [融合比较差值](../reports/extra_diagnostics/minusFusion_control_pp.csv) · [τ=0.4 明细](../reports/extra_diagnostics/README.md)

[返回正式30组大表](../README.md)
