# Figure Designer 检查：Sim-DETR / SoccerGMR Standard

使用技能：`figure-designer`，本地来源 `/home/guoxiangyu/.codex/skills/figure-designer/SKILL.md`。

## 1. 图类型

Experimental Results：比较GMR、GEQ Full/Frozen的存在性阈值敏感性和Val选模，保持既有README的图形结构。

## 2. 设计范式

阈值扫描用折线，拒绝/保留用关系曲线，三个固定工作点用分组柱。相比只显示0.4处的柱图，101点扫描能同时显示校准收益和部分阈值处的下降。

## 3. 布局

六面板按2×3排列：G1、G3、相对同一GMR的G1增减、N-acc/positive recall、拒绝F1、三个固定工作点。另提供G1/G3双面板和Val双面板。没有使用无实测依据的Oracle或置信区间。

## 4. 标签和编码

图例为GMR baseline、GMR + GEQ Full、GMR + GEQ Frozen，并列出实测mAP。灰色虚线圆点、橙色点线菱形、绿色点划线三角同时区分方法，不仅依赖颜色。明确写SoccerGMR Standard Test/Val，不标成Charades。G-mIoU使用0–1，mAP使用百分数。

## 5. 工具

复用仓库`shared/geq_protocol/plotting.py`，由NumPy/Matplotlib导出PNG和真实矢量PDF/SVG；101点CSV与JSON快照支持离线重绘。元数据日期清除后，六份Test图PNG/PDF/SVG从快照重绘逐字节一致。

## 6. 通用设计检查

- 矢量格式：通过，PDF/SVG未包裹截图。
- 字体：全宽README与独立PDF中可读，已查看三张PNG及实际PDF渲染；论文单栏排版时应使用拆分图并按最终尺寸设置字体，未声称六面板缩到任意尺寸仍≥8pt。
- 色觉友好：通过，颜色、线型、marker双重编码。
- 自包含：通过，图例、数据集、指标单位、阈值和选模依据明确；README首段给出Full相对GMR的实测收益。
- 坐标诚实：G1/G3绝对值范围0–1，差值保留负数，未隐藏Frozen局部下降；Val mAP折线明确标出所用数值范围。
- 装饰：没有3D、渐变、阴影或人工修改数据。

## 7. 完整性门槛

范式与数据类型一致；所有系列有真实方法名；固定点来自官方重算，曲线来自完整1036条预测；相对增益以相同GMR为基准。独立审核见`AUDIT.json`，实际查看产物的当前SHA见`VISUAL_REVIEW.json`。本次不依赖论文Introduction的示例，因此该项不适用。

## 8. 严重性

当前独立图/README交付：0 CRITICAL、0 MAJOR。论文最终版面的缩放需按投稿尺寸重新检查字体，这是排版限制，不影响本次发布验收。
