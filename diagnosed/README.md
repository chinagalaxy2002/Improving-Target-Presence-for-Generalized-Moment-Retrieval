# SoccerGMR 诊断实验目录 (`diagnosed/`)

本目录严格依据 [`diagnosed/实验设计.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/实验设计.md) 实现了对 SoccerGMR 框架中全部 8 个 DETR 骨干网络（Method 1 原生基线与 Method 2 DQ-CGPv3）的底层机理诊断。

## 📖 核心实验索引 (五大科学问题)
- **实验 1：Multi-GT 数据分布审计**（[`run_exp1_multigt.py`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/run_exp1_multigt.py) $\to$ [`tables/exp1_multigt_summary.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/exp1_multigt_summary.md)）
- **实验 2：Decoder Slot 活跃度与索引塌缩**（Table D: [`tables/table_d_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/)）
- **实验 3：候选预算分解与候选拥挤度**（Table A: [`tables/table_a_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/)）
- **实验 4：生成缺陷 vs 排序缺陷加法分解**（Table B: [`tables/table_b_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/)）
- **实验 5：Decoder 逐层演化与退化归因**（Table C: [`tables/table_c_*.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/tables/)）
- **综合主报告**：[`实验报告_诊断分析结论.md`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed/实验报告_诊断分析结论.md)

---

> [!NOTE]
> **补充实验独立归档导航**：
> 针对审稿关切的四项补充实验（检查点加载与指标一致性核验、首次命中排名口径修复、多时刻 $M \ge 2$ 查询 4 态覆盖与分工分析、IoU=0.3/0.5 敏感性检查），已按照要求独立归档至单独文件夹：
> 👉 **[`diagnosed_supplementary/`](file:///home/guoxiangyu/SoccerGMR-DQ-CGPv3/diagnosed_supplementary/)**
