"""Describe measured primary-threshold effects without selecting a new model."""
import json
from pathlib import Path
from summarize import BACKBONES, RESULTS, METRICS, values, delta, write_csv

ROOT = Path(__file__).resolve().parent


def main():
    lines = ['# 两组件结果解读：固定第 4 个微调 epoch', '',
             '主工作点 τ=0.4，Standard Test，seed 2026。下表为 Global Event Query + Hard-RoI；括号为相对同主干 GMR 的变化，单位 pp。', '',
             '| Backbone | AUROC | Rej-F1 | mAP | mR+@5 | G-mIoU@1 |',
             '|---|---:|---:|---:|---:|---:|']
    comparisons = []
    hard_deltas = []
    g_deltas = []
    old_control = ROOT.parent / 'component_ablation_20261023'
    for backbone, name in BACKBONES.items():
        reports = {setting: values(json.loads((RESULTS / backbone / setting / 'test_metrics.json').read_text())['0.4'], '0.4')
                   for setting in ['gmr', 'global_only', 'hard_only', 'global_hard']}
        full = reports['global_hard']
        changes = delta(full, reports['gmr'])
        lines.append('| ' + name + ' | ' + ' | '.join(f'{full[k]:.2f} ({changes[k]:+.2f})' for k in ['AUROC', 'Rej-F1', 'mAP', 'mR+@5', 'G-mIoU@1']) + ' |')
        conditional = delta(full, reports['global_only'])
        hard_deltas.append(conditional['mAP'])
        g_deltas.append(changes['G-mIoU@1'])
        for reference, label in [('gmr', 'Combined versus GMR'), ('global_only', 'Hard added to Global (no LoRA)'), ('hard_only', 'Global added to Hard (no LoRA)')]:
            comparisons.append(dict(backbone=backbone, comparison=label, **delta(full, reports[reference])))
        old_results = old_control / ('runs' if (old_control / 'runs').exists() else 'results/formal')
        previous = old_results / backbone / 'row_04/test_metrics.json'
        if previous.exists():
            old = values(json.loads(previous.read_text())['0.4'], '0.4')
            comparisons.append(dict(backbone=backbone, comparison='Removing LoRA given Global + Hard (previous row_04 reference)', **delta(full, old)))
    increases = sum(d > 0 for d in hard_deltas)
    decreases = sum(d < 0 for d in hard_deltas)
    lines.extend(['',
        f'无 LoRA 时，在 Global-only 上增加 Hard-RoI 的 Test mAP 变化为 {min(hard_deltas):+.2f} 至 {max(hard_deltas):+.2f} pp：{increases}/5 个主干上升、{decreases}/5 个下降。旧消融中 Hard-RoI 的增益是在 LoRA 开启时测得，不能直接归因到当前无 LoRA 方法。', '',
        f'精简组合相对 GMR 的 G-mIoU@1 变化为 {min(g_deltas):+.2f} 至 {max(g_deltas):+.2f} pp。该综合指标受存在性拒识与定位共同影响；应同时查看 mAP、mR+@5、Rej-F1 与 AUROC，避免把拒识收益写成一致的定位收益。', '',
        '保留完整 2×2 对照，可分别回答「Global 是否改善存在性」「Hard 是否改善定位」「两者组合是否优于各自单独使用」。条件差异和差中差基于一次训练，含随机优化及联合梯度裁剪影响，不证明统计显著性或多种子稳定交互。', '',
        '本次不调整超参数、不改变固定第 4 轮、不按 Test 重新选择方案。后续若要确认方法稳定性，应先冻结这套设计，再使用额外种子及独立评估数据。', '',
        '[完整四配置结果](reports/RESULTS.md) · [条件比较 CSV](reports/PRIMARY_COMPARISONS.csv) · [独立数值复算](reports/METRIC_VERIFICATION.json)'])
    (ROOT / 'CONCLUSIONS.md').write_text('\n'.join(lines) + '\n')
    write_csv(ROOT / 'reports/PRIMARY_COMPARISONS.csv', comparisons)


if __name__ == '__main__':
    main()
