#!/usr/bin/env python3
"""Collect completed 0-1-domain GMR metrics into CSV and Markdown."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKBONES = ("eatr", "qd_detr", "cg_detr")
VARIANTS = ("baseline", "shared", "sepffn", "instruction", "multi", "adaptive")
FIELDS = ("mAP", "mR@1", "mR@5", "mR+@5", "AUROC", "Rej-F1", "G-mIoU@3 (0-1)")


def find_metric(brief, name):
    aliases = {
        "mR@1": ("mR@1", "R@1"),
        "mR@5": ("mR@5", "R@5"),
        "mR+@5": ("mR+@5", "mR+_at_5"),
        "Rej-F1": ("Rej-F1@0.4", "Rej-F1", "Rej_F1"),
        "G-mIoU@3 (0-1)": ("G-mIoU@3", "G-mIoU", "G_mIoU@3"),
    }
    for key in aliases.get(name, (name,)):
        if key in brief:
            value = brief[key]
            value = float(value)
            return value / 100.0 if name == "G-mIoU@3 (0-1)" else value
    return ""


rows = []
for backbone in BACKBONES:
    for variant in VARIANTS:
        path = HERE / "runs" / backbone / variant / "test_metrics.json"
        brief = json.loads(path.read_text())["brief"] if path.exists() else {}
        rows.append({"backbone": backbone, "variant": variant, **{field: find_metric(brief, field) for field in FIELDS}})

with (HERE / "results.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=("backbone", "variant", *FIELDS))
    writer.writeheader()
    writer.writerows(rows)

lines = ["# Multi-route results (single seed 2023)", "", "G-mIoU is explicitly converted from the official percentage output to its full 0-1 domain. Other metrics retain the official evaluator's percentage scale.", ""]
for backbone in BACKBONES:
    lines += [f"## {backbone}", "", "| Variant | " + " | ".join(FIELDS) + " |", "|---|" + "---:|" * len(FIELDS)]
    for row in (item for item in rows if item["backbone"] == backbone):
        values = ["—" if row[field] == "" else f'{row[field]:.4f}' for field in FIELDS]
        lines.append(f'| {row["variant"]} | ' + " | ".join(values) + " |")
    lines.append("")
(HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
print(HERE / "RESULTS.md")
