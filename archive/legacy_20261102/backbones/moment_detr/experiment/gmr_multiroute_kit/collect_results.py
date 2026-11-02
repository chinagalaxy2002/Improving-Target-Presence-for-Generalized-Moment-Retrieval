"""Collect the current six-checkpoint test metrics into JSON and Markdown."""
from __future__ import annotations

import json
import os
import pathlib

ROOT = pathlib.Path(os.environ.get("GMR_PROJECT_ROOT", pathlib.Path(__file__).resolve().parents[1]))
EXPERIMENTS = ("baseline", "shared", "sepffn", "instruction", "multi", "adaptive")


def main() -> None:
    rows = []
    for variant in EXPERIMENTS:
        test_dir = ROOT / "mr_experiments" / variant / "test"
        manifest = json.loads((test_dir / "evaluation_manifest.json").read_text())
        primary = json.loads((test_dir / "metrics_gmiou_0.4.json").read_text())
        brief = primary["brief"]
        row = {
            "experiment": variant,
            "best_epoch": manifest["best_epoch"],
            "mAP": brief["mAP"],
            "mR@1": brief["mR@1"],
            "mR@5": brief["mR@5"],
            "mR+@5": brief["mR+@5"],
            "AUROC": brief["AUROC"],
            "Rej-F1@0.4": brief["Rej-F1@0.4"],
            "Rej-F1@0.5": brief["Rej-F1@0.5"],
            "Rej-F1@0.6": brief["Rej-F1@0.6"],
            "G-mIoU@3 (gate 0.4)": brief["G-mIoU@3"],
        }
        for threshold in (0.5, 0.6):
            metrics = json.loads((test_dir / f"metrics_gmiou_{threshold:.1f}.json").read_text())
            row[f"G-mIoU@3 (gate {threshold:.1f})"] = metrics["brief"]["G-mIoU@3"]
        rows.append(row)

    output_dir = ROOT / "mr_experiments" / "test_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(str(row[col]) for col in columns) + " |" for row in rows]
    (output_dir / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
