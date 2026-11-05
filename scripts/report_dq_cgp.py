#!/usr/bin/env python3
"""Summary report generator for SoccerGMR with DQ-CGPv3.

Scans runs/dq_cgp_v3/ for all completed backbones and formats comparison tables
for Val mAP, Standard Test, and Full-Decodable Test.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = [
    "moment_detr",
    "eatr",
    "qd_detr",
    "cg_detr",
    "sim_detr",
    "bm_detr",
    "bam_detr",
    "uvcom",
]

# Baseline Val mAP from reproduction document
BASELINE_VAL_MAP = {
    "moment_detr": 7.35,
    "eatr": 19.34,
    "qd_detr": 17.51,
    "cg_detr": 8.65,
    "sim_detr": 22.84,
    "bm_detr": 3.61,
    "bam_detr": 25.49,
    "uvcom": 8.77,
}


def load_json(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT / "runs" / "dq_cgp_v3"))
    args = parser.parse_args()

    root = Path(args.root).resolve()

    print("\n" + "=" * 95)
    print("                SOCCERGMR WITH DQ-CGPv3 EXPERIMENT REPORT")
    print("=" * 95)

    headers = [
        "Backbone",
        "Status",
        "Best Epoch",
        "Base Val mAP",
        "DQ-CGPv3 Val mAP",
        "Δ Val",
        "Std Test mAP",
        "Std mR@1",
        "Full Test mAP",
        "Full mR@1",
    ]
    fmt = "{:<12s} | {:<10s} | {:<10s} | {:<12s} | {:<16s} | {:<8s} | {:<12s} | {:<9s} | {:<13s} | {:<9s}"
    print(fmt.format(*headers))
    print("-" * 125)

    for name in BACKBONES:
        gmr_dir = root / name / "gmr"
        test_dir = root / name / "gmr_test"

        complete_json = load_json(gmr_dir / "complete.json")
        val_metrics = load_json(gmr_dir / "best_val_metrics.json")
        history_jsonl = gmr_dir / "history.jsonl"
        std_metrics = load_json(test_dir / "test_standard" / "metrics.json")
        full_metrics = load_json(test_dir / "test_full" / "metrics.json")

        status = "pending"
        best_epoch = "-"
        dq_val_map = "-"
        delta_val = "-"

        base_val = BASELINE_VAL_MAP.get(name, 0.0)

        if complete_json:
            status = "complete" if complete_json.get("status") == "complete" else "running"
            best_epoch = str(complete_json.get("best_epoch", "-"))
            score = complete_json.get("best_score", 0.0)
            dq_val_map = f"{score:.2f}%"
            diff = score - base_val
            delta_val = f"{'+' if diff >= 0 else ''}{diff:.2f}%"
        elif (gmr_dir / "last.pt").is_file():
            status = "training"
            if history_jsonl.is_file():
                lines = [line.strip() for line in history_jsonl.read_text().splitlines() if line.strip()]
                if lines:
                    last_rec = json.loads(lines[-1])
                    cur_epoch = last_rec.get("epoch", 0)
                    best_epoch = f"cur:{cur_epoch}"
                    # Find highest so far
                    best_score = max((json.loads(l).get("selection_score", 0.0) for l in lines), default=0.0)
                    dq_val_map = f"{best_score:.2f}%"
                    diff = best_score - base_val
                    delta_val = f"{'+' if diff >= 0 else ''}{diff:.2f}%"

        std_map = f"{std_metrics['brief']['mAP']:.2f}%" if std_metrics else "-"
        std_mr1 = f"{std_metrics['brief']['mR@1']:.2f}%" if std_metrics else "-"
        full_map = f"{full_metrics['brief']['mAP']:.2f}%" if full_metrics else "-"
        full_mr1 = f"{full_metrics['brief']['mR@1']:.2f}%" if full_metrics else "-"

        print(
            fmt.format(
                name,
                status,
                best_epoch,
                f"{base_val:.2f}%",
                dq_val_map,
                delta_val,
                std_map,
                std_mr1,
                full_map,
                full_mr1,
            )
        )

    print("=" * 125 + "\n")


if __name__ == "__main__":
    main()
