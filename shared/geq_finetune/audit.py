#!/usr/bin/env python3
"""Independent CPU audit of the complete eight-job GEQ fine-tuning request."""
import argparse
import csv
import math
from pathlib import Path

import torch

from common import (BACKBONES, EPOCHS, FIGURES, OUTPUT, VARIANTS, dump,
                    original_metrics, original_prediction, read, rows, sha, source_dir)
from model import GlobalEventQueryHead, state_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-baseline", action="store_true")
    args = parser.parse_args()
    variants = ("baseline", *VARIANTS) if args.include_baseline else VARIANTS
    torch.set_num_threads(2)
    checked = []
    for backbone in BACKBONES:
        for variant in variants:
            directory = OUTPUT / backbone / variant
            complete = read(directory / "COMPLETE.json")
            assert complete.get("status") == "COMPLETED", f"Unfinished: {directory}"
            config = read(directory / "experiment_config.json")
            assert config["only_geq"] and config["seed"] == 2023
            assert config["epochs"] == EPOCHS == complete["epochs"]
            assert config["num_workers"] == 2
            assert all(name.startswith("geq.") for name in config["trainable_names"])
            assert sha(config["source_checkpoint"]) == config["source_sha256"] == complete["source_sha256"]
            assert sha(original_prediction(backbone, variant)) == config["source_prediction_sha256"]
            assert read(directory / "smoke.json")["status"] == "PASSED"
            history = rows(directory / "history.jsonl")
            assert [r["epoch"] for r in history] == [1, 2, 3, 4]
            assert all(r["samples"] == 4138 and r["frozen_state_unchanged"] and math.isfinite(r["mean_loss"]) for r in history)
            for key in ("frozen_state_unchanged", "source_checkpoint_unchanged", "baseline_metrics_reproduced",
                        "all_test_proposals_and_ranking_scores_exactly_equal", "localization_metrics_equal", "geq_updated"):
                assert complete[key], key
            state = torch.load(directory / "final.ckpt", map_location="cpu", weights_only=False)
            assert state["epoch"] == 4 and state["source_sha256"] == config["source_sha256"]
            assert sha(directory / "final.ckpt") == complete["final_checkpoint_sha256"]
            depth = max(int(name.split(".")[1]) for name in state["geq"] if name.startswith("video_reads.")) + 1
            head = GlobalEventQueryHead(depth)
            head.load_state_dict(state["geq"], strict=True)
            assert state_digest(head) == complete["final_geq_sha256"] != config["initial_geq_sha256"]
            before = rows(directory / "test_before_predictions.jsonl")
            after = rows(directory / "test_after_predictions.jsonl")
            assert before == rows(original_prediction(backbone, variant))
            assert len(before) == len(after) == 1036
            assert len({p["qid"] for p in after}) == 1036
            for old, new in zip(before, after):
                assert 0 <= new["pred_exist_score"] <= 1 and math.isfinite(new["pred_exist_score"])
                assert {k: v for k, v in old.items() if k != "pred_exist_score"} == {k: v for k, v in new.items() if k != "pred_exist_score"}
            before_metrics = read(directory / "test_before_metrics.json")
            after_metrics = read(directory / "test_after_metrics.json")
            canonical = read(original_metrics(backbone, variant))["brief"]
            for key, value in canonical.items():
                if key in before_metrics["0.4"]["brief"]:
                    assert math.isclose(float(value), before_metrics["0.4"]["brief"][key], abs_tol=1e-6), key
            for threshold in ("0.4", "0.5", "0.6"):
                for key in ("mAP", "mR@1", "mR@5", "mR+@5"):
                    assert before_metrics[threshold]["brief"][key] == after_metrics[threshold]["brief"][key]
            for phase, metrics in (("before", before_metrics), ("after", after_metrics)):
                with (directory / f"test_{phase}_sweep.csv").open(newline="") as stream:
                    sweep = list(csv.DictReader(stream))
                assert len(sweep) == 101
                for index, row in enumerate(sweep):
                    assert math.isclose(float(row["threshold"]), index / 100, abs_tol=1e-8)
                    assert all(0 <= float(row[key]) <= 1 for key in ("G-mIoU@1", "G-mIoU@3", "G-mIoU@5"))
                for threshold in ("0.4", "0.5", "0.6"):
                    assert math.isclose(float(sweep[round(float(threshold) * 100)]["G-mIoU@3"]),
                                        metrics[threshold]["brief"]["G-mIoU@3"] / 100, abs_tol=1e-8)
            for extension in ("png", "pdf", "svg"):
                assert (FIGURES / backbone / f"gmiou_before_after.{extension}").stat().st_size > 1000
            checked.append(dict(backbone=backbone, variant=variant, status="PASSED"))
    for backbone in BACKBONES:
        paired = [read(OUTPUT / backbone / variant / "experiment_config.json") for variant in variants]
        assert len({item["initial_geq_sha256"] for item in paired}) == 1
    assert read(OUTPUT / "REFERENCE_EQUIVALENCE.json")["masked_batch_outputs_exactly_equal"]
    assert len(read(OUTPUT / "BASELINE_METRIC_AUDIT.json")) == 8
    assert read(OUTPUT / "QUEUE.json")["status"] == "COMPLETED"
    if args.include_baseline:
        assert read(OUTPUT / "BASELINE_GEQ_QUEUE.json")["status"] == "COMPLETED"
        assert all(read(OUTPUT / b / "baseline/experiment_config.json")["source_checkpoint"] ==
                   str(source_dir(b, "baseline") / "best.ckpt") for b in BACKBONES)
    report_name = "FINAL_AUDIT_12_GEQ.json" if args.include_baseline else "FINAL_AUDIT.json"
    dump(OUTPUT / report_name, dict(status="PASSED", jobs=checked, count=len(checked),
         exact_frozen_test_proposals=True, full_domain_0_1=True, paired_geq_initialization_equal=True))
    print("FINAL_AUDIT_PASSED", len(checked))


if __name__ == "__main__":
    main()
