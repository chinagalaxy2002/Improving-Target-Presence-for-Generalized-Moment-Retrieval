#!/usr/bin/env python3
"""Fine tune only a GEQ head on one frozen shared/multi source checkpoint."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import copy
import csv
import fcntl
import importlib.util
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from common import (BACKBONES, CORE, EPOCHS, OUTPUT, PROJECT, REFERENCE_COMMIT, SEED,
                    VARIANTS, dump, original_metrics, original_prediction, read, rows,
                    sha, source_dir, write_rows)
from model import FrozenGEQ, state_digest


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Adapter:
    def __init__(self, backbone, variant):
        self.backbone = backbone
        self.batch_size = 8 if backbone == "moment_detr" else 32
        if backbone == "moment_detr":
            sys.path.insert(0, str(PROJECT / "gmr_multiroute_kit"))
            import run_train
            self.native = run_train
            self.options, self.model, criterion = run_train.setup(variant)
            self.options.num_workers = 2
            self.options.eval_bsz = 4
            self.train_data, self.val_data = self.dataset("train"), self.dataset("val")
            del criterion
        else:
            spec = importlib.util.spec_from_file_location("native_multiroute_run", CORE / "run.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            self.experiment = module.Experiment(backbone, variant).build()
            self.experiment.microbatch = self.batch_size
            self.options = self.experiment.options
            self.model = self.experiment.model
            self.train_data = self.experiment.train_data
            self.val_data = self.experiment.val_data
            self.experiment.criterion = None
        for split, data, expected in (("train", self.train_data, 4138), ("val", self.val_data, 465)):
            if len(data) != expected:
                raise RuntimeError(f"{split} sample loss: {len(data)} != {expected}")

    def dataset(self, split):
        if self.backbone == "moment_detr":
            path = PROJECT / "soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard" / f"{split}.jsonl"
            return self.native.StartEndDataset(**self.native.build_dataset_config(
                self.options, str(path), load_labels=True, keep_empty_gt=True,
            ))
        if self.backbone == "eatr":
            return self.experiment.module.SoccerDataset(self.options, split)
        return self.experiment.module.Soccer(split)

    def loader(self, data, training=False, generator=None):
        if self.backbone != "moment_detr":
            loader = self.experiment.loader(data, training)
            if generator is not None:
                loader.generator = generator
                loader.sampler.generator = generator
            return loader
        return DataLoader(
            data, batch_size=self.batch_size if training else self.options.eval_bsz,
            shuffle=training, num_workers=2, collate_fn=self.native.start_end_collate,
            generator=generator, pin_memory=True, persistent_workers=True,
        )

    def unpack(self, batch):
        if self.backbone == "moment_detr":
            inputs, _ = self.native.prepare_batch_inputs(batch[1], self.options.device)
            return batch[0], inputs
        metadata, inputs, _, _ = self.experiment.unpack(batch)
        return metadata, inputs

    def predict_native(self, model, data):
        if self.backbone == "moment_detr":
            from evaluate import compute_mr_results
            with torch.no_grad():
                predictions, _ = compute_mr_results(None, model, self.loader(data), self.options, criterion=None)
            return predictions
        self.experiment.model = model
        return self.experiment.predict(data)


def presence_loss(output, metadata):
    logits = output["pred_exist_logits"].reshape(-1)
    labels = logits.new_tensor([bool(item["relevant_windows"]) for item in metadata])
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite presence BCE")
    return loss


def smoke(adapter, model, output):
    initial = state_digest(model.frozen)
    data = adapter.train_data
    # Mixed null/single/multi samples, then a separate all-null batch.
    groups = [[next(i for i, row in enumerate(data.data) if len(row["relevant_windows"]) == n)
               for n in (0, 1, 2)],
              [i for i, row in enumerate(data.data) if not row["relevant_windows"]][:2]]
    results = []
    for indices in groups:
        metadata, inputs = adapter.unpack(next(iter(adapter.loader(Subset(data, indices)))))
        model.train()
        predictions = model(**inputs)
        loss = presence_loss(predictions, metadata)
        loss.backward()
        if model.frozen.training or any(p.grad is not None for p in model.frozen.parameters()):
            raise RuntimeError("Frozen backbone received training state or gradients")
        active = [name for name, p in model.geq.named_parameters()
                  if p.grad is not None and bool(p.grad.abs().max() > 0)]
        if not any(name == "token" for name in active) or not any("video_reads" in name for name in active):
            raise RuntimeError("GEQ token/video evidence has no gradient")
        if any(not bool(torch.isfinite(p.grad).all()) for p in model.geq.parameters() if p.grad is not None):
            raise RuntimeError("Nonfinite GEQ gradient")
        if "o2m_outputs" in predictions:
            raise RuntimeError("Frozen auxiliary routes executed")
        with torch.no_grad():
            original = model.frozen(**inputs)
        for key in ("pred_spans", "pred_logits"):
            if not torch.equal(original[key], predictions[key]):
                raise RuntimeError(f"Raw localizer outputs changed: {key}")
        model.eval()
        with torch.no_grad():
            expected = model(**inputs)["pred_exist_logits"].clone()
        saved = copy.deepcopy(model.geq.state_dict())
        model.geq.load_state_dict(saved, strict=True)
        with torch.no_grad():
            if not torch.equal(expected, model(**inputs)["pred_exist_logits"]):
                raise RuntimeError("GEQ strict reload changed inference")
        results.append(dict(qids=[row["qid"] for row in metadata], loss=float(loss.detach())))
        model.zero_grad(set_to_none=True)
    if state_digest(model.frozen) != initial:
        raise RuntimeError("Smoke changed frozen tensors")
    dump(output / "smoke.json", dict(status="PASSED", mixed_and_all_null=results,
         raw_localization_equal=True, strict_geq_reload_equal=True, auxiliary_routes_disabled=True,
         frozen_state_unchanged=True, num_workers=2))
    print("SMOKE_PASSED", output, flush=True)


def evaluator(backbone):
    # Match each source experiment's evaluation and GT normalization exactly.
    if backbone == "moment_detr":
        directory = PROJECT / "generalized-moment-retrieval/eval"
        sys.path.insert(0, str(directory))
        from eval_main import evaluate_gmr
        from metrics import compute_G_mIoU, prepare_submission_for_gmiou
        from normalization import normalize_ground_truth
        normalize = lambda gt: normalize_ground_truth(gt, None, drop_empty_gt=False)[0]
    else:
        sys.path.insert(0, "/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared")
        from official_eval.eval_main import evaluate_gmr
        from official_eval.metrics import compute_G_mIoU, prepare_submission_for_gmiou
        normalize = lambda gt: gt
    return evaluate_gmr, compute_G_mIoU, prepare_submission_for_gmiou, normalize


def evaluate_predictions(backbone, predictions, gt, output, prefix):
    evaluate, gmiou, gate, normalize = evaluator(backbone)
    gt = normalize(gt)
    if len(predictions) != len(gt) or len({p["qid"] for p in predictions}) != len(gt):
        raise RuntimeError("Prediction count or unique-QID mismatch")
    if {p["qid"] for p in predictions} != {p["qid"] for p in gt}:
        raise RuntimeError("Prediction/GT QID coverage differs")
    metrics = {}
    for threshold in (0.4, 0.5, 0.6):
        metrics[str(threshold)] = evaluate(
            predictions, gt, cls_thresholds=(0.4, 0.5, 0.6),
            gmiou_cls_threshold=threshold, map_num_workers=0, verbose=False,
        )
    dump(output / f"{prefix}_metrics.json", metrics)
    sweep = []
    for index in range(101):
        threshold = index / 100
        gated, stats = gate(predictions, cls_threshold=threshold, max_pred_windows=10)
        values = gmiou(gated, gt, k_list=(1, 3, 5))
        sweep.append(dict(threshold=threshold, **{k: float(v) / 100 for k, v in values.items()},
                          predicted_positive=stats["num_predicted_positive"],
                          predicted_negative=stats["num_predicted_negative"]))
    with (output / f"{prefix}_sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sweep[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(sweep)
    for key in ("G-mIoU@1", "G-mIoU@3", "G-mIoU@5"):
        if not math.isclose(sweep[40][key], metrics["0.4"]["brief"][key] / 100, abs_tol=1e-8):
            raise RuntimeError("Sweep and fixed-threshold metric disagree")
    return metrics


def run(args, output):
    torch.set_num_threads(2)
    seed_all()
    source = source_dir(args.backbone, args.variant) / "best.ckpt"
    source_sha = sha(source)
    adapter = Adapter(args.backbone, args.variant)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    adapter.model.load_state_dict(checkpoint["model"], strict=True)
    source_epoch = checkpoint.get("epoch")
    del checkpoint
    seed_all()  # Paired shared/multi GEQ starts with identical new-head tensors.
    model = FrozenGEQ(adapter.model, args.backbone).cuda()
    frozen_digest = state_digest(model.frozen)
    config = dict(backbone=args.backbone, variant=args.variant, seed=SEED, epochs=EPOCHS,
        batch_size=adapter.batch_size, gradient_accumulation=1, num_workers=2, amp=False,
        lr=3e-5, weight_decay=1e-4, grad_clip=0.1, lr_drop=8, lr_gamma=0.5,
        selection="fixed final epoch 4; no Test-based checkpoint or threshold selection",
        reference_commit=REFERENCE_COMMIT, reference_seed=2026,
        source_checkpoint=str(source), source_sha256=source_sha, source_checkpoint_epoch_raw=source_epoch,
        source_prediction=str(original_prediction(args.backbone, args.variant)),
        source_prediction_sha256=sha(original_prediction(args.backbone, args.variant)),
        frozen_state_sha256=frozen_digest, initial_geq_sha256=state_digest(model.geq),
        trainable_names=[name for name, p in model.named_parameters() if p.requires_grad],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        feature_video_offset=int(args.backbone in ("qd_detr", "cg_detr")),
        only_geq=True, train_samples=len(adapter.train_data), val_samples=len(adapter.val_data),
        code_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")})
    existing = read(output / "experiment_config.json")
    for key in ("source_sha256", "source_prediction_sha256", "seed", "batch_size", "only_geq"):
        if existing and existing[key] != config[key]:
            raise RuntimeError(f"Refusing incompatible resume: {key}")
    if not existing:
        dump(output / "experiment_config.json", config)
    smoke(adapter, model, output)
    if args.smoke_only:
        return
    optimizer = torch.optim.AdamW(model.geq.parameters(), lr=3e-5, weight_decay=1e-4, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    seed_all()
    generator = torch.Generator().manual_seed(SEED)
    start_epoch = 0
    last = output / "last.ckpt"
    if last.is_file():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["source_sha256"] != source_sha:
            raise RuntimeError("Resume source changed")
        model.geq.load_state_dict(state["geq"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        for group in optimizer.state.values():
            for key, value in group.items():
                if torch.is_tensor(value) and key != "step":
                    group[key] = value.cuda()
        start_epoch = state["epoch"]
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
        generator.set_state(state["loader_generator"])
        print("RESUMED_EPOCH", start_epoch, flush=True)
    loader = adapter.loader(adapter.train_data, True, generator)
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        model.train()
        begun = time.monotonic()
        total, count = 0.0, 0
        for step, batch in enumerate(loader, 1):
            metadata, inputs = adapter.unpack(batch)
            optimizer.zero_grad(set_to_none=True)
            loss = presence_loss(model(**inputs), metadata)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.geq.parameters(), 0.1, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(metadata)
            count += len(metadata)
            if step == 1 or step % 50 == 0:
                progress = dict(status="TRAINING", epoch=epoch, step=step, steps=len(loader),
                                loss=float(loss.detach()), pid=os.getpid())
                dump(output / "status.json", progress)
                print(json.dumps(progress), flush=True)
        scheduler.step()
        if count != 4138 or state_digest(model.frozen) != frozen_digest:
            raise RuntimeError("Missing train samples or frozen tensors changed")
        if any(p.grad is not None for p in model.frozen.parameters()):
            raise RuntimeError("Frozen gradient detected")
        record = dict(epoch=epoch, mean_loss=total / count, samples=count,
                      seconds=time.monotonic() - begun, frozen_state_unchanged=True,
                      peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20)
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        temporary = last.with_suffix(".tmp")
        torch.save(dict(geq=model.geq.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), epoch=epoch, source_sha256=source_sha,
            rng_python=random.getstate(), rng_numpy=np.random.get_state(),
            rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
            loader_generator=generator.get_state()), temporary)
        os.replace(temporary, last)
        print("EPOCH_DONE", json.dumps(record), flush=True)
    del loader
    model.eval()
    final = torch.load(last, map_location="cpu", weights_only=False)
    if final["epoch"] != EPOCHS:
        raise RuntimeError("Incomplete training")
    model.geq.load_state_dict(final["geq"], strict=True)
    if state_digest(model.geq) == config["initial_geq_sha256"]:
        raise RuntimeError("GEQ did not change during fine tuning")
    final_path = output / "final.ckpt"
    torch.save(dict(geq=final["geq"], epoch=EPOCHS, source_sha256=source_sha,
                    source_checkpoint=str(source), config=config), final_path)
    dump(output / "status.json", dict(status="TESTING", epoch=EPOCHS, pid=os.getpid()))
    # Validation is reported once at fixed epoch 4 and never selects a checkpoint.
    with torch.no_grad():
        val_predictions = adapter.predict_native(model, adapter.val_data)
    write_rows(output / "val_predictions.jsonl", val_predictions)
    evaluate_predictions(args.backbone, val_predictions, adapter.val_data.data, output, "val_after")
    test = adapter.dataset("test")
    if len(test) != 1036:
        raise RuntimeError("Incomplete Test set")
    scores = {}
    with torch.no_grad():
        for batch in adapter.loader(test):
            metadata, inputs = adapter.unpack(batch)
            probabilities = model(**inputs)["pred_exist_score"].detach().cpu().tolist()
            for meta, probability in zip(metadata, probabilities):
                if meta["qid"] in scores or not math.isfinite(probability):
                    raise RuntimeError("Duplicate QID/nonfinite presence score")
                scores[meta["qid"]] = probability
    before = rows(original_prediction(args.backbone, args.variant))
    if set(scores) != {row["qid"] for row in before} or len(before) != 1036:
        raise RuntimeError("Original Test proposal set and GEQ QID coverage differ")
    after = copy.deepcopy(before)
    for row in after:
        row["pred_exist_score"] = scores[row["qid"]]
    for old, new in zip(before, after):
        if {k: v for k, v in old.items() if k != "pred_exist_score"} != {
            k: v for k, v in new.items() if k != "pred_exist_score"
        }:
            raise RuntimeError("Frozen exported proposals changed")
    write_rows(output / "test_before_predictions.jsonl", before)
    write_rows(output / "test_after_predictions.jsonl", after)
    before_metrics = evaluate_predictions(args.backbone, before, test.data, output, "test_before")
    canonical = read(original_metrics(args.backbone, args.variant))["brief"]
    for key, value in canonical.items():
        if key in before_metrics["0.4"]["brief"] and not math.isclose(float(value), before_metrics["0.4"]["brief"][key], abs_tol=1e-6):
            raise RuntimeError(f"Baseline metric mismatch: {key}, {value}, {before_metrics['0.4']['brief'][key]}")
    after_metrics = evaluate_predictions(args.backbone, after, test.data, output, "test_after")
    for threshold in ("0.4", "0.5", "0.6"):
        for key in ("mAP", "mR@1", "mR@5", "mR+@5", "mIoU@1", "mIoU@3"):
            if before_metrics[threshold]["brief"][key] != after_metrics[threshold]["brief"][key]:
                raise RuntimeError(f"Localization metric changed: {key}")
    if state_digest(model.frozen) != frozen_digest or sha(source) != source_sha:
        raise RuntimeError("Source checkpoint/frozen state changed")
    dump(output / "COMPLETE.json", dict(status="COMPLETED", backbone=args.backbone,
        variant=args.variant, epochs=EPOCHS, test_samples=len(after), val_samples=len(val_predictions),
        frozen_state_unchanged=True, source_checkpoint_unchanged=True,
        all_test_proposals_and_ranking_scores_exactly_equal=True,
        baseline_metrics_reproduced=True, localization_metrics_equal=True,
        final_checkpoint_sha256=sha(final_path), source_sha256=source_sha,
        final_geq_sha256=state_digest(model.geq), geq_updated=True,
        before=before_metrics["0.4"]["brief"], after=after_metrics["0.4"]["brief"]))
    dump(output / "status.json", dict(status="COMPLETED", epoch=EPOCHS, pid=os.getpid()))
    print("COMPLETE", args.backbone, args.variant, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    output = OUTPUT / args.backbone / args.variant
    output.mkdir(parents=True, exist_ok=True)
    with (output / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(output / "COMPLETE.json").get("status") == "COMPLETED":
            print("ALREADY_COMPLETE", output, flush=True)
            return
        try:
            run(args, output)
        except Exception:
            dump(output / "failure.json", dict(time=time.time(), pid=os.getpid(), traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
