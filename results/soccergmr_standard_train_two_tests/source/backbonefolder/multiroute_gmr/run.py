#!/usr/bin/env python3
"""Unified smoke/train/test runner for three GMR multi-route backbones."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

# This directory contains queue.py; remove it temporarily so PyTorch imports the
# Python standard-library ``queue`` module instead of the local scheduler.
_SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
if sys.path and str(Path(sys.path[0]).resolve()) == _SCRIPT_DIRECTORY:
    sys.path.pop(0)

import numpy as np
# NumPy 2.0 removed the historical alias used by the frozen official GMR
# evaluator. Keep the evaluator source unchanged and provide its exact
# trapezoidal-integration compatibility locally.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
BACKBONES = HERE.parent
sys.path.insert(0, str(HERE))
from multiroute import MultiRouteCriterion, MultiRouteModel, VARIANTS
from execution_plan import job_profile, prepare_restart

SHARED_EVAL = Path("/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared")
sys.path.insert(0, str(SHARED_EVAL))
from official_eval.eval_main import evaluate_gmr

SEED = 2023
EPOCHS = 400
MICROBATCH = 2
ACCUMULATION = 4
EVAL_EVERY = 5
EARLY_STOP_PATIENCE_EPOCHS = 20
NUM_WORKERS = 2
FIGURE_REFRESH = Path(
    os.environ.get(
        "GMR_FIGURE_REFRESH",
        "/home/guoxiangyu/ImprovingGeneralizedMomentRetrieval/newidea/figure/"
        "gmr_mr_detr_test_20261030/refresh_all.sh",
    )
)


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def jsonl(path: Path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(value) + "\n" for value in values))
    os.replace(temporary, path)


@contextmanager
def exclusive_job(backbone, variant):
    """Serialize newly dispatched jobs with later attempts from the old queue."""
    lock_dir = HERE / "job_locks"
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f"{backbone}_{variant}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"WAITING_FOR_JOB_LOCK {backbone}/{variant}", flush=True)
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """Deterministically seed NumPy/Python inside each DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def cg_collate_cpu(items):
    """CG-DETR collate without CUDA calls, safe in forked DataLoader workers."""
    metadata, videos, queries, spans, labels, positive, negative = zip(*items)

    def pad(sequence):
        padded = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=True)
        lengths = torch.tensor([len(value) for value in sequence])
        mask = torch.arange(padded.shape[1])[None, :] < lengths[:, None]
        return padded, mask.float()

    videos, video_mask = pad(videos)
    queries, query_mask = pad(queries)
    labels, _ = pad(labels)
    inputs = {
        "src_vid": videos,
        "src_vid_mask": video_mask,
        "src_txt": queries,
        "src_txt_mask": query_mask,
        "vid": None,
        "qid": [item["qid"] for item in metadata],
    }
    targets = {
        "span_labels": [{"spans": value} for value in spans],
        "saliency_all_labels": labels,
        "relevant_clips": labels,
        "saliency_pos_labels": torch.tensor(positive),
        "saliency_neg_labels": torch.tensor(negative),
    }
    return list(metadata), inputs, targets


def load_source(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class EaTRGMRCriterion(nn.Module):
    """EaTR native loss on positive queries and existence BCE on all queries."""

    def __init__(self, native, subset_function):
        super().__init__()
        self.native = native
        self.subset_function = subset_function
        self.weight_dict = {**native.weight_dict, "loss_exist": 1.0}
        self.epoch = 0

    def forward(self, outputs, targets):
        labels = targets["exist_label"].float()
        positive = labels.bool()
        if positive.any():
            ids = positive.nonzero().flatten().tolist()
            selected_targets = {"span_labels": [targets["span_labels"][i] for i in ids]}
            losses = self.native(self.subset_function(outputs, positive), selected_targets, self.epoch)
        else:
            zero = outputs["pred_logits"].sum() * 0 + outputs["pred_spans"].sum() * 0
            losses = {key: zero for key in self.native.weight_dict}
        losses["loss_exist"] = F.binary_cross_entropy_with_logits(outputs["pred_exist_logits"], labels)
        return losses


class Experiment:
    def __init__(self, backbone: str, variant: str):
        self.backbone = backbone
        self.variant = variant
        self.execution_profile = job_profile(backbone, variant)
        self.microbatch = self.execution_profile['microbatch']
        self.accumulation = self.execution_profile['gradient_accumulation']
        self.out = HERE / "runs" / backbone / variant
        self.out.mkdir(parents=True, exist_ok=True)
        self.module = None
        self.options = None
        self.model = None
        self.criterion = None
        self.matcher_owner = None
        self.train_data = None
        self.val_data = None

    def build(self):
        seed_all()
        if self.backbone == "eatr":
            source = BACKBONES / "eatr"
            self.module = load_source("multiroute_eatr_deploy", source / "deploy.py")
            options = self.module.options("gmr")
            options.seed = SEED
            options.lr = 5e-5
            options.n_epoch = EPOCHS
            options.bsz = self.microbatch
            options.eval_bsz = self.execution_profile.get('eval_bsz', 32)
            options.num_workers = NUM_WORKERS
            base, native = self.module.build_model(options)
            base.exist_head = self.module.MLP(options.hidden_dim, options.hidden_dim, 1, 2)
            main_criterion = EaTRGMRCriterion(native, self.module.subset)
            matcher_owner = native
            dataset = self.module.SoccerDataset
            self.train_data, self.val_data = dataset(options, "train"), dataset(options, "val")
        elif self.backbone == "qd_detr":
            source = BACKBONES / "qd_detr"
            self.module = load_source("multiroute_qd_runner", source / "train_gmr.py")
            options = self.module.options()
            options.seed = SEED
            options.lr = 5e-5
            options.n_epoch = EPOCHS
            options.bsz = self.microbatch
            options.eval_bsz = 32
            options.num_workers = NUM_WORKERS
            base, main_criterion = self.module.build_model(options)
            matcher_owner = main_criterion.native
            self.train_data, self.val_data = self.module.Soccer("train"), self.module.Soccer("val")
        else:
            source = BACKBONES / "cg_detr"
            self.module = load_source("multiroute_cg_runner", source / "train_gmr.py")
            options = self.module.config()
            options.seed = SEED
            options.lr = 5e-5
            options.n_epoch = EPOCHS
            options.bsz = self.microbatch
            options.eval_bsz = 16
            options.num_workers = NUM_WORKERS
            base, main_criterion = self.module.build_model(options)
            main_criterion.__class__ = self.module.GMRCriterion
            main_criterion.weight_dict["loss_exist"] = 1.0
            matcher_owner = main_criterion
            self.train_data, self.val_data = self.module.Soccer("train"), self.module.Soccer("val")

        self.options = options
        self.matcher_owner = matcher_owner
        self.model = MultiRouteModel(base, self.variant, self.backbone, options.hidden_dim).cuda()
        self.criterion = MultiRouteCriterion(main_criterion, matcher_owner, self.variant).cuda()
        return self

    def loader(self, data, training=False):
        generator = torch.Generator().manual_seed(SEED)
        collate_function = cg_collate_cpu if self.backbone == "cg_detr" else self.module.collate
        worker_function = self.module.worker_seed if self.backbone == "eatr" else seed_worker
        return DataLoader(
            data,
            batch_size=self.microbatch if training else self.options.eval_bsz,
            shuffle=training,
            collate_fn=collate_function,
            num_workers=NUM_WORKERS,
            worker_init_fn=worker_function,
            generator=generator,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def unpack(self, batch):
        if self.backbone == "eatr":
            metadata, inputs, targets = self.module.cuda_batch(batch)
            return metadata, inputs, targets, {}
        if self.backbone == "qd_detr":
            metadata, inputs, targets = batch
            inputs, targets = self.module.move(inputs, targets, self.options.device)
            return metadata, inputs, targets, {}
        metadata, inputs, targets = batch
        inputs = {key: (value.cuda(non_blocking=True) if torch.is_tensor(value) else value) for key, value in inputs.items()}
        targets = {
            key: ([{"spans": item["spans"].cuda(non_blocking=True)} for item in value]
                  if key == "span_labels" else value.cuda(non_blocking=True))
            for key, value in targets.items()
        }
        return metadata, inputs, targets, {"targets": targets}

    def predict(self, data):
        if self.backbone == "eatr":
            return self.module.predict(self.model, self.options, data)
        loader = self.loader(data, False)
        if self.backbone == "qd_detr":
            return self.module.predictions(self.model, loader, self.options.device)

        # CG-DETR's original infer() assumes its collate function already moved
        # batches to CUDA. Our worker-safe collate is deliberately CPU-only, so
        # perform the transfer in the main process before yielding each batch.
        experiment = self

        class MainProcessDeviceLoader:
            def __init__(self, source):
                self.source = source
                self.dataset = source.dataset

            def __len__(self):
                return len(self.source)

            def __iter__(self):
                for batch in self.source:
                    metadata, inputs, targets, _ = experiment.unpack(batch)
                    yield metadata, inputs, targets

        return self.module.infer(self.model, MainProcessDeviceLoader(loader))

    @property
    def ground_truth(self):
        return self.val_data.data


def weighted_loss(criterion, losses):
    included = [losses[key] * weight for key, weight in criterion.weight_dict.items() if key in losses]
    if not included:
        raise RuntimeError("No weighted losses were produced")
    return sum(included)


def smoke(experiment: Experiment):
    model, criterion = experiment.model, experiment.criterion
    positive = next(i for i, item in enumerate(experiment.train_data.data) if len(item["relevant_windows"]) >= 2)
    negative = next(i for i, item in enumerate(experiment.train_data.data) if not item["relevant_windows"])
    indices = [positive, negative]
    if experiment.execution_profile.get('action') == 'restart':
        indices.extend(i for i in range(len(experiment.train_data)) if i not in (positive, negative))
        indices = indices[:experiment.microbatch]
    subset = torch.utils.data.Subset(experiment.train_data, indices)
    batch = next(iter(experiment.loader(subset, True)))
    _, inputs, targets, extra = experiment.unpack(batch)
    model.train()
    output = model(**inputs, **extra)
    losses = criterion(output, targets)
    loss = weighted_loss(criterion, losses)
    if not torch.isfinite(loss):
        raise RuntimeError(losses)
    loss.backward()
    gradients = {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}
    if not gradients or not all(torch.isfinite(value).all() for value in gradients.values()):
        raise RuntimeError("Missing or nonfinite gradients")
    for route in model.routes:
        marker = "private" if route == "sepffn" else "instructions" if route == "instruction" else "aux_refs"
        if route == "shared" and experiment.backbone == "eatr":
            marker = "aux_slots.shared"
        if not any(marker in name and value.abs().sum() > 0 for name, value in gradients.items()):
            raise RuntimeError(f"No auxiliary gradient found for {route}; marker={marker}")
    model.eval()
    with torch.no_grad():
        inference = model(**inputs, **({} if experiment.backbone != "cg_detr" else {}))
    if "o2m_outputs" in inference:
        raise RuntimeError("Auxiliary predictions leaked into inference")
    required = {"pred_logits", "pred_spans"}
    if not required.issubset(inference):
        raise RuntimeError(f"Missing main outputs: {required - set(inference)}")
    report = {
        "status": "PASSED",
        "backbone": experiment.backbone,
        "variant": experiment.variant,
        "seed": SEED,
        "microbatch": experiment.microbatch,
        "gradient_accumulation": experiment.accumulation,
        "measured_batch_size": len(batch[0]),
        "execution_plan_revision": experiment.execution_profile.get('revision'),
        "num_workers": NUM_WORKERS,
        "worker_collate": "cpu_then_main_process_cuda" if experiment.backbone == "cg_detr" else "native_cpu_collate",
        "positive_qid": experiment.train_data.data[positive]["qid"],
        "negative_qid": experiment.train_data.data[negative]["qid"],
        "loss": float(loss.detach()),
        "losses": {key: float(value.detach()) for key, value in losses.items() if torch.is_tensor(value)},
        "match_stats": criterion.last_match_stats,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
        "inference_keys": sorted(inference),
    }
    dump(experiment.out / "smoke.json", report)
    print("SMOKE_PASSED", json.dumps(report), flush=True)


def evaluate(experiment: Experiment, split: str):
    data = experiment.val_data if split == "val" else (
        experiment.module.SoccerDataset(experiment.options, "test")
        if experiment.backbone == "eatr"
        else experiment.module.Soccer("test")
    )
    predictions = experiment.predict(data)
    metrics = evaluate_gmr(predictions, data.data, cls_thresholds=(0.4,), map_num_workers=1, verbose=False)
    # The official evaluator reports percentages. Preserve that canonical output,
    # and add an explicit full-domain 0-1 view for every G-mIoU@k value.
    metrics["G-mIoU_0_1"] = {
        key: float(value) / 100.0 for key, value in metrics["brief"].items() if key.startswith("G-mIoU@")
    }
    metrics["brief_0_1"] = {
        key: (float(value) / 100.0 if key.startswith("G-mIoU@") else value)
        for key, value in metrics["brief"].items()
    }
    jsonl(experiment.out / f"{split}_predictions.jsonl", predictions)
    dump(experiment.out / f"{split}_metrics.json", metrics)
    return metrics


def checkpoint_state(experiment, optimizer, scheduler, epoch, best, best_epoch, stale, generator):
    return {
        "model": experiment.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best": best,
        "best_epoch": best_epoch,
        "stale": stale,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "loader_rng": generator.get_state(),
    }


def train(experiment: Experiment):
    output = experiment.out
    smoke_path = output / "smoke.json"
    if experiment.execution_profile.get('action') == 'restart':
        report = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
        if report.get('status') != 'PASSED' or report.get('measured_batch_size') != experiment.microbatch:
            smoke_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
            smoke(experiment)
            experiment.model.zero_grad(set_to_none=True)
            # A smoke gate must not consume the training run's seeded RNG stream.
            random.setstate(smoke_rng[0])
            np.random.set_state(smoke_rng[1])
            torch.set_rng_state(smoke_rng[2])
            torch.cuda.set_rng_state_all(smoke_rng[3])
    if not smoke_path.exists() or json.loads(smoke_path.read_text()).get("status") != "PASSED":
        raise RuntimeError(f"Smoke gate missing: {smoke_path}")
    model, criterion = experiment.model, experiment.criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=experiment.options.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, experiment.options.lr_drop, gamma=0.1)
    train_loader = experiment.loader(experiment.train_data, True)
    generator = train_loader.generator
    start, best, best_epoch, stale = 0, -float("inf"), 0, 0
    latest = output / "last.ckpt"
    if latest.exists():
        state = torch.load(latest, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best, stale = state["epoch"] + 1, state["best"], state["stale"]
        # Backward-compatible recovery for checkpoints written before best_epoch
        # was recorded explicitly. Validation occurs every EVAL_EVERY epochs.
        last_evaluated_epoch = (start // EVAL_EVERY) * EVAL_EVERY
        best_epoch = state.get("best_epoch", max(0, last_evaluated_epoch - stale * EVAL_EVERY))
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        generator.set_state(state["loader_rng"].cpu())
    configuration = {
        "backbone": experiment.backbone,
        "variant": experiment.variant,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": 5e-5,
        "microbatch": experiment.microbatch,
        "gradient_accumulation": experiment.accumulation,
        "effective_batch": experiment.microbatch * experiment.accumulation,
        "execution_plan_revision": experiment.execution_profile.get('revision'),
        "num_workers": NUM_WORKERS,
        "persistent_workers": True,
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "eval_every_epochs": EVAL_EVERY,
        "early_stop_patience_epochs": EARLY_STOP_PATIENCE_EPOCHS,
        "early_stop_rule": "stop when current_epoch - best_epoch >= 20",
        "main_queries": 10,
        "auxiliary_queries": 24,
        "o2m_k": 2,
        "auxiliary_loss_weight_sum": 0.3,
        "instruction_tokens": 4,
        "adaptive_iou_threshold": 0.25,
        "existence_loss_weight": 1.0,
        "checkpoint_selection": "validation positive-query mAP",
        "gmiou_reporting": "official percentage preserved plus explicit G-mIoU_0_1 full-domain values",
    }
    dump(output / "experiment_config.json", configuration)
    dump(output / "status.json", {"status": "TRAINING", "start_epoch": start + 1, "best_val_map": best})
    for epoch in range(start, EPOCHS):
        begun = time.time()
        model.train()
        if hasattr(criterion.native, "epoch"):
            criterion.native.epoch = epoch
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        match_sum = match_gt = 0
        for step, batch in enumerate(train_loader):
            _, inputs, targets, extra = experiment.unpack(batch)
            output_dict = model(**inputs, **extra)
            losses = criterion(output_dict, targets)
            loss = weighted_loss(criterion, losses)
            if not torch.isfinite(loss):
                raise RuntimeError({key: float(value) for key, value in losses.items() if torch.is_tensor(value)})
            (loss / experiment.accumulation).backward()
            total += float(loss.detach())
            match_sum += criterion.last_match_stats.get("matches", 0)
            match_gt += criterion.last_match_stats.get("gt", 0)
            if (step + 1) % experiment.accumulation == 0 or step + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), experiment.options.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(json.dumps({"backbone": experiment.backbone, "variant": experiment.variant, "epoch": epoch + 1, "step": step, "loss": float(loss.detach())}), flush=True)
        scheduler.step()
        evaluated = (epoch + 1) % EVAL_EVERY == 0
        validation = None
        improved = False
        if evaluated:
            validation = evaluate(experiment, "val")
            score = validation["brief"]["mAP"]
            improved = score > best
            if improved:
                best, best_epoch, stale = score, epoch + 1, 0
            else:
                stale += 1
        state = checkpoint_state(experiment, optimizer, scheduler, epoch, best, best_epoch, stale, generator)
        torch.save(state, latest.with_suffix(".tmp"))
        os.replace(latest.with_suffix(".tmp"), latest)
        if improved:
            torch.save(state, output / "best.tmp")
            os.replace(output / "best.tmp", output / "best.ckpt")
            jsonl(output / "best_val_predictions.jsonl", [json.loads(line) for line in (output / "val_predictions.jsonl").read_text().splitlines()])
            dump(output / "best_val_metrics.json", validation)
        record = {
            "epoch": epoch + 1,
            "train_loss": total / max(1, len(train_loader)),
            "matches_per_gt": match_sum / max(1, match_gt),
            "validation": validation["brief"] if validation else None,
            "best_val_map": best,
            "best_epoch": best_epoch,
            "epochs_since_best": epoch + 1 - best_epoch,
            "stale_evaluations": stale,
            "seconds": time.time() - begun,
        }
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        dump(output / "status.json", {"status": "TRAINING", **record})
        print("EPOCH", json.dumps(record), flush=True)
        if evaluated and epoch + 1 - best_epoch >= EARLY_STOP_PATIENCE_EPOCHS:
            print(f"EARLY_STOP epoch={epoch + 1} best_epoch={best_epoch} patience_epochs={EARLY_STOP_PATIENCE_EPOCHS}", flush=True)
            break
    best_path = output / "best.ckpt"
    digest = hashlib.sha256(best_path.read_bytes()).hexdigest()
    dump(output / "frozen_checkpoint.json", {"path": str(best_path), "sha256": digest, "best_val_map": best})
    dump(output / "status.json", {"status": "TRAINED", "best_val_map": best, "checkpoint_sha256": digest})


def test(experiment: Experiment):
    checkpoint = experiment.out / "best.ckpt"
    if not checkpoint.exists():
        raise RuntimeError(f"Missing best checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location="cuda", weights_only=False)
    experiment.model.load_state_dict(state["model"], strict=True)
    metrics = evaluate(experiment, "test")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    dump(experiment.out / "status.json", {"status": "COMPLETED", "best_val_map": state["best"], "checkpoint_sha256": digest, "test": metrics["brief"]})
    print("TEST_COMPLETED", experiment.backbone, experiment.variant, json.dumps(metrics["brief"]), flush=True)
    if FIGURE_REFRESH.is_file():
        result = subprocess.run(["bash", str(FIGURE_REFRESH)], check=False)
        if result.returncode:
            print(
                f"FIGURE_REFRESH_WARNING returncode={result.returncode} script={FIGURE_REFRESH}",
                flush=True,
            )
        else:
            print(f"FIGURE_REFRESH_COMPLETED script={FIGURE_REFRESH}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("eatr", "qd_detr", "cg_detr"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--mode", choices=("smoke", "train", "test"), required=True)
    parser.add_argument("--test-after-training", action="store_true", help="Keep the job lock across training and frozen-best Test")
    args = parser.parse_args()
    if args.test_after_training and args.mode != "train":
        parser.error("--test-after-training requires --mode train")
    assignment_path = HERE / "gpu_assignments.json"
    if assignment_path.is_file():
        assignment = json.loads(assignment_path.read_text()).get("jobs", {}).get(f"{args.backbone}/{args.variant}")
        if assignment is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(assignment)
            print(f"GPU_ASSIGNMENT_OVERRIDE {args.backbone}/{args.variant} physical_gpu={assignment}", flush=True)
    with exclusive_job(args.backbone, args.variant):
        execute(args)


def execute(args):
    run_dir = HERE / "runs" / args.backbone / args.variant
    stop_path = run_dir / "stop_training.json"
    if args.mode == "train" and stop_path.is_file():
        request = json.loads(stop_path.read_text())
        checkpoint = run_dir / "best.ckpt"
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != request["checkpoint_sha256"]:
            raise RuntimeError("The user-frozen best checkpoint changed after the stop request")
        dump(run_dir / "frozen_checkpoint.json", {"path": str(checkpoint), "sha256": digest, "best_val_map": request["best_val_map"], "best_epoch": request["best_epoch"], "reason": request["reason"]})
        dump(run_dir / "status.json", {"status": "TRAINED", "best_val_map": request["best_val_map"], "best_epoch": request["best_epoch"], "checkpoint_sha256": digest, "stopped_by_user": True, "last_completed_epoch": request["last_completed_epoch"]})
        print(f"USER_STOPPED_TRAINING_SKIP_TO_TEST {args.backbone}/{args.variant}", flush=True)
        return
    external = HERE / "external_dispatch" / f"{args.backbone}_{args.variant}.json"
    current = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
    if external.exists() and current.get("status") == "COMPLETED" and (run_dir / "test_metrics.json").exists():
        print(f"SKIPPED_EXTERNALLY_COMPLETED {args.backbone}/{args.variant}", flush=True)
        return
    profile = job_profile(args.backbone, args.variant)
    if profile.get('action') == 'reuse_completed_baseline' and args.mode in ('smoke', 'train'):
        marker = HERE / 'runs' / args.backbone / args.variant / 'reused_baseline.json'
        if not marker.is_file():
            raise RuntimeError(f'Reused baseline has not been installed: {marker}')
        print('REUSED_COMPLETED_BASELINE', marker, flush=True)
        return
    if args.mode in ('smoke', 'train'):
        prepare_restart(args.backbone, args.variant)
    experiment = Experiment(args.backbone, args.variant).build()
    try:
        if args.test_after_training and external.exists() and current.get("status") == "TRAINED":
            test(experiment)
        else:
            {"smoke": smoke, "train": train, "test": test}[args.mode](experiment)
            if args.test_after_training:
                test(experiment)
    finally:
        del experiment
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
