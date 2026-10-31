#!/usr/bin/env python3
"""Val-selected joint and frozen GEQ on baseline/shared/multi checkpoints."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import fcntl
import hashlib
import json
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
from torch.utils.data import Subset
from study_paths import BACKBONES, VARIANTS, CORE, KIT, MAX_EPOCHS, MODES, OUTPUT, PROJECT, SEED, source_dir, original_prediction, sha, dump
from validation_selection import ValidationSelector
from native_adapter import Adapter
from geq_model import RouteGEQ


def rows(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]


def evaluator(backbone):
    if backbone == "moment_detr":
        sys.path.insert(0, str(PROJECT / "generalized-moment-retrieval/eval"))
        from eval_main import evaluate_gmr
    else:
        sys.path.insert(0, "/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared")
        from official_eval.eval_main import evaluate_gmr
    return evaluate_gmr


def reports_for(backbone, predictions, gt):
    assert len(predictions) == len(gt) == len({p["qid"] for p in predictions})
    assert {p["qid"] for p in predictions} == {g["qid"] for g in gt}
    evaluate = evaluator(backbone)
    return {str(t): evaluate(predictions, gt, cls_thresholds=(0.4, 0.5, 0.6),
            gmiou_cls_threshold=t, map_num_workers=0, verbose=False) for t in (0.4, 0.5, 0.6)}

def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def state_hash(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_worker(worker):
    value = torch.initial_seed() % 2**32
    np.random.seed(value)
    random.seed(value)


def construct(backbone, variant, mode):
    seed_all()
    adapter = Adapter(backbone, variant)
    source = source_dir(backbone, variant) / "best.ckpt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    adapter.base.load_state_dict(checkpoint["model"], strict=True)
    adapter.manifest = dict(checkpoint=str(source), sha256=sha(source), epoch_raw=checkpoint.get("epoch"))
    seed_all()
    model = RouteGEQ(adapter.base, backbone, mode).cuda()
    return adapter, model


def training_mode(adapter, model, mode):
    model.train()
    adapter.criterion.train()
    if mode == "frozen":
        model.base.eval()


def objective(adapter, model, mode, metadata, inputs, targets, epoch, extra=None):
    output = model(**inputs, **(extra or {} if mode == "full" else {}))
    z = output["pred_exist_logits"].reshape(-1)
    labels = z.new_tensor([bool(item["relevant_windows"]) for item in metadata])
    geq = F.binary_cross_entropy_with_logits(z, labels)
    parts = dict(geq=geq)
    total = geq
    if mode == "full":
        native = adapter.native_loss(output["native_output"], targets, epoch)
        parts["native"] = native
        total = total + native
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Nonfinite training loss")
    return total, parts


class NativePresence(torch.nn.Module):
    """Use the source checkpoint's original existence head with identical export."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        output = self.model(**inputs)
        native = output["native_output"]
        logit = native.get("pred_exist_logits", native.get("pred_exist_logit"))
        if logit is None:
            raise RuntimeError("Original GMR head did not produce logits")
        logit = logit.reshape(-1)
        return dict(output, pred_exist_logits=logit, pred_exist_score=logit.sigmoid())


def measure(adapter, model, data):
    saved = rng_state()
    try:
        predictions = adapter.predict(model, data)
        if len(predictions) != len(data.data) or {p["qid"] for p in predictions} != {m["qid"] for m in data.data}:
            raise RuntimeError("Evaluation dropped or duplicated queries")
        evaluate_gmr = evaluator(adapter.name)
        reports = {str(threshold): evaluate_gmr(predictions, data.data,
            cls_thresholds=(0.4, 0.5, 0.6), gmiou_cls_threshold=threshold,
            map_num_workers=0, verbose=False) for threshold in (0.4, 0.5, 0.6)}
        return predictions, reports
    finally:
        restore_rng(saved)


def write_jsonl(path, records):
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))
    temporary.replace(path)


def preflight(adapter, model, mode, data, directory):
    original = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    base_before = state_hash(model.base)
    group = [next(i for i, m in enumerate(data.data) if len(m["relevant_windows"]) == n) for n in (0, 1, 2)]
    optimizer = optimizer_for(model, mode)
    cases = []
    for indices in ([group[0]], group, list(range(adapter.bs))):
        training_mode(adapter, model, mode)
        optimizer.zero_grad(set_to_none=True)
        metadata, inputs, targets, extra = adapter.unpack(next(iter(adapter.loader(Subset(data, indices)))))
        loss, parts = objective(adapter, model, mode, metadata, inputs, targets, 1, extra)
        loss.backward()
        gradients = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients.values()):
            raise RuntimeError("Missing or nonfinite gradients")
        if not any(n.startswith("geq.") and bool(g.abs().max() > 0) for n, g in gradients.items()):
            raise RuntimeError("GEQ received no gradient")
        if mode == "full":
            if not any(n.startswith("base.base.transformer.") and bool(g.abs().max() > 0) for n, g in gradients.items()):
                raise RuntimeError("Full fine tuning has no native Transformer gradient")
        elif any(n.startswith("base.") for n in gradients):
            raise RuntimeError("Frozen native model received a gradient")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.1, error_if_nonfinite=True)
        optimizer.step()
        cases.append(dict(size=len(indices), loss=float(loss.detach()), parts={k: float(v.detach()) for k, v in parts.items()}))
    changed = state_hash(model.base) != base_before
    if changed != (mode == "full"):
        raise RuntimeError("Native tensor update does not match requested mode")
    model.load_state_dict(original, strict=True)
    model.zero_grad(set_to_none=True)
    if state_hash(model.base) != base_before:
        raise RuntimeError("Smoke weights leaked into training")
    dump(directory / "PREFLIGHT.json", dict(passed=True, mode=mode, cases=cases,
        native_updated_in_smoke=changed, initial_weights_restored=True,
        feature_order=adapter.opt.v_feat_dirs, source_checkpoint=adapter.manifest))
    print("PREFLIGHT_PASSED", adapter.name, mode, flush=True)


def optimizer_for(model, mode):
    groups = [dict(params=list(model.geq.parameters()), lr=3e-5, name="geq")]
    if mode == "full":
        groups.insert(0, dict(params=list(model.base.parameters()), lr=1e-5, name="pretrained_gmr"))
    return torch.optim.AdamW(groups, weight_decay=1e-4, foreach=False)


def run(args, directory):
    torch.set_num_threads(2)
    adapter, model = construct(args.backbone, args.variant, args.mode)
    initial_state = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    base_initial, geq_initial = state_hash(model.base), state_hash(model.geq)
    train, val = adapter.dataset("train"), adapter.dataset("val")
    if (len(train), len(val)) != (4138, 465):
        raise RuntimeError("Incorrect Standard Train/Val split")
    config = dict(backbone=args.backbone, variant=args.variant, mode=args.mode, seed=SEED, batch_size=adapter.bs,
        max_epochs=MAX_EPOCHS, num_workers=2, lr_base=1e-5 if args.mode == "full" else 0,
        lr_geq=3e-5, wd=1e-4, gradient_clip=0.1, amp=False,
        loss="native GMR loss + GEQ BCE" if args.mode == "full" else "GEQ BCE only",
        initialization=adapter.manifest, initial_base_sha256=base_initial, initial_geq_sha256=geq_initial,
        actual_video_feature_order=adapter.opt.v_feat_dirs,
        preserve_original_source_feature_order=True,
        metric="Val mean G-mIoU@1 at thresholds 0.4/0.5/0.6", gmiou_unit="0–1",
        early_stop="strictly more than 5 consecutive non-improving epochs (6)",
        test_used_for_selection=False, all_three_above_baseline_recorded=True,
        code_sha256={p.name: sha(p) for p in KIT.glob("*.py")},
        native_source_sha256={str(p): sha(p) for p in (CORE / "run.py", CORE / "multiroute.py", PROJECT / "gmr_multiroute_kit/run_train.py", PROJECT / "gmr_multiroute_kit/multiroute.py")})
    if (directory / "CONFIG.json").is_file():
        old = json.loads((directory / "CONFIG.json").read_text())
        for key in ("mode", "seed", "batch_size", "initial_base_sha256", "initial_geq_sha256"):
            if old[key] != config[key]:
                raise RuntimeError(f"Incompatible resume: {key}")
    if not (directory / "last.ckpt").is_file():
        dump(directory / "CONFIG.json", config)
    pair_path = OUTPUT / args.backbone / args.variant / ("frozen" if args.mode == "full" else "full") / "CONFIG.json"
    if pair_path.is_file():
        paired = json.loads(pair_path.read_text())
        for key in ("initial_base_sha256", "initial_geq_sha256", "actual_video_feature_order", "native_source_sha256"):
            if paired[key] != config[key]:
                raise RuntimeError(f"Full/frozen pair differs: {key}")
    preflight(adapter, model, args.mode, train, directory)
    if args.preflight_only:
        return
    baseline_predictions, baseline_reports = measure(adapter, NativePresence(model), val)
    dump(directory / "baseline_val_metrics.json", baseline_reports)
    write_jsonl(directory / "baseline_val_predictions.jsonl", baseline_predictions)
    selector = ValidationSelector(baseline_reports)
    optimizer = optimizer_for(model, args.mode)
    generator = torch.Generator().manual_seed(SEED)
    start_epoch = 0
    seed_all()
    if (directory / "last.ckpt").is_file():
        saved = torch.load(directory / "last.ckpt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for group in optimizer.state.values():
            for key, value in group.items():
                if torch.is_tensor(value) and key != "step":
                    group[key] = value.cuda()
        selector.load_state_dict(saved["selector"])
        generator.set_state(saved["generator"])
        restore_rng(saved["rng"])
        start_epoch = saved["epoch"]
    loader = adapter.loader(train, True, generator)
    for epoch in range(start_epoch + 1, MAX_EPOCHS + 1):
        if selector.stale_epochs > selector.patience:
            break
        begun = time.monotonic()
        training_mode(adapter, model, args.mode)
        total, count, parts_sum = 0.0, 0, {}
        for step, batch in enumerate(loader, 1):
            metadata, inputs, targets, extra = adapter.unpack(batch)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = objective(adapter, model, args.mode, metadata, inputs, targets, epoch, extra)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 0.1, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(metadata)
            count += len(metadata)
            for key, value in parts.items():
                parts_sum[key] = parts_sum.get(key, 0.0) + float(value.detach()) * len(metadata)
            if step == 1 or step % 50 == 0:
                dump(directory / "STATUS.json", dict(status="TRAINING", epoch=epoch, step=step, steps=len(loader), pid=os.getpid()))
                print("TRAIN", args.backbone, args.mode, epoch, step, float(loss.detach()), flush=True)
        if count != 4138:
            raise RuntimeError("Training dropped queries")
        if args.mode == "frozen" and state_hash(model.base) != base_initial:
            raise RuntimeError("Frozen native tensors changed")
        predictions, reports = measure(adapter, model, val)
        selection = selector.update(epoch, reports)
        dump(directory / "val_metrics.json", reports)
        write_jsonl(directory / "val_predictions.jsonl", predictions)
        record = dict(epoch=epoch, loss=total / count, losses={k: v / count for k, v in parts_sum.items()},
            seconds=time.monotonic() - begun, selection=selection,
            val_mAP=reports["0.4"]["brief"]["mAP"], val_AUROC=reports["0.4"]["brief"]["AUROC"],
            val_gmiou_3={t: reports[t]["brief"]["G-mIoU@3"] / 100 for t in ("0.4", "0.5", "0.6")})
        with (directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if selection["save_best"]:
            temporary = directory / "best.tmp"
            torch.save(dict(model=model.state_dict(), epoch=epoch, selection=selection, config=config), temporary)
            temporary.replace(directory / "best.ckpt")
        temporary = directory / "last.tmp"
        torch.save(dict(model=model.state_dict(), epoch=epoch, optimizer=optimizer.state_dict(),
            selector=selector.state_dict(), rng=rng_state(), generator=generator.get_state()), temporary)
        temporary.replace(directory / "last.ckpt")
        dump(directory / "STATUS.json", dict(status="TRAINING", **record))
        print("EPOCH_DONE", json.dumps(record), flush=True)
        if selection["stop"]:
            break
    del loader
    best = torch.load(directory / "best.ckpt", map_location="cpu", weights_only=False)
    # Test labels are used only after Val-based checkpoint selection is final.
    test = adapter.dataset("test")
    model.load_state_dict(initial_state, strict=True)
    baseline_test_predictions = rows(original_prediction(args.backbone, args.variant, "test"))
    baseline_test_metrics = reports_for(adapter.name, baseline_test_predictions, test.data)
    write_jsonl(directory / "baseline_test_predictions.jsonl", baseline_test_predictions)
    dump(directory / "baseline_test_metrics.json", baseline_test_metrics)
    model.load_state_dict(best["model"], strict=True)
    test_predictions, test_metrics = measure(adapter, model, test)
    if args.mode == "frozen":
        scores = {p["qid"]: p["pred_exist_score"] for p in test_predictions}
        test_predictions = copy.deepcopy(baseline_test_predictions)
        for row in test_predictions:
            row["pred_exist_score"] = scores[row["qid"]]
        test_metrics = reports_for(adapter.name, test_predictions, test.data)
    write_jsonl(directory / "test_predictions.jsonl", test_predictions)
    dump(directory / "test_metrics.json", test_metrics)
    _, selected_val = measure(adapter, model, val)
    dump(directory / "best_val_metrics.json", selected_val)
    if args.mode == "frozen":
        assert state_hash(model.base) == base_initial
        assert all(a["pred_relevant_windows"] == b["pred_relevant_windows"]
                   for a, b in zip(baseline_test_predictions, test_predictions))
        assert test_metrics["0.4"]["brief"]["mAP"] == baseline_test_metrics["0.4"]["brief"]["mAP"]
    assert state_hash(model.geq) != geq_initial
    dump(directory / "COMPLETE.json", dict(status="COMPLETED", backbone=args.backbone, variant=args.variant, mode=args.mode,
        best_epoch=best["epoch"], selection=best["selection"], selection_split="val", test_used_for_selection=False,
        native_changed=state_hash(model.base) != base_initial, geq_updated=True,
        checkpoint_sha256=sha(directory / "best.ckpt"), source_checkpoint_sha256=adapter.manifest["sha256"],
        test_samples=len(test_predictions), val_samples=len(val), source_matches_original_route=True, protocol_matches_baseline_epoch_study=True))
    dump(directory / "STATUS.json", dict(status="COMPLETED", best_epoch=best["epoch"]))
    print("COMPLETE", args.backbone, args.mode, flush=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    directory = OUTPUT / args.backbone / args.variant / args.mode
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "COMPLETE.json").is_file():
            print("ALREADY_COMPLETE", directory, flush=True)
            return
        try:
            run(args, directory)
        except Exception:
            dump(directory / "FAILURE.json", dict(pid=os.getpid(), traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
