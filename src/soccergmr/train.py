"""Two-stage Standard-only training and GEQ fine-tuning. Test labels are never opened here."""
import json
import math
import signal
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from . import checkpoint as ck
from .backbones.registry import build, get_spec, forward
from .data import SoccerDataset, collate, move
from .evaluation import evaluate
from .geq import GEQFull
from .io import dump_json, dump_jsonl, fingerprint, sha256
from .predict import predict
from .selection import Selector, score


def extract_top1_slots(outputs, targets):
    pos_mask = targets.get("exist_label", None)
    if pos_mask is None:
        return None
    pos_indices = torch.where(pos_mask > 0)[0]
    if len(pos_indices) == 0:
        return None

    if "pred_quality" in outputs:
        foreground = outputs["pred_quality"].squeeze(-1)
    elif "pred_logits" in outputs:
        foreground = outputs["pred_logits"].softmax(-1)[..., 0]
    else:
        return None

    if "iou_scores" in outputs:
        foreground = foreground * outputs["iou_scores"].squeeze(-1).sigmoid()

    pos_foreground = foreground[pos_indices]
    top1_slots = pos_foreground.argmax(dim=-1)
    return top1_slots.detach()


def train(config, output, stage, device='cuda:0', baseline=None, resume=False):
    directory = Path(output).resolve()
    if stage not in ('gmr', 'geq_full', 'gmr_geq'):
        raise ValueError(f"Unknown stage: {stage}")
    with ck.run_lock(directory):
        return _train(config, directory, stage, device, baseline, resume)


def _train(config, directory, stage, device, baseline, resume):
    p = config['protocol']
    spec = get_spec(config['backbone'])
    training = SoccerDataset(config, 'train', spec)
    validation = SoccerDataset(config, 'val', spec)
    if not len(training) or not len(validation):
        raise ValueError('Train and Val must be nonempty')
    if {Path(r['vid']).stem for r in training.data} & {Path(r['vid']).stem for r in validation.data}:
        raise ValueError('Train/Val video leakage')
    source = ck.source_identity()
    assets = ck.asset_identity([training, validation])
    identity = fingerprint(dict(config=config['identity'], source=source, train_val_assets=assets, stage=stage))

    if (directory / 'complete.json').exists():
        completed = json.loads((directory / 'complete.json').read_text())
        if completed.get('status') == 'complete' and (directory / 'best.pt').exists():
            if sha256(directory / 'best.pt') == completed.get('best_sha256'):
                print(f"Run in {directory} is already complete.")
                return completed

    if (directory / 'last.pt').exists() and not resume:
        raise ValueError(f"Existing last.pt in {directory}: pass --resume to continue, or use a new directory")
    if resume and not (directory / 'last.pt').is_file():
        raise FileNotFoundError('Resume requires legal last.pt')

    ck.seed_all(p['seed'])
    model, criterion = build(config, device)
    baseline_sha = None
    native_epoch_offset = 0

    if stage in ('geq_full', 'gmr_geq'):
        if baseline is None:
            raise ValueError(f"{stage} requires a finalized baseline best.pt")
        baseline = Path(baseline).resolve()
        if not baseline.is_file():
            raise FileNotFoundError(f"Baseline checkpoint not found: {baseline}")
        baseline_sha = sha256(baseline)
        initial = ck.load(baseline, stage='gmr')
        native_epoch_offset = initial.get('epoch', 0)
        model.load_state_dict(initial['model'], strict=True)
        model = GEQFull(model, spec).to(device)
        groups = [
            dict(params=model.base.parameters(), lr=p.get('geq_lr_native', 1e-5)),
            dict(params=model.geq.parameters(), lr=p.get('geq_lr_head', 3e-5))
        ]
    else:
        groups = [dict(params=model.parameters(), lr=p['learning_rate'])]

    optimizer = torch.optim.AdamW(groups, weight_decay=p['weight_decay'])
    generator = torch.Generator().manual_seed(p['seed'])
    loader = DataLoader(
        training, batch_size=p['batch_size'], shuffle=True, generator=generator,
        collate_fn=collate, num_workers=p['workers']
    )
    maximum = p.get('geq_max_epochs', 200) if stage in ('geq_full', 'gmr_geq') else p['max_epochs']

    if stage == 'gmr_geq':
        patience = 20
        selector = Selector('Val Orc-gap (%)', patience=patience, mode='min')
    elif stage == 'gmr':
        patience = p['patience']
        selector = Selector('Val mAP', patience=patience, mode='max')
    else:
        patience = p.get('geq_patience', 10)
        selector = Selector('Val mean G-mIoU@1 (.4/.5/.6)', patience=patience, mode='max')

    epoch = 0
    history = []

    def payload():
        return dict(
            identity=identity, stage=stage, config=config, source=source, train_val_assets=assets,
            model=model.state_dict(), optimizer=optimizer.state_dict(), selector=selector.state_dict(),
            epoch=epoch, rng=ck.rng_state(generator), history=history, baseline_sha256=baseline_sha
        )

    if resume:
        saved = ck.load(directory / 'last.pt', stage=stage)
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        selector = Selector.from_state(saved['selector'])
        epoch = saved['epoch']
        history = saved['history']
        ck.restore_rng(saved['rng'], generator)
        dump_jsonl(directory / 'history.jsonl', history)
    else:
        dump_json(directory / 'config.json', config)
        ck.save(directory / 'last.pt', payload())

    stop_requested = []

    def request_stop(signum, frame):
        stop_requested.append(signum)

    previous = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}

    try:
        while epoch < maximum and selector.stale < selector.patience:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            batches = len(loader)
            for batch_index, (metas, inputs, targets) in enumerate(loader):
                if stop_requested:
                    return dict(status='interrupted', resume_from=str(directory / 'last.pt'), completed_epoch=epoch)
                inputs, targets = move(inputs, device), move(targets, device)
                outputs = forward(model, config['backbone'], inputs, targets, metas, epoch=native_epoch_offset + epoch)
                loss, _ = criterion(outputs.get('native_output', outputs), targets, epoch)
                for module in model.modules():
                    if hasattr(module, "update_top1_ema"):
                        top1 = extract_top1_slots(outputs.get('native_output', outputs), targets)
                        module.update_top1_ema(top1, epoch)
                if stage in ('geq_full', 'gmr_geq'):
                    loss = loss + F.binary_cross_entropy_with_logits(outputs['geq_logit'], targets['exist_label'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                window_start = (batch_index // p['accumulation']) * p['accumulation']
                window_size = min(p['accumulation'], batches - window_start)
                (loss / window_size).backward()
                total_loss += float(loss.detach())
                if (batch_index + 1) % p['accumulation'] == 0 or batch_index + 1 == batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), p['gradient_clip'], error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            epoch += 1
            predictions = predict(model, validation, config, device)
            metrics = evaluate(predictions, validation.data)
            value = score(stage, metrics)
            decision = selector.update(epoch, value)
            improved = decision['improved']

            refresh_stats = {}
            for module in model.modules():
                if hasattr(module, "actual_refresh_count") and hasattr(module, "top1_ema"):
                    refresh_stats = {
                        "actual_refresh_count": int(module.actual_refresh_count.item()),
                        "opportunity_count": int(module.opportunity_count.item()),
                        "top1_ema": [round(float(x), 4) for x in module.top1_ema.cpu().tolist()],
                    }
                    break

            history.append(dict(
                epoch=epoch, loss=total_loss / batches, selection_score=value,
                improved=improved, stale=selector.stale, metrics=metrics,
                refresh_stats=refresh_stats
            ))
            state = payload()
            if improved:
                ck.save(directory / 'best.pt', state)
                dump_json(directory / 'best_val_metrics.json', metrics)
                dump_jsonl(directory / 'best_val_predictions.jsonl', predictions)
            ck.save(directory / 'last.pt', state)
            dump_jsonl(directory / 'history.jsonl', history)

            log_dict = dict(
                stage=stage, epoch=epoch, score=round(value, 2),
                best=round(selector.best, 2), stale=selector.stale,
                loss=round(total_loss / batches, 4)
            )
            if 'orc_gap' in metrics:
                log_dict["orc_gap"] = metrics['orc_gap']['gap_pct']
                log_dict["auc"] = metrics['orc_gap']['g1_auc_pct']
                log_dict["oracle"] = metrics['orc_gap']['oracle_g1_pct']
            if refresh_stats and refresh_stats.get("opportunity_count", 0) > 0:
                log_dict["refresh_rate"] = round(refresh_stats["actual_refresh_count"] / max(1, refresh_stats["opportunity_count"]), 4)
            print(json.dumps(log_dict), flush=True)

            if stop_requested:
                return dict(status='interrupted', resume_from=str(directory / 'last.pt'), completed_epoch=epoch)

        completed = dict(
            status='complete', stage=stage, identity=identity, completed_epoch=epoch,
            best_epoch=selector.best_epoch, best_score=selector.best, metric=selector.metric,
            reason='patience' if selector.stale >= selector.patience else 'max_epochs',
            best_sha256=sha256(directory / 'best.pt'), source=source, train_val_assets=assets,
            baseline_sha256=baseline_sha, test_used_for_selection=False
        )
        dump_json(directory / 'complete.json', completed)
        return completed
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
