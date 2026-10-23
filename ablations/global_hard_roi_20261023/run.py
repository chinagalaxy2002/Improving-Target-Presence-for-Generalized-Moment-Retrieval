#!/usr/bin/env python3
"""Fixed epoch-four training; evaluation is performed only after training."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import argparse
import hashlib
import importlib.util
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from components import build, SETTINGS
from official_eval.eval_main import evaluate_gmr
from postprocessing import export

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('boundary_reference', ROOT / 'code/model.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def seed():
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)


def group(name):
    if 'native_exist_head' in name:
        return 'frozen_gmr'
    if 'sal_proj' in name:
        return 'disabled_saliency'
    if 'extractor.' in name:
        return 'hard_roi_extractor'
    if 'refiner.' in name:
        return 'boundary_refiner'
    if any(k in name for k in ['event_query.', 'global_query.', 'global_event_query.',
                               '.exist_head.token', '.exist_head.text_read.',
                               '.exist_head.video_reads.', '.exist_head.global_head.']):
        return 'global_event_query'
    return 'frozen_gmr_or_inactive'


def loss_for(out, metas, job):
    logits = out['pred_exist_logits'].reshape(-1)
    labels = logits.new_tensor([bool(x['relevant_windows']) for x in metas])
    presence = F.binary_cross_entropy_with_logits(logits, labels)
    boundary = reference.compute_boundary_refine_loss(out['refined_spans'], metas, logits.device, job)
    # A frozen native presence head contributes no gradients in hard-only.
    return presence + job['refine_weight'] * boundary


def controls(extractor, refiner, device):
    extractor.eval()
    refiner.eval()
    toy = torch.randn(1, 10, 256, device=device)
    valid = torch.ones(1, 10, dtype=torch.bool, device=device)
    spans = torch.tensor([[[4., 8., .5]]], device=device)
    outside = toy.clone()
    outside[:, 4:] += torch.arange(256, device=device) * .1
    text = torch.randn(1, 3, 256, device=device)
    mask = torch.ones(1, 3, dtype=torch.bool, device=device)
    with torch.no_grad():
        tokens = extractor(toy, valid, spans)[0]
        assert torch.equal(tokens, extractor(outside, valid, spans)[0])
        before = refiner(tokens, text, mask, spans, torch.zeros(1, 10, device=device))[0]
        after = refiner(tokens, text, mask, spans, torch.randn(1, 10, device=device) * 1000)[0]
        assert torch.equal(before, after), 'Disabled saliency affects refinement'
        valid[:, -1] = False
        padded = toy.clone()
        padded[:, -1] += torch.arange(256, device=device)
        assert torch.equal(extractor(toy, valid, spans)[0], extractor(padded, valid, spans)[0])
    return dict(hard_roi_outside_invariant=True, padding_invariant=True, saliency_perturbation_invariant=True)


@torch.no_grad()
def predict(backend, model, data, bs, device, flash):
    model.eval()
    if flash:
        from prediction import predictions
        return predictions(model, data, backend.opt)[1]
    result = []
    for start in range(0, len(data), bs):
        metas, inputs = backend.collate([data[i] for i in range(start, min(start + bs, len(data)))], device)
        result.extend(export(model(**inputs), metas))
    return result


def evaluate(pred, gt, output, split):
    assert len(pred) == len(gt) and len({p['qid'] for p in pred}) == len(gt)
    assert {p['qid'] for p in pred} == {g['qid'] for g in gt}
    (output / f'{split}_predictions.jsonl').write_text(''.join(json.dumps(p) + '\n' for p in pred))
    reports = {}
    for tau in [.4, .5, .6]:
        report = evaluate_gmr(pred, gt, cls_thresholds=(tau,), gmiou_cls_threshold=tau, map_num_workers=0, verbose=False)
        negatives = {g['qid'] for g in gt if not g['relevant_windows']}
        report['brief']['N-acc'] = 100 * sum(p['pred_exist_score'] <= tau for p in pred if p['qid'] in negatives) / len(negatives)
        reports[str(tau)] = report
    dump(output / f'{split}_metrics.json', reports)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True)
    parser.add_argument('--setting', choices=list(SETTINGS), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    seed()
    directory = ROOT / ('preflights' if args.preflight_only else 'runs') / args.backbone / args.setting
    directory.mkdir(parents=True, exist_ok=False)
    job = json.loads((ROOT / 'config.json').read_text())
    job.update(setting=args.setting, use_global_branch=SETTINGS[args.setting][0], use_boundary_refine=True)
    backend, model, init, extractor, refiner = build(args.backbone, job, args.device)
    assert init['verified_gmr_baseline']
    bs = int(init['baseline_config']['bsz'])
    eval_bs = 1 if args.backbone == 'flashvtg' else int(init['baseline_config'].get('eval_bsz', bs))
    initial = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    active = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            active[group(n)] = active.get(group(n), 0) + p.numel()
    assert set(active) == ({'hard_roi_extractor', 'boundary_refiner', 'global_event_query'} if SETTINGS[args.setting][0] else {'hard_roi_extractor', 'boundary_refiner'}), active
    dump(directory / 'CONFIG.json', dict(job=job, initialization=init, batch_size=bs,
         eval_batch_size=eval_bs, trainable_parameters_by_group=active, trainable=sum(active.values()),
         initial_tensor_sha256={n: hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() for n, t in initial.items()},
         source_sha256={str(p.relative_to(ROOT)): sha(p) for p in ROOT.rglob('*.py')}))
    train, val = backend.get_dataset('train'), backend.get_dataset('val')
    assert len(train) == 4138 and len(val) == 465
    indices = [next(i for i, m in enumerate(train.data) if len(m['relevant_windows']) == n) for n in [0, 1, 2]]
    metas, inputs = backend.collate([train[i] for i in indices], args.device)
    model.train()
    out = model(**inputs)
    loss_for(out, metas, job).backward()
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.parameters() if p.requires_grad)
    model.zero_grad(set_to_none=True)
    checks = controls(extractor, refiner, args.device)
    model.train()
    negative_metas, negative_inputs = backend.collate([train[indices[0]]], args.device)
    negative_loss = loss_for(model(**negative_inputs), negative_metas, job)
    assert torch.isfinite(negative_loss)
    if not SETTINGS[args.setting][0]:
        assert not negative_loss.requires_grad, 'Hard-only must have no supervision on all-negative batches'
    dump(directory / 'PREFLIGHT.json', dict(passed=True, mixed_positive_negative=True, frozen_no_grad=True, **checks))
    print('PREFLIGHT_PASSED', args.backbone, args.setting, flush=True)
    if args.preflight_only:
        return
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=job['lr'], weight_decay=job['wd'], foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, job['lr_drop'], gamma=.5)
    seed()
    loader = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=list, num_workers=0, generator=torch.Generator().manual_seed(2026))
    for epoch in range(1, 5):
        started = time.monotonic()
        total = 0.
        no_gradient_batches = 0
        for step, items in enumerate(loader, 1):
            model.train()
            metas, inputs = backend.collate(items, args.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(**inputs)
            loss = loss_for(out, metas, job)
            assert torch.isfinite(loss)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], job['grad_clip'], error_if_nonfinite=True)
                optimizer.step()
            else:
                # No positive GT -> no boundary target, and native presence is
                # frozen. Still visit the batch, without a spurious weight update.
                assert not SETTINGS[args.setting][0] and all(not m['relevant_windows'] for m in metas)
                no_gradient_batches += 1
            total += float(loss.detach())
            if step == 1 or step % 50 == 0:
                progress = dict(backbone=args.backbone, setting=args.setting, epoch=epoch, step=step, total_steps=len(loader), loss=float(loss.detach()))
                dump(directory / 'progress.json', progress)
                print(json.dumps(progress), flush=True)
        scheduler.step()
        with (directory / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(epoch=epoch, mean_loss=total / len(loader), no_gradient_batches=no_gradient_batches, seconds=time.monotonic() - started)) + '\n')
    changed = {}
    for n, p in model.state_dict().items():
        if n not in trainable:
            assert torch.equal(initial[n], p.cpu()), ('Frozen parameter/buffer changed', n)
        elif not torch.equal(initial[n], p.cpu()):
            changed[group(n)] = changed.get(group(n), 0) + p.numel()
    assert set(changed) == set(active), changed
    checkpoint = directory / 'epoch_004.ckpt'
    torch.save(dict(model=model.state_dict(), job=job, initialization=init, epoch=4), checkpoint)
    model.eval()
    # Verify serialization at the exact evaluation batch shape.
    metas, inputs = backend.collate([val[0]], args.device)
    with torch.no_grad():
        expected = model(**inputs)['pred_exist_logits'].clone()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'], strict=True)
        assert torch.equal(expected, model(**inputs)['pred_exist_logits'])
    dump(directory / 'CHECKPOINT.json', dict(sha256=sha(checkpoint), epoch=4, strict_reload_equal=True))
    dump(directory / 'EFFECTIVE_CONFIG.json', dict(passed=True, global_event_query=SETTINGS[args.setting][0], hard_roi=True,
         lora=False, gaussian_roi=False, saliency_6d=False, candidate_presence_fusion=False,
         frozen_parameters_and_buffers_unchanged=True, changed_parameters_by_group=changed,
         trainable_parameters_by_group=active, **controls(extractor, refiner, args.device)))
    for split, data in [('val', val), ('test', backend.get_dataset('test'))]:
        assert len(data) == (465 if split == 'val' else 1036)
        evaluate(predict(backend, model, data, eval_bs, args.device, args.backbone == 'flashvtg'), data.data, directory, split)
    dump(directory / 'COMPLETE.json', dict(status='complete', backbone=args.backbone, setting=args.setting, epochs=4,
         seed=2026, fixed_epoch=4, frozen_unchanged=True, reused=False))
    print('COMPLETE', args.backbone, args.setting, flush=True)


if __name__ == '__main__':
    main()
