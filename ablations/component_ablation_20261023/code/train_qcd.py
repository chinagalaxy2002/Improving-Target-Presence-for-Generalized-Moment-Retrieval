"""S4-Fusion + A3 QCD, unchanged model/loss, with every epoch checkpoint saved."""
import argparse
import json
import time
from pathlib import Path

from environment import ROOT, read_json, write_json, write_rows, seed_all, make_dataset
from api import build_model, job_for
from epoch_test import capture_rng, restore_rng, evaluate_epoch
from evaluation import Evaluation
from prediction import predictions
from model import compute_mcev_loss
from qcd import QCD
from qcd_metrics import worldcup_report
from runtime import selection_score, training_mode
from training.flash_vtg_gmr.dataset import prepare_batch_inputs, start_end_collate
import torch
from torch.utils.data import DataLoader


def save_checkpoint(path, state):
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--cache', type=Path, default=ROOT/'cache/qcd_a3.npz')
    parser.add_argument('--test-from', type=int, help='Explicitly evaluate Test after each epoch starting here; omit for Train/Val only')
    parser.add_argument('--smoke', action='store_true', help='Stop after two optimizer updates')
    args = parser.parse_args()
    if args.epochs < 1 or (args.test_from is not None and not 1 <= args.test_from <= args.epochs):
        parser.error('Epoch counts must be positive and test-from must fall within training')
    directory = (ROOT/'runs'/args.run_name).resolve()
    if not directory.is_relative_to((ROOT/'runs').resolve()) or directory.exists():
        raise ValueError('A fresh directory inside runs/ is required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    job = dict(job_for('S4Fusion'), epochs=args.epochs)
    config = read_json(ROOT/'configs/qcd_a3.json')
    model, opt, checks, _ = build_model('S4Fusion', args.device, initialize=True)
    opt.n_epoch = args.epochs
    frozen = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}
    active = [p for p in model.parameters() if p.requires_grad]
    groups = [{'params': [p for n,p in model.named_parameters() if p.requires_grad and '.lora_' not in n], 'lr': job['lr']},
              {'params': [p for n,p in model.named_parameters() if p.requires_grad and '.lora_' in n], 'lr': job['lora_lr']}]
    optimizer = torch.optim.AdamW(groups, weight_decay=job['wd'], foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, job['lr_drop'], gamma=.5)
    seed_all(2026)
    train, val = make_dataset(opt, 'train'), make_dataset(opt, 'val')
    loader = DataLoader(train, batch_size=opt.bsz, shuffle=True, num_workers=0,
                        collate_fn=start_end_collate, generator=torch.Generator().manual_seed(2026))
    augmentation = QCD(train, config, args.cache)
    seed_all(102026)
    directory.mkdir(parents=True)
    write_json(directory/'INITIALIZATION_CHECK.json', checks)
    write_json(directory/'CONFIG.json', dict(model='S4Fusion', augmentation='A3', job=job, qcd=config,
        epochs=args.epochs, optimizer_steps=args.epochs*len(loader), save_every_epoch=True,
        early_stopping=False, seed=2026, training_rng_seed=102026, test_from=args.test_from,
        Test_used_for_selection=False, qcd_cache=str(args.cache.resolve())))
    test = None
    best = -1.
    stale = 0
    for epoch in range(1, args.epochs+1):
        start = time.time()
        training_mode(model)
        sums = {}
        for step, (metadata, batch) in enumerate(loader, 1):
            features, targets = prepare_batch_inputs(batch, opt.device, non_blocking=False)
            targets.update(label=metadata, fps=torch.full((len(metadata),), 1/opt.clip_length, device=opt.device))
            audit = augmentation.apply(features, metadata)
            if epoch == 1 and step <= 3:
                write_json(directory/f'QCD_batch{step:03d}.json', audit)
            output = model(**features, targets=targets)
            loss, details = compute_mcev_loss(output, metadata, targets, job, model=model, model_inputs=features)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(active, job['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            for key, value in details.items():
                sums[key] = sums.get(key, 0.)+float(value.detach())
            if step == 1 or step % 50 == 0:
                print(f'A3 epoch={epoch} batch={step}/{len(loader)} loss={float(loss.detach()):.5f}', flush=True)
            if args.smoke and step == 2:
                save_checkpoint(directory/'smoke.ckpt', dict(epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=capture_rng(loader, augmentation)))
                write_json(directory/'COMPLETE.json', dict(state='smoke_complete', batches=step, finite_loss=True))
                return
        scheduler.step()
        for name, parameter in model.named_parameters():
            if name in frozen:
                assert torch.equal(parameter.detach().cpu(), frozen[name]), name
        model.eval()
        pre, post = predictions(model, val, opt)
        report = Evaluation(pre, post, val.data).report('val', comprehensive=True)
        report['worldcup_qcd'] = worldcup_report(pre, post, val.data)
        write_json(directory/'validation'/f'epoch{epoch:02d}.json', report)
        score = selection_score(report)
        improved = score > best
        stale = 0 if improved else stale+1
        if improved:
            best = score
            write_json(directory/'selected.json', dict(epoch=epoch, score=score, source='Val only'))
            write_json(directory/'validation_metrics.json', report)
            write_rows(directory/'validation_predictions.jsonl', post)
            write_rows(directory/'validation_predictions_pre_nms.jsonl', pre)
        state = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), best_score=best, stale=stale,
                     rng=capture_rng(loader, augmentation), qcd_counts=dict(augmentation.counts), job=job, qcd_config=config)
        for filename in [f'epoch{epoch:02d}.ckpt', 'latest.ckpt']+(['best.ckpt'] if improved else []):
            save_checkpoint(directory/filename, state)
        test_report = None
        if args.test_from is not None and epoch >= args.test_from:
            if test is None:
                rng = capture_rng()
                try:
                    test = make_dataset(opt, 'test')
                finally:
                    restore_rng(rng)
            test_report = evaluate_epoch(model, test, opt, report, directory/'test'/f'epoch{epoch:02d}', epoch)
        record = dict(epoch=epoch, score=score, best=best, improved=improved, stale=stale,
            seconds=time.time()-start, losses={k:v/len(loader) for k,v in sums.items()},
            qcd=dict(augmentation.counts), worldcup_qcd=report['worldcup_qcd'],
            test_worldcup_qcd=test_report['worldcup_qcd'] if test_report else None)
        with (directory/'trace.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(f'VAL epoch={epoch} score={score:.4f} best={best:.4f}', flush=True)
    write_json(directory/'COMPLETE.json', dict(state='training_complete', epochs=args.epochs,
        optimizer_steps=args.epochs*len(loader), Test_used_for_selection=False,
        saved_epochs=list(range(1,args.epochs+1)), qcd=dict(augmentation.counts)))


if __name__ == '__main__':
    main()
