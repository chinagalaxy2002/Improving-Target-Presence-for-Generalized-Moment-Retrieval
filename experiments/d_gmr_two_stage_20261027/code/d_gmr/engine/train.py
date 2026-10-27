"""One fixed four-epoch end-to-end D-GMR training path; Test is a separate queue."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from gemr.engine import checkpoint
from d_gmr.adapters import build_adapter
from d_gmr.common import BACKBONES, REPO, RUNS, configuration, dump, seed_all
from d_gmr.evaluate import save_evaluation
from d_gmr.losses import objective
from d_gmr.model import DGMRModel


def construct(backbone, device, config):
    adapter = build_adapter(backbone, device, config)
    model = DGMRModel(adapter, config).to(device)
    return adapter, model


def optimizer_for(model, config):
    base = [p for p in model.base.parameters() if p.requires_grad]
    geq = [p for p in model.geq.parameters() if p.requires_grad]
    claimed = {id(value) for value in [*model.base.parameters(), *model.geq.parameters()]}
    added = [parameter for parameter in model.parameters()
             if parameter.requires_grad and id(parameter) not in claimed]
    assert base and added, f"base={len(base)}, added={len(added)}"
    rates = config['learning_rates']
    groups = [dict(params=base, lr=rates['native_gmr'], name='native_gmr')]
    if geq and rates.get('geq', 0.0) > 0:
        groups.append(dict(params=geq, lr=rates['geq'], name='geq'))
    groups.append(dict(params=added, lr=rates['tem_tecr'], name='tem_tecr'))
    optimizer = torch.optim.AdamW(groups, weight_decay=config['weight_decay'], foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    return optimizer, scheduler


def selection(run):
    complete = []
    for path in sorted((run/'val_results').glob('epoch_*/metrics.json')):
        epoch = int(path.parent.name.split('_')[1])
        archive = run/'checkpoints'/f'epoch_{epoch:03d}.pth'
        if archive.exists():
            complete.append((epoch, archive, json.loads(path.read_text())['brief']))
    assert complete
    epoch, archive, metrics = max(
        complete, key=lambda value: (value[2]['mAP'], value[2]['mR@1'], -value[0]))
    record = dict(selection_metric='val_mAP', best_epoch=epoch,
                  checkpoint_path=str(archive.resolve()), selection_split='val',
                  val_mAP=metrics['mAP'], val_mR_at_1=metrics['mR@1'],
                  tie_break='Val mR@1 then earlier epoch',
                  epochs_considered=[value[0] for value in complete])
    dump(run/'best_epoch.json', record)
    return record


def train(args):
    torch.set_num_threads(2)
    seed_all()
    config = configuration()
    run = Path(args.output or RUNS/args.backbone).resolve()
    if args.resume:
        state = checkpoint.load(args.resume)
        assert state['config']['backbone'] == args.backbone
        config = dict(state['config'])
    run.mkdir(parents=True, exist_ok=False)
    for folder in ('checkpoints', 'logs', 'val_results', 'test_results'):
        (run/folder).mkdir()
    config.update(backbone=args.backbone, output_directory=str(run), device=args.device,
                  resume_from=args.resume)
    adapter, model = construct(args.backbone, args.device, config)
    train_data, val_data = adapter.dataset('train'), adapter.dataset('val')
    optimizer, scheduler = optimizer_for(model, config)
    groups = {group['name']: sum(parameter.numel() for parameter in group['params'])
              for group in optimizer.param_groups}
    expected_groups = {'native_gmr', 'tem_tecr'} if config.get('freeze_geq') else {'native_gmr', 'geq', 'tem_tecr'}
    assert set(groups) == expected_groups, f"groups={set(groups)}, expected={expected_groups}"
    config.update(
        initialization=adapter.manifest, optimizer_parameter_groups=groups,
        trainable_parameters={name: parameter.numel() for name, parameter in model.named_parameters()
                              if parameter.requires_grad},
        native_parameter_count=sum(parameter.numel() for parameter in model.base.parameters()),
        geq_parameter_count=sum(parameter.numel() for parameter in model.geq.parameters()),
        added_parameter_count=groups['tem_tecr'],
        total_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        environment=dict(python=platform.python_version(), torch=torch.__version__,
                         cuda=torch.version.cuda, cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                         gpu_name=torch.cuda.get_device_name() if torch.cuda.is_available() else None))
    dump(run/'config.json', config, exclusive=True)
    start_epoch = 1
    if args.resume:
        restored = checkpoint.resume(args.resume, model, optimizer, scheduler)
        start_epoch = restored['epoch']+1
    effective = config['batch_sizes'][args.backbone]
    micro = config['micro_batch_sizes'][args.backbone]
    try:
        if args.resume:
            epoch = restored['epoch']
            saved_rng = checkpoint.rng_state()
            metrics, _, _, _ = save_evaluation(
                model, val_data, 'val', run/'val_results'/f'epoch_{epoch:03d}',
                1 if args.backbone == 'flashvtg' else effective,
                config['presence_threshold'], args.resume)
            if restored['validation_metrics'] is not None:
                checkpoint.restore_rng(saved_rng)
            record = checkpoint.save(run/'checkpoints'/f'epoch_{epoch:03d}.pth', model,
                                     optimizer, scheduler, epoch, config, metrics)
            dump(run/'checkpoints'/f'epoch_{epoch:03d}.json', record, exclusive=True)
            selection(run)
        for epoch in range(start_epoch, config['epochs']+1):
            begun = time.monotonic()
            model.train()
            adapter.criterion.train()
            loader = DataLoader(
                train_data, batch_size=effective, shuffle=True, collate_fn=list, num_workers=0,
                generator=torch.Generator().manual_seed(config['seed']+epoch))
            sums, count, diagnostics = {}, 0, {}
            for step, items in enumerate(loader, 1):
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, len(items), micro):
                    block = items[start:start+micro]
                    metas, inputs, targets = adapter.collate(block)
                    output = model(inputs)
                    loss, parts, contributions, info, native_details = objective(
                        adapter, output, metas, targets, config, epoch-1)
                    (loss*len(block)/len(items)).backward()
                    values = {**parts,
                              **{f'weighted_{key}': value for key, value in contributions.items()},
                              **{f'native/{key}': value for key, value in native_details.items()
                                 if torch.is_tensor(value) and value.ndim == 0 and 'loss' in key},
                              'total': loss}
                    for key, value in values.items():
                        sums[key] = sums.get(key, 0.)+float(value.detach())*len(block)
                    for key, value in info.items():
                        diagnostics[key] = diagnostics.get(key, 0)+value
                    count += len(block)
                grad = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config['grad_clip_norm'], error_if_nonfinite=True)
                assert torch.isfinite(grad)
                optimizer.step()
                if step == 1 or step % 25 == 0 or step == len(loader):
                    progress = dict(epoch=epoch, step=step, steps=len(loader), samples=count,
                                    losses={key: value/count for key, value in sums.items()},
                                    gradient_norm=float(grad), effective_batch_size=effective,
                                    micro_batch_size=micro)
                    dump(run/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            scheduler.step()
            training_seconds = time.monotonic()-begun
            pre = run/'checkpoints'/f'epoch_{epoch:03d}_preval.pth'
            pre_record = checkpoint.save(pre, model, optimizer, scheduler, epoch, config)
            dump(run/'checkpoints'/f'epoch_{epoch:03d}_preval.json', pre_record, exclusive=True)
            result_dir = run/'val_results'/f'epoch_{epoch:03d}'
            metrics, original, joint_native, joint_geq = save_evaluation(
                model, val_data, 'val', result_dir,
                1 if args.backbone == 'flashvtg' else effective,
                config['presence_threshold'], pre)
            final = run/'checkpoints'/f'epoch_{epoch:03d}.pth'
            record = checkpoint.save(final, model, optimizer, scheduler, epoch, config, metrics)
            dump(run/'checkpoints'/f'epoch_{epoch:03d}.json', record, exclusive=True)
            dump(run/'logs'/f'epoch_{epoch:03d}.json', dict(
                epoch=epoch, losses={key: value/count for key, value in sums.items()},
                coverage_supervision=diagnostics, training_seconds=training_seconds,
                epoch_with_validation_seconds=time.monotonic()-begun,
                validation=metrics['brief'], original_gmr_validation=original['brief'],
                joint_native_validation=joint_native['brief'], joint_geq_validation=joint_geq['brief'],
                checkpoint=record), exclusive=True)
            selection(run)
            subprocess.run([sys.executable, '-m', 'd_gmr.scripts.report'], cwd=REPO, check=True)
            print(json.dumps(dict(epoch=epoch, validation=metrics['brief'], checkpoint=str(final))), flush=True)
        best = selection(run)
        dump(run/'TRAIN_COMPLETE.json', dict(epochs=config['epochs'], best=best,
                                             test_used=False, completed=time.time()), exclusive=True)
    except BaseException as error:
        dump(run/'TRAIN_FAILED.json', dict(
            error=str(error), traceback=traceback.format_exc(),
            completed_epoch_checkpoints=[str(path) for path in
                                         (run/'checkpoints').glob('epoch_*.pth')]))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', required=True, choices=BACKBONES)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', help='Any completed epoch; requires a fresh --output directory')
    parser.add_argument('--output')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
