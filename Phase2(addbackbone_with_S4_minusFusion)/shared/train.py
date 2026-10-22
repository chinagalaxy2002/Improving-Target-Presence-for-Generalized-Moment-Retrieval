#!/usr/bin/env python3
"""Real GMR-initialized S4 fine-tuning: independent GPU preflight, then Train/Val."""
import argparse
import functools
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from official_eval.eval_main import evaluate_gmr
from postprocessing import export
from selection import Evaluation, selection_score

ROOT = Path(__file__).resolve().parents[1]


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def save(path, state):
    temporary = path.with_suffix('.tmp'); torch.save(state, temporary); temporary.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def fingerprint(path):
    with Path(path).open('rb') as stream: return hashlib.sha256(stream.read()).hexdigest()


def implementation(backbone):
    files = list((ROOT / 'shared').glob('*.py')) + list((ROOT / backbone).glob('*.py'))
    files += list((ROOT / backbone / 'source').rglob('*.py'))
    return {str(p.relative_to(ROOT)): fingerprint(p) for p in sorted(files)}


def optimizer_for(model, job):
    groups = [{'params': [p for n,p in model.named_parameters() if p.requires_grad and '.lora_' not in n], 'lr': job['lr']},
              {'params': [p for n,p in model.named_parameters() if p.requires_grad and '.lora_' in n], 'lr': job['lora_lr']}]
    optimizer = torch.optim.AdamW(groups, weight_decay=job['wd'], foreach=False)
    assert not optimizer.state
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, job['lr_drop'], gamma=.5)
    return optimizer, scheduler


def frozen_values(model):
    return {n: p.detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad}


def assert_frozen(model, frozen):
    for n,p in model.named_parameters():
        if n in frozen:
            assert p.grad is None and torch.equal(p.detach().cpu(), frozen[n]), n


def sample_indices(dataset):
    indices = [next(i for i,m in enumerate(dataset.data) if len(m['relevant_windows']) == 1),
               next(i for i,m in enumerate(dataset.data) if len(m['relevant_windows']) > 1),
               next(i for i,m in enumerate(dataset.data) if not m['relevant_windows'])]
    short = next((i for i,m in enumerate(dataset.data) if 0 < m['duration'] < 150 and m['relevant_windows'] and i not in indices), None)
    if short is not None: indices.append(short)
    return indices


@torch.no_grad()
def predict(backend, model, dataset, batch_size, device, indices=None):
    model.eval(); predictions = []; groundtruth = []
    order = list(range(len(dataset))) if indices is None else indices
    for start in range(0, len(order), batch_size):
        metas, inputs = backend.collate([dataset[i] for i in order[start:start+batch_size]], device)
        predictions.extend(export(model(**inputs), metas)); groundtruth.extend(metas)
    assert len(predictions) == len(groundtruth) == len(order)
    assert {p['qid'] for p in predictions} == {g['qid'] for g in groundtruth}
    return predictions, groundtruth


def metrics_for(predictions, groundtruth):
    official = evaluate_gmr(predictions, groundtruth, cls_thresholds=(.4, .5, .6), map_num_workers=0, verbose=False)
    native = [{**p, 'pred_relevant_windows': p['pred_relevant_windows_pre_exist']} for p in predictions]
    report = Evaluation(native, predictions, groundtruth).report('val')
    score = selection_score(report)
    assert math.isfinite(score)
    assert abs(official['brief']['mAP'] - report['official_postprocessed']['mAP']) < .02
    assert abs(official['brief']['G-mIoU@1'] - report['gate_04']['G-mIoU@1']) < .02
    return {'official': official, 's4_selection': report, 'selection_score': score}


def step(backend, model, optimizer, metas, inputs, job):
    model.train(); optimizer.zero_grad(set_to_none=True)
    output = model(**inputs)
    assert torch.isfinite(output['pred_exist_logits']).all() and torch.isfinite(output['candidate_seconds']).all()
    value, detail = backend.loss(output, metas, job)
    assert torch.isfinite(value)
    value.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], job['grad_clip'], error_if_nonfinite=True)
    optimizer.step()
    return float(value.detach()), {k:float(v.detach()) for k,v in detail.items()}


def preflight(backend, args, job, directory):
    seed_all(job['seed'])
    model, init = backend.initialize(args.device, job, snapshot=args.snapshot)
    assert init['verified_gmr_baseline']
    assert fingerprint(init['snapshot']) == init['sha256']
    optimizer, scheduler = optimizer_for(model, job)
    train = backend.get_dataset('train'); val = backend.get_dataset('val')
    indices = sample_indices(train)
    batch_size = int(init['baseline_config']['bsz'])
    metas, inputs = backend.collate([train[indices[i % len(indices)]] for i in range(batch_size)], args.device)
    frozen = frozen_values(model)
    before = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    gradients = set(); losses = []
    if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
    for index in range(3):
        value, _ = step(backend, model, optimizer, metas, inputs, job)
        gradients.update(n for n,p in model.named_parameters() if p.grad is not None and bool(p.grad.abs().max() > 0))
        assert_frozen(model, frozen); losses.append(value)
        print(json.dumps({'event':'preflight_step','backbone':args.backbone,'step':index+1,'loss':value}), flush=True)
    changed = [n for n,p in model.named_parameters() if n in before and not torch.equal(before[n],p.detach().cpu())]
    prefixes = {n.split('.')[0] for n in before if not n.startswith('base.')}
    for prefix in prefixes:
        assert any(n.startswith(prefix+'.') for n in gradients) and any(n.startswith(prefix+'.') for n in changed), prefix
    for suffix in ('lora_A', 'lora_B'):
        assert any(n.endswith(suffix) for n in gradients) and any(n.endswith(suffix) for n in changed), suffix
    neg_index = next(i for i,m in enumerate(train.data) if not m['relevant_windows'])
    negative, neg_inputs = backend.collate([train[neg_index]], args.device)
    optimizer.zero_grad(set_to_none=True); neg_out = model(**neg_inputs); neg_loss,_ = backend.loss(neg_out,negative,job)
    assert torch.isfinite(neg_loss); neg_loss.backward(); assert_frozen(model,frozen)
    check = directory / 'preflight_checkpoint.pt'
    save(check, {'model':model.state_dict(), 'job':job, 'initialization':init, 'implementation':implementation(args.backbone)})
    model.eval()
    with torch.no_grad(): expected = model(**inputs)
    restored, other_init = backend.initialize(args.device, job, snapshot=init['snapshot'])
    restored.load_state_dict(torch.load(check,map_location='cpu',weights_only=False)['model'],strict=True); restored.eval()
    assert other_init['sha256'] == init['sha256']
    with torch.no_grad(): actual = restored(**inputs)
    assert torch.equal(expected['pred_exist_logits'], actual['pred_exist_logits'])
    assert torch.equal(expected['candidate_seconds'], actual['candidate_seconds'])
    predictions, groundtruth = predict(backend, restored, val, batch_size, args.device, sample_indices(val))
    metrics = metrics_for(predictions, groundtruth)
    result = {'passed':True, 'device':args.device,'initialization':init,'checkpoint_sha256':init['sha256'],
              'job_sha256':hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),
              'implementation':implementation(args.backbone),'losses':losses,'all_negative_loss':float(neg_loss.detach()),
              'batch_size':batch_size,'frozen_unchanged':True,'strict_reload_equal':True,
              'metrics':metrics,'peak_gpu_mib':torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else None}
    dump(directory/'PREFLIGHT.json',result)
    print('GPU_PREFLIGHT_PASSED' if args.device.startswith('cuda') else 'CPU_PREFLIGHT_PASSED', args.backbone, flush=True)
    del model,restored,optimizer,scheduler,expected,actual,inputs,neg_inputs; gc.collect()
    if args.device.startswith('cuda'): torch.cuda.empty_cache()
    return init


def train(backend,args,job,directory):
    from epoch_plots import generate as generate_epoch_plots
    seed_all(job['seed']); model,init = backend.initialize(args.device,job,snapshot=args.snapshot)
    assert init['verified_gmr_baseline'] and job['epochs'] == 6
    preflight_result = json.loads((directory/'PREFLIGHT.json').read_text())
    assert preflight_result['passed'] and preflight_result['checkpoint_sha256'] == init['sha256']
    assert preflight_result['job_sha256'] == hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
    assert preflight_result['implementation'] == implementation(args.backbone)
    optimizer,scheduler = optimizer_for(model,job); frozen = frozen_values(model)
    tr = backend.get_dataset('train')
    bsz = int(init['baseline_config']['bsz']); eval_bsz = int(init['baseline_config'].get('eval_bsz',bsz))
    loader = DataLoader(tr,batch_size=bsz,shuffle=True,num_workers=0,collate_fn=list,
                        generator=torch.Generator().manual_seed(job['seed']))
    code_hashes = implementation(args.backbone)
    dump(directory/'CONFIG.json',{'job':job,'initialization':init,'batch_size':bsz,'eval_batch_size':eval_bsz,
                                'train_count':len(tr),'implementation':code_hashes,
                                'fresh_optimizer_scheduler':True,'evaluation_split':'test',
                                'evaluation_epochs':[3,4,5,6],'checkpoint_epochs':[1,2,3,4,5,6],
                                'checkpoint_selection':'none; every epoch retained'})
    checkpoints = directory/'checkpoints'; checkpoints.mkdir()
    print('TRAINING_STARTED',args.backbone,'exact_epochs',6,'batch_size',bsz,flush=True)
    updates = 0
    for epoch in range(1,7):
        begun = time.monotonic(); total = 0.
        for index,items in enumerate(loader):
            metas,inputs = backend.collate(items,args.device)
            value,detail = step(backend,model,optimizer,metas,inputs,job); total += value; updates += 1
            if index == 0 or (index+1)%20 == 0:
                progress = {'status':'TRAINING','backbone':args.backbone,'epoch':epoch,'batch':index+1,
                            'batches_per_epoch':len(loader),'optimizer_updates':updates,'loss':value,'losses':detail,
                            'physical_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'baseline_sha256':init['sha256']}
                dump(directory/'progress.json',progress); print(json.dumps(progress),flush=True)
        scheduler.step(); assert_frozen(model,frozen)
        state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                 'epoch':epoch,'optimizer_updates':updates,'job':job,'initialization':init,'implementation':code_hashes,
                 'checkpoint_selection':'none; retain all six epochs',
                 'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                        'cuda':torch.cuda.get_rng_state_all() if args.device.startswith('cuda') else [],'loader':loader.generator.get_state()}}
        checkpoint = checkpoints/f'epoch_{epoch:03d}.ckpt'
        assert not checkpoint.exists()
        save(checkpoint,state); checkpoint.chmod(0o444)
        checksum = fingerprint(checkpoint)
        temporary = directory/'.latest.tmp'; temporary.unlink(missing_ok=True)
        temporary.symlink_to('checkpoints/'+checkpoint.name); temporary.replace(directory/'latest.ckpt')
        dump(checkpoint.with_suffix('.json'),{'epoch':epoch,'checkpoint':str(checkpoint),'sha256':checksum,
             'optimizer_updates':updates,'baseline_sha256':init['sha256'],'saved_every_epoch':True})
        record = {'epoch':epoch,'mean_loss':total/len(loader),'optimizer_updates':updates,
                  'checkpoint':str(checkpoint),'checkpoint_sha256':checksum,'checkpoint_selection':'none',
                  'evaluation_split':'test' if epoch >= 3 else None,'evaluation_performed':False}
        if epoch >= 3:
            dump(directory/'progress.json',dict(record,status='TEST_EVALUATION',backbone=args.backbone))
            # Explicitly reload the matching saved epoch before test inference.
            saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert saved['epoch'] == epoch and fingerprint(checkpoint) == checksum
            model.load_state_dict(saved['model'],strict=True)
            test = backend.get_dataset('test')
            assert not ({m['qid'] for m in tr.data} & {m['qid'] for m in test.data})
            predictions,groundtruth = predict(backend,model,test,eval_bsz,args.device)
            epochdir = directory/'evaluations'/f'epoch_{epoch:03d}'; epochdir.mkdir(parents=True)
            prediction_path = epochdir/'test_predictions.jsonl'
            prediction_path.write_text(''.join(json.dumps(p)+'\n' for p in predictions))
            figure_manifest,metrics = generate_epoch_plots(args.backbone,epoch,predictions,groundtruth,checkpoint,prediction_path)
            dump(epochdir/'test_metrics.json',metrics); dump(epochdir/'FIGURES.json',figure_manifest)
            dump(directory/'latest_test_metrics.json',metrics)
            record.update(evaluation_performed=True,test_queries=len(groundtruth),test_metrics=metrics['brief'],figures=figure_manifest)
            del saved,test,predictions,groundtruth
        record['elapsed_seconds'] = time.monotonic()-begun
        with (directory/'history.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        dump(directory/'progress.json',dict(record,status='TRAINING',backbone=args.backbone))
        print('EPOCH_COMPLETE',args.backbone,json.dumps(record),flush=True)
    assert len(list(checkpoints.glob('epoch_*.ckpt'))) == 6
    assert len(list((directory/'evaluations').glob('epoch_*/test_metrics.json'))) == 4
    dump(directory/'COMPLETE.json',{'status':'COMPLETED','epochs':6,'optimizer_updates':updates,
         'checkpoints':[str(checkpoints/f'epoch_{e:03d}.ckpt') for e in range(1,7)],
         'evaluated_test_epochs':[3,4,5,6],'checkpoint_selection':'none','frozen_unchanged':True})
    dump(directory/'progress.json',{'status':'COMPLETED','epoch':6,'optimizer_updates':updates,
                                  'checkpoint_count':6,'test_evaluation_count':4,'checkpoint_selection':'none'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',choices=('moment_detr','qd_detr','cg_detr','eatr'),required=True)
    parser.add_argument('--mode',choices=('deploy','preflight','eval'),default='deploy')
    parser.add_argument('--device',default='cuda:0'); parser.add_argument('--run-name',default='s2026_gmr_s4_e6')
    parser.add_argument('--snapshot'); parser.add_argument('--checkpoint',type=Path)
    args = parser.parse_args(); torch.set_num_threads(2)
    sys.path.insert(0,str(ROOT/args.backbone)); backend = importlib.import_module('backend')
    directory = (ROOT/args.backbone/'runs'/args.run_name).resolve()
    assert directory.is_relative_to((ROOT/args.backbone/'runs').resolve())
    if directory.exists():raise FileExistsError('Fresh output directory required: '+str(directory))
    directory.mkdir(parents=True)
    job = json.loads((ROOT/'shared/config.json').read_text())
    assert job['full_training_authorized'] and job['seed'] == 2026 and job['presence_source'] == 'global_refine'
    assert not any(job.get(k,False) for k in ('augmentation','qcd','query_swap','background_replacement','synthetic_negatives','synthetic_samples','hard_negative'))
    if args.mode == 'eval':
        if args.checkpoint is None:parser.error('--checkpoint required for eval')
        state = torch.load(args.checkpoint,map_location='cpu',weights_only=False); job = state['job']
        assert state['implementation'] == implementation(args.backbone),'Saved checkpoint implementation differs'
        model,init = backend.initialize(args.device,job,snapshot=state['initialization']['snapshot'])
        assert init['sha256'] == state['initialization']['sha256']
        model.load_state_dict(state['model'],strict=True)
        predictions,groundtruth = predict(backend,model,backend.get_dataset('val'),int(init['baseline_config'].get('eval_bsz',32)),args.device)
        dump(directory/'validation_metrics.json',metrics_for(predictions,groundtruth)); return
    initialization = preflight(backend,args,job,directory)
    args.snapshot = initialization['snapshot']
    if args.mode == 'deploy':train(backend,args,job,directory)


if __name__ == '__main__':
    main()
