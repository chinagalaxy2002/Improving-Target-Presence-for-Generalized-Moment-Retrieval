#!/usr/bin/env python3
"""Fixed joint training, isolated preflight, and final-epoch-only evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, hashlib, json, time, traceback
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from common import ROOT, OLD, BACKBONES, dump, rows, seed, sha
from adapters import Adapter
from model import JointGMR

def objectives(adapter, out, meta, targets, epoch=0):
    native, ld = adapter.native_loss(out['native_output'], targets, meta, epoch)
    geq = F.binary_cross_entropy_with_logits(out['pred_exist_logits'].reshape(-1), out['pred_exist_logits'].new_tensor([bool(m['relevant_windows']) for m in meta]))
    parts = dict(native=native, geq=geq)
    if 'quality_logit' in out:
        from pv import losses
        quality, representation = losses(out, meta)
        parts.update(quality=quality, representation=representation)
    total = native+geq+parts.get('quality', 0)+.2*parts.get('representation', 0)
    assert torch.isfinite(total), {k:float(v.detach()) for k,v in parts.items()}
    return total, parts, ld

def optimizer_for(model):
    return torch.optim.AdamW([dict(params=list(model.base.parameters()), lr=1e-5, name='pretrained_gmr'),
                             dict(params=[p for n,p in model.named_parameters() if not n.startswith('base.')], lr=3e-5, name='new_geq_pv')], weight_decay=1e-4, foreach=False)

def gradient_norm(loss, named):
    named = [(n,p) for n,p in named if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for n,p in named], retain_graph=True, allow_unused=True)
    nonzero = {n:float(g.norm()) for (n,p),g in zip(named,grads) if g is not None and bool(g.abs().max()>0)}
    assert nonzero, 'Requested loss has no nonzero gradient to expected module'
    assert all(torch.isfinite(g).all() for g in grads if g is not None)
    return nonzero

def preflight(adapter, model, train, val, directory):
    initial_base = {n:p.detach().cpu().clone() for n,p in model.base.named_parameters()}
    model.train(); assert model.base.training
    assert all(p.requires_grad for p in model.parameters())
    indices = {n:next(i for i,m in enumerate(train.data) if (not m['relevant_windows'] if n=='negative' else len(m['relevant_windows'])>=2 if n=='multi' else bool(m['relevant_windows']))) for n in ('positive','negative','multi')}
    short = next(i for i,m in enumerate(train.data) if m['duration']<150 and m['relevant_windows'])
    meta, inputs, targets = adapter.collate([train[i] for i in [indices['positive'],indices['negative'],indices['multi'],short]])
    out = model(**inputs); loss, parts, native_details = objectives(adapter,out,meta,targets)
    transformer = list(model.base.transformer.named_parameters())
    checks = dict(geq_to_transformer=gradient_norm(parts['geq'],transformer),
                  geq_to_text_projection=gradient_norm(parts['geq'],model.base.input_txt_proj.named_parameters()),
                  geq_to_video_projection=gradient_norm(parts['geq'],model.base.input_vid_proj.named_parameters()))
    localization_names = ('span_embed','moment_span_embed','coord_head','class_head','class_embed')
    loc_parameters = [(n,p) for n,p in model.base.named_parameters() if any(x in n for x in localization_names)]
    checks['native_to_localization_head'] = gradient_norm(parts['native'],loc_parameters)
    regression = ('coord_head',) if adapter.name == 'flashvtg' else ('moment_span_embed',) if adapter.name == 'eatr' else ('span_embed',)
    checks['native_to_boundary_regression'] = gradient_norm(parts['native'],[(n,p) for n,p in model.base.named_parameters() if any(x in n for x in regression)])
    checks['native_to_existence_head'] = gradient_norm(parts['native'],model.base.exist_head.named_parameters())
    if hasattr(model,'pv'):
        checks['quality_to_transformer'] = gradient_norm(parts['quality'],transformer)
        checks['representation_to_transformer'] = gradient_norm(parts['representation'],transformer)
    if adapter.name == 'flashvtg':
        order = out['native_output']['out_class'][...,0].argsort(dim=1,descending=True)
        decoded = out['candidate_seconds'].gather(1,order[...,None].expand(-1,-1,3))
        original = out['native_output']['pred_spans'].clone()
        original[...,:2] = torch.minimum(original[...,:2].clamp_min(0), inputs['durations'][:,None,None])
        torch.testing.assert_close(decoded, original)
    # Top-K IDs gather exactly from original point/slot order.
    assert torch.equal(out['candidate_spans'],out['candidate_seconds'].gather(1,out['candidate_indices'][...,None].expand(-1,-1,3)))
    assert (out['candidate_spans'][...,0]>=0).all() and (out['candidate_spans'][...,1]<=inputs['durations'][:,None]).all()
    del out, loss, parts
    optimizer = optimizer_for(model); cases = {}
    for case, ids in [('positive',[indices['positive']]), ('negative',[indices['negative']]), ('multi',[indices['multi']]), ('native_full_batch',list(range(adapter.bs)))]:
        model.train(); optimizer.zero_grad(set_to_none=True)
        meta, inputs, targets = adapter.collate([train[i] for i in ids])
        out = model(**inputs); loss, parts, ld = objectives(adapter,out,meta,targets)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True); optimizer.step()
        cases[case] = dict(loss=float(loss.detach()), parts={k:float(v.detach()) for k,v in parts.items()}, gradient_norm=float(norm))
        if case=='native_full_batch':
            missing = [n for n,p in model.base.named_parameters() if p.grad is None]
            zero = [n for n,p in model.base.named_parameters() if p.grad is not None and not p.grad.count_nonzero()]
        del out, loss, parts, ld
    allowed_missing = ['txt_position_embed.']
    if adapter.name in ('moment_detr','eatr'): allowed_missing += ['saliency_proj.']
    if adapter.name == 'flashvtg': allowed_missing += ['saliency_proj1.','saliency_proj2.']
    if adapter.name == 'cg_detr': allowed_missing += ['txt_proj_linear.']
    unexpected = [n for n in missing if not any(n.startswith(prefix) for prefix in allowed_missing)]
    assert not unexpected, ('Unexplained absent native gradients', unexpected)
    changed = [n for n,p in model.base.named_parameters() if not torch.equal(p.detach().cpu(),initial_base[n])]
    assert any('transformer.' in n for n in changed)
    assert any(any(x in n for x in regression) for n in changed)
    checkpoint = directory/'smoke.ckpt'
    torch.save(dict(model=model.state_dict(), smoke_only=True),checkpoint)
    model.eval()
    metas, inputs, _ = adapter.collate([val[0]])
    with torch.no_grad():
        before = model(**inputs)
        expected = {k:before[k].clone() for k in ('pred_exist_logits','candidate_seconds','quality_score') if k in before}
    # Rebuild a new object and strictly reload every tensor.
    second_adapter = Adapter(adapter.name, adapter.device)
    second = JointGMR(second_adapter.base, adapter.name, hasattr(model,'pv')).to(adapter.device)
    second.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True); second.eval()
    with torch.no_grad():
        after = second(**inputs)
        assert all(torch.equal(v,after[k]) for k,v in expected.items())
    from evaluation import predict, evaluate
    # Actual native export and official evaluation on a mixed, multi-GT Val subset.
    vi = [next(i for i,m in enumerate(val.data) if len(m['relevant_windows'])>=2),next(i for i,m in enumerate(val.data) if not m['relevant_windows'])]
    class View:
        def __len__(self): return len(vi)
        def __getitem__(self, i): return val[vi[i]]
    pred, diag = predict(adapter, second, View())
    evaluate(pred,[val.data[i] for i in vi],directory,'smoke_val',diag)
    report = dict(passed=True, backbone=adapter.name, setting='E2' if hasattr(model,'pv') else 'E1', native_batch_size=adapter.bs,
                  strict_rebuilt_reload_equal=True, native_inference_and_official_eval=True,
                  all_parameters_require_grad=True, base_training_mode=True, cases=cases, gradient_checks=checks,
                  no_gradient_parameters=missing, zero_gradient_parameters=zero,
                  no_gradient_explanation='Listed explicitly; unused native saliency/projection branches or conditional branches are not evidence of an updated tensor. See architecture audit.',
                  peak_vram_mib=torch.cuda.max_memory_allocated()/2**20,
                  changed_native_parameters=changed,
                  code_sha256={n:sha(ROOT/n) for n in ('run.py','adapters.py','model.py','pv.py','evaluation.py','independent_metrics.py')},
                  checkpoint_sha256=sha(checkpoint))
    dump(directory/'PREFLIGHT.json',report)
    return report

def main():
    p=argparse.ArgumentParser(); p.add_argument('--backbone',choices=BACKBONES,required=True); p.add_argument('--setting',choices=['E1','E2'],required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args(); torch.set_num_threads(2); seed()
    directory=ROOT/('preflights' if args.preflight_only else 'runs')/args.backbone/args.setting
    directory.mkdir(parents=True,exist_ok=False)
    dump(directory/'STATUS.json',dict(status='initializing',pid=os.getpid(),started=time.time()))
    try:
        adapter=Adapter(args.backbone,args.device); model=JointGMR(adapter.base,args.backbone,args.setting=='E2').to(args.device)
        shared_initial = {n:hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,p in model.state_dict().items() if n.startswith(('base.','geq.'))}
        if args.setting == 'E2' and not args.preflight_only:
            sibling = json.loads((ROOT/f'runs/{args.backbone}/E1/CONFIG.json').read_text())
            assert shared_initial == sibling['shared_initial_tensor_sha256'], 'E1/E2 shared initial tensors differ'
        dump(directory/'CONFIG.json',dict(shared_initial_tensor_sha256=shared_initial,protocol=json.loads((ROOT/'config.json').read_text()),initialization=adapter.manifest,
             trainable_parameters={n:p.numel() for n,p in model.named_parameters()},
             source_sha256={str(f.relative_to(ROOT)):sha(f) for f in ROOT.glob('*.py')},
             native_source_sha256={str(f.relative_to(OLD)):sha(f) for f in (OLD/('baseline_source' if args.backbone=='flashvtg' else f'backends/{args.backbone}')).rglob('*.py')},
             data_sha256={s:sha(ROOT/f'data/{s}.jsonl') for s in ('train','val','test')}))
        train=adapter.dataset('train'); val=adapter.dataset('val')
        assert {m['qid'] for m in train.data}.isdisjoint(m['qid'] for m in val.data)
        if args.preflight_only:
            preflight(adapter,model,train,val,directory)
            dump(directory/'STATUS.json',dict(status='preflight_passed',finished=time.time())); print('PREFLIGHT_PASSED',args.backbone,args.setting,flush=True); return
        gate = json.loads((ROOT/'preflights'/args.backbone/args.setting/'PREFLIGHT.json').read_text())
        assert gate['passed']
        assert all(sha(ROOT/n) == checksum for n,checksum in gate['code_sha256'].items()), 'Code changed after preflight; revalidate before training'
        assert json.loads((ROOT/'reports/ASSET_AUDIT.json').read_text())['passed']
        # Fresh original checkpoint factory above: preflight optimizer never reused.
        optimizer=optimizer_for(model); scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:1.)
        seed(); loader=DataLoader(train,batch_size=adapter.bs,shuffle=True,num_workers=0,collate_fn=list,generator=torch.Generator().manual_seed(2026),drop_last=False)
        for epoch in range(1,5):
            start=time.monotonic(); sums={}
            for step,items in enumerate(loader,1):
                model.train(); adapter.criterion.train(); optimizer.zero_grad(set_to_none=True)
                meta,inputs,targets=adapter.collate(items); out=model(**inputs)
                loss,parts,ld=objectives(adapter,out,meta,targets,epoch-1); loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),.1,error_if_nonfinite=True); optimizer.step()
                log_values = dict(parts,total=loss)
                log_values.update({f'native/{k}':v for k,v in ld.items() if 'loss' in k and torch.is_tensor(v) and v.ndim == 0})
                for k,v in log_values.items(): sums[k]=sums.get(k,0)+float(v.detach())
                if step==1 or step%25==0:
                    progress=dict(status='training',backbone=args.backbone,setting=args.setting,epoch=epoch,step=step,total_steps=len(loader),loss=float(loss.detach()),gradient_norm=float(norm),peak_vram_mib=torch.cuda.max_memory_allocated()/2**20)
                    dump(directory/'STATUS.json',progress); print(json.dumps(progress),flush=True)
            scheduler.step()
            with (directory/'history.jsonl').open('a') as f: f.write(json.dumps(dict(epoch=epoch,mean_losses={k:v/len(loader) for k,v in sums.items()},seconds=time.monotonic()-start))+'\n')
            checkpoint=directory/f'epoch_{epoch:03d}.ckpt'
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,initialization=adapter.manifest,setting=args.setting,rng=dict(python=__import__('random').getstate(),numpy=__import__('numpy').random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),dataloader=loader.generator.get_state())),checkpoint)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
        dump(directory/'CHECKPOINT.json',dict(path=str(checkpoint),sha256=sha(checkpoint),epoch=4,initialization_sha256=adapter.manifest['sha256']))
        dump(directory/'STATUS.json',dict(status='evaluating',epoch=4))
        from evaluation import predict,evaluate
        for split,data in [('val',val),('test',adapter.dataset('test'))]:
            pred,diag=predict(adapter,model,data)
            baseline=rows(OLD/f'results/{args.backbone}/gmr/{split}_predictions.jsonl')
            evaluate(pred,data.data,directory,split,diag,baseline)
        dump(directory/'COMPLETE.json',dict(status='complete',epoch=4,seed=2026,backbone=args.backbone,setting=args.setting))
        dump(directory/'STATUS.json',dict(status='complete',epoch=4,finished=time.time()))
    except Exception:
        dump(directory/'STATUS.json',dict(status='failed',traceback=traceback.format_exc(),finished=time.time())); raise

if __name__=='__main__': main()
