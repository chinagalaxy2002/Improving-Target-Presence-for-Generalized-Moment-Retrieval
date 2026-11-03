"""Two-stage Standard-only training. Test labels are never opened here."""
import json
import math
import signal
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from . import checkpoint as ck
from .backbones.registry import build,get_spec,forward
from .data import SoccerDataset,collate,move
from .evaluation import evaluate
from .geq import GEQFull
from .clip import wrap_gmr, clip_enabled, masked_clip_loss, clip_auxiliary_loss, load_native_gmr
from .config import stage_config_identity
from .io import dump_json,dump_jsonl,fingerprint,sha256
from .predict import predict
from .selection import Selector,score


def train(config,output,stage,device='cuda:0',baseline=None,resume=False):
    directory=Path(output).resolve()
    if stage not in ('gmr','geq_full'):raise ValueError(stage)
    with ck.run_lock(directory):
        return _train(config,directory,stage,device,baseline,resume)


def _train(config,directory,stage,device,baseline,resume):
    p=config['protocol'];spec=get_spec(config['backbone'])
    training=SoccerDataset(config,'train',spec);validation=SoccerDataset(config,'val',spec)
    if not len(training) or not len(validation):raise ValueError('Train and Val must be nonempty')
    if {Path(r['vid']).stem for r in training.data}&{Path(r['vid']).stem for r in validation.data}:
        raise ValueError('Train/Val video leakage')
    source=ck.source_identity();assets=ck.asset_identity([training,validation])
    identity=fingerprint(dict(config=stage_config_identity(config,stage),source=source,train_val_assets=assets,stage=stage))
    if (directory/'complete.json').exists():
        completed=json.loads((directory/'complete.json').read_text())
        if completed['identity']!=identity:raise ValueError('Completed run has a different identity')
        if stage=='geq_full' and (baseline is None or sha256(baseline)!=completed['baseline_sha256']):
            raise ValueError('Completed GEQ run has a different GMR parent checkpoint')
        if sha256(directory/'best.pt')!=completed['best_sha256']:raise ValueError('Completed best checkpoint was changed')
        return completed
    if (directory/'last.pt').exists() and not resume:raise ValueError('Existing last.pt: pass --resume to continue, or use a new directory')
    if resume and not (directory/'last.pt').is_file():raise FileNotFoundError('Resume requires legal last.pt')
    ck.seed_all(p['seed'])
    model,criterion=build(config,device)
    baseline_sha=None
    native_epoch_offset=0
    if stage=='geq_full':
        if baseline is None:raise ValueError('GEQ Full requires a finalized GMR best.pt')
        baseline=Path(baseline).resolve()
        completion=json.loads((baseline.parent/'complete.json').read_text())
        baseline_sha=sha256(baseline)
        if completion['stage']!='gmr' or completion['best_sha256']!=baseline_sha:raise ValueError('GMR checkpoint is not the finalized selected best')
        initial=ck.load(baseline,stage='gmr')
        native_epoch_offset=initial['epoch']
        expected=fingerprint(dict(config=stage_config_identity(config,'gmr'),source=source,train_val_assets=assets,stage='gmr'))
        if initial['identity']!=expected:raise ValueError('GMR initialization does not match this experiment')
        if initial['source'] != source or initial['train_val_assets'] != assets or completion['identity'] != initial['identity']:
            raise ValueError('GMR parent source/data/completion mismatch')
        if 'method' in config.get('clip',{}):
            model=load_native_gmr(model,spec,config,initial['model']).to(device)
        else:
            model=wrap_gmr(model,spec,config).to(device)
            model.load_state_dict(initial['model'],strict=True)
        inherited=None
        if hasattr(model,'clip_head'):
            inherited=model.clip_head
            model.close()
            model=model.base
        model=GEQFull(model,spec,clip_head=clip_enabled(config) and 'method' not in config.get('clip',{})).to(device)
        if inherited is not None:model.clip_head.load_state_dict(inherited.state_dict(),strict=True)
        if model.clip_head is not None:
            model.clip_head.requires_grad_(config['clip']['geq_weight'] > 0)
        groups=[dict(params=model.base.parameters(),lr=p['geq_lr_native']),dict(params=model.geq.parameters(),lr=p['geq_lr_head'])]
    else:
        model=wrap_gmr(model,spec,config).to(device)
        groups=[dict(params=model.parameters(),lr=p['learning_rate'])]
    if stage=='geq_full' and model.clip_head is not None and config['clip']['geq_weight']>0:
        groups.append(dict(params=model.clip_head.parameters(),lr=p['geq_lr_head']))
    optimizer=torch.optim.AdamW(groups,weight_decay=p['weight_decay'])
    # Constant learning rate is explicit; no hidden native scheduler/defaults.
    generator=torch.Generator().manual_seed(p['seed'])
    loader=DataLoader(training,batch_size=p['batch_size'],shuffle=True,generator=generator,
                      collate_fn=collate,num_workers=p['workers'])
    maximum=p['max_epochs'] if stage=='gmr' else p['geq_max_epochs']
    selector=Selector('Val mAP' if stage=='gmr' else 'Val mean G-mIoU@1 (.4/.5/.6)',p['patience'] if stage=='gmr' else 10)
    epoch=0;history=[]
    def payload():
        return dict(identity=identity,stage=stage,config=config,source=source,train_val_assets=assets,
                    model=model.state_dict(),optimizer=optimizer.state_dict(),selector=selector.state_dict(),
                    epoch=epoch,rng=ck.rng_state(generator),history=history,baseline_sha256=baseline_sha)
    if resume:
        saved=ck.load(directory/'last.pt',identity,stage)
        if saved['baseline_sha256']!=baseline_sha:raise ValueError('Different GMR parent checkpoint')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        selector=Selector.from_state(saved['selector']);epoch=saved['epoch'];history=saved['history']
        ck.restore_rng(saved['rng'],generator)
        # Discard an interrupted epoch's unpublished history; checkpoint is authoritative.
        dump_jsonl(directory/'history.jsonl',history)
    else:
        dump_json(directory/'config.json',config)
        ck.save(directory/'last.pt',payload())
    stop_requested=[]
    def request_stop(signum,frame):stop_requested.append(signum)
    previous={s:signal.signal(s,request_stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        while epoch<maximum and selector.stale<selector.patience:
            model.train();optimizer.zero_grad(set_to_none=True)
            total_loss=0.;total_clip=0.;batches=len(loader)
            for batch_index,(metas,inputs,targets) in enumerate(loader):
                if stop_requested:
                    # last.pt describes the previous COMPLETE epoch, including RNG.
                    return dict(status='interrupted',resume_from=str(directory/'last.pt'),completed_epoch=epoch)
                inputs,targets=move(inputs,device),move(targets,device)
                outputs=forward(model,config['backbone'],inputs,targets,metas,epoch=native_epoch_offset+epoch)
                loss,_=criterion(outputs.get('native_output',outputs),targets,epoch)
                if stage=='geq_full':
                    loss=loss+F.binary_cross_entropy_with_logits(outputs['geq_logit'],targets['exist_label'])
                clip_weight=config.get('clip',{}).get('gmr_weight' if stage=='gmr' else 'geq_weight',0)
                if 'method' in config.get('clip',{}):
                    clip_weight=config['clip']['loss_weight'] if stage=='gmr' else 0
                if clip_weight>0:
                    if 'method' in config.get('clip',{}):
                        key='saliency_scores' if config['clip']['method']=='native' else 'clip_logits'
                        clip_loss,_=clip_auxiliary_loss(outputs[key],targets,inputs['src_vid_mask'],config['clip'])
                    else:
                        clip_loss=masked_clip_loss(outputs['clip_logits'],targets['saliency_all_labels'],inputs['src_vid_mask'])
                    loss=loss+clip_weight*clip_loss
                    total_clip+=float(clip_loss.detach())
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                window_start=(batch_index//p['accumulation'])*p['accumulation']
                window_size=min(p['accumulation'],batches-window_start)
                (loss/window_size).backward();total_loss+=float(loss.detach())
                if (batch_index+1)%p['accumulation']==0 or batch_index+1==batches:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
            epoch+=1
            predictions=predict(model,validation,config,device)
            metrics=evaluate(predictions,validation.data)
            value=score(stage,metrics)
            decision=selector.update(epoch,value)
            improved=decision['improved']
            history.append(dict(epoch=epoch,loss=total_loss/batches,clip_loss=total_clip/batches,selection_score=value,
                                improved=improved,stale=selector.stale,metrics=metrics))
            state=payload()
            if improved:
                ck.save(directory/'best.pt',state)
                dump_json(directory/'best_val_metrics.json',metrics)
                dump_jsonl(directory/'best_val_predictions.jsonl',predictions)
            ck.save(directory/'last.pt',state)
            dump_jsonl(directory/'history.jsonl',history)
            print(json.dumps(dict(stage=stage,epoch=epoch,score=value,best=selector.best,stale=selector.stale)),flush=True)
            if stop_requested:return dict(status='interrupted',resume_from=str(directory/'last.pt'),completed_epoch=epoch)
        completed=dict(status='complete',stage=stage,identity=identity,completed_epoch=epoch,
            best_epoch=selector.best_epoch,best_score=selector.best,metric=selector.metric,
            reason='patience' if selector.stale>=selector.patience else 'max_epochs',
            best_sha256=sha256(directory/'best.pt'),source=source,train_val_assets=assets,
            baseline_sha256=baseline_sha,test_used_for_selection=False)
        dump_json(directory/'complete.json',completed)
        return completed
    finally:
        for signum,handler in previous.items():signal.signal(signum,handler)
