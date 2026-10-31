import argparse
import json
import os
import pathlib
import sys
import time
import torch
from torch.utils.data import DataLoader
ROOT = pathlib.Path(os.environ.get("GMR_PROJECT_ROOT", pathlib.Path(__file__).resolve().parents[1]))
REPO = ROOT / 'generalized-moment-retrieval'
sys.path[:0] = [str(REPO / 'training/moment_detr_gmr'), str(REPO)]
from config import BaseOptions
from train import build_dataset_config, set_seed
from dataset import StartEndDataset, prepare_batch_inputs, start_end_collate
from evaluate import eval_epoch
from models.moment_detr_gmr.moment_detr import build_model
from multiroute import MultiRoute, MultiCriterion, VARIANTS


def setup(variant, device='cuda'):
    manager = BaseOptions('moment_detr', 'soccer_gmr', 'clip_slowfast')
    manager.parse()
    opt = manager.option
    opt.update(seed=2023, device=device, bsz=2, n_epoch=400, num_workers=0, mr_only=True,
               lw_saliency=0, use_exist_head=True, eval_epoch_interval=5, max_es_cnt=4)
    labels = ROOT / 'soccergmr_standard/d_gmr_two_stage/dataset/data/label/Standard'
    opt.train_path, opt.eval_path = str(labels/'train.jsonl'), str(labels/'val.jsonl')
    opt.v_feat_dirs = [str(ROOT/'mr_experiments/features'/k) for k in ('clip','slowfast')]
    opt.t_feat_dir = str(ROOT/'mr_experiments/features/clip_text')
    out = ROOT/'mr_experiments'/variant
    out.mkdir(parents=True, exist_ok=True)
    opt.results_dir = str(out)
    opt.ckpt_filepath = str(out/'best.ckpt')
    opt.eval_log_filepath = str(out/'val.log')
    set_seed(2023, device == 'cuda')
    base, criterion = build_model(opt)
    return opt, MultiRoute(base, variant).to(device), MultiCriterion(criterion, variant).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    opt, model, criterion = setup(args.variant)
    dataset = StartEndDataset(**build_dataset_config(opt,opt.train_path,keep_empty_gt=True))
    loader = DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=start_end_collate,num_workers=0)
    if args.smoke:
        # Explicit positive and empty batches from real feature files.
        for empty in (False, True):
            ids = [i for i,r in enumerate(dataset.data) if (len(r['relevant_windows']) == 0) == empty][:2]
            batch = start_end_collate([dataset[i] for i in ids])
            inputs, targets = prepare_batch_inputs(batch[1],opt.device)
            outputs = model(**inputs)
            losses = criterion(outputs,targets)
            loss = sum(v*criterion.weight_dict[k] for k,v in losses.items() if k in criterion.weight_dict)
            assert torch.isfinite(loss)
            loss.backward()
            for name,p in model.queries.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(),name
            model.zero_grad(set_to_none=True)
            model.eval()
            with torch.no_grad():
                result=model(**inputs)
                assert 'o2m_outputs' not in result and result['pred_spans'].shape[1] == 10
            model.train()
        print('SMOKE PASS',args.variant,flush=True)
        return
    out = pathlib.Path(opt.results_dir)
    if (out/'experiment_config.json').exists():
        raise RuntimeError('Output already initialized; refusing overwrite')
    (out/'experiment_config.json').write_text(json.dumps(dict(opt, variant=args.variant,accum=4,amp=False),indent=2))
    val = StartEndDataset(**build_dataset_config(opt,opt.eval_path,keep_empty_gt=False))
    optimizer = torch.optim.AdamW(model.parameters(),lr=opt.lr,weight_decay=opt.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,400)
    best, stale = -1.,0
    for epoch in range(400):
        start=time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total=0.
        for i,batch in enumerate(loader):
            inputs,targets=prepare_batch_inputs(batch[1],opt.device)
            loss_dict=criterion(model(**inputs),targets)
            loss=sum(v*criterion.weight_dict[k] for k,v in loss_dict.items() if k in criterion.weight_dict)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            group=min(4,len(loader)-(i//4)*4)
            (loss/group).backward()
            total+=float(loss.detach())
            if (i+1)%4==0 or i+1==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),.1)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        matches=criterion.matches
        stats=dict(epoch=epoch+1,loss=total/len(loader),seconds=time.time()-start,
                   matches_per_gt=sum(x for x,y in matches)/max(1,sum(y for x,y in matches)),
                   peak_memory_bytes=torch.cuda.max_memory_allocated())
        criterion.matches=[]
        with (out/'train.log').open('a') as f: f.write(json.dumps(stats)+'\n')
        print(stats,flush=True)
        if (epoch+1)%5: continue
        with torch.no_grad():
            metrics,_,_=eval_epoch(epoch,model,val,opt,'latest_val_preds.jsonl',criterion)
        score=metrics['brief']['MR-full-mAP']
        with (out/'val.log').open('a') as f: f.write(json.dumps(dict(epoch=epoch+1,metrics=metrics))+'\n')
        if score>best:
            best,stale=score,0
            torch.save(dict(model=model.state_dict(),opt=opt,epoch=epoch,variant=args.variant,
                            optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),out/'best.ckpt')
        else: stale+=1
        if stale>=10: break

if __name__=='__main__': main()
