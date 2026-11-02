#!/usr/bin/env python3
"""Bounded integration check on real assets; produces NO benchmark result.

Invoked once per backbone in a fresh process. Builds a small, declared subset
from train/Val and two Test manifests. No scores drive any hyperparameter choice.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def worker(args):
    import torch
    from soccergmr.config import load_config
    from soccergmr import train as training
    from soccergmr.test import test
    from soccergmr.io import dump_json,sha256
    from soccergmr import checkpoint as ck
    config=load_config(args.config)
    root=Path(args.output)
    if args.phase=='nulls':
        from soccergmr.backbones.registry import build,get_spec,forward
        from soccergmr.data import SoccerDataset,collate,move
        from soccergmr.geq import GEQFull
        spec=get_spec(config['backbone']);dataset=SoccerDataset(config,'train',spec)
        indices=[next(i for i,r in enumerate(dataset.data) if len(r['relevant_windows'])>=2),
                 next(i for i,r in enumerate(dataset.data) if not r['relevant_windows'])]
        model,criterion=build(config,args.device)
        proof=[]
        for stage in ('gmr','geq_full'):
            if stage=='geq_full':model=GEQFull(model,spec).to(args.device)
            model.train()
            for selection in (indices,[indices[1],indices[1]]):
                metas,inputs,targets=collate([dataset[i] for i in selection])
                inputs,targets=move(inputs,args.device),move(targets,args.device)
                outputs=forward(model,config['backbone'],inputs,targets,metas)
                loss,_=criterion(outputs.get('native_output',outputs),targets)
                if stage=='geq_full':loss=loss+torch.nn.functional.binary_cross_entropy_with_logits(outputs['geq_logit'],targets['exist_label'])
                loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                proof.append(dict(stage=stage,all_null=selection[0]==selection[1],loss=float(loss.detach()),finite_gradients=True))
                model.zero_grad(set_to_none=True)
        dump_json(root/'null_batch_proof.json',proof)
    elif args.phase=='interrupt':
        original=training.evaluate
        def terminate_after_validation(*values,**kwargs):
            result=original(*values,**kwargs)
            os.kill(os.getpid(),signal.SIGTERM)
            return result
        training.evaluate=terminate_after_validation
        result=training.train(config,root/'gmr','gmr',args.device)
        if result['status']!='interrupted' or result['completed_epoch']!=1:raise AssertionError(result)
    elif args.phase=='resume':
        result=training.train(config,root/'gmr','gmr',args.device,resume=True)
        assert result['completed_epoch']==2
    elif args.phase=='reference':
        training.train(config,root/'reference','gmr',args.device)
        resumed=ck.load(root/'gmr'/'last.pt');reference=ck.load(root/'reference'/'last.pt')
        assert all(torch.equal(v,reference['model'][k]) for k,v in resumed['model'].items()),'Resume changed model tensors'
        assert resumed['selector']==reference['selector'],'Resume changed selection'
        dump_json(root/'resume_proof.json',dict(exact_model_tensor_equality=True,exact_selector_equality=True,
            checkpoint_epoch=resumed['epoch'],tensor_count=len(resumed['model'])))
    elif args.phase=='geq':
        training.train(config,root/'geq_full','geq_full',args.device,baseline=root/'gmr'/'best.pt')
        initial=ck.load(root/'gmr'/'best.pt')['model'];tuned=ck.load(root/'geq_full'/'last.pt')['model']
        changed=[k for k,v in initial.items() if not torch.equal(v,tuned['base.'+k])]
        assert changed,'Native backbone did not update during GEQ Full'
        dump_json(root/'geq_proof.json',dict(native_changed_tensors=len(changed),native_total_tensors=len(initial),examples=changed[:8]))
    else:
        test(config,root/args.phase/'best.pt',root/(args.phase+'_test'),args.device)
    # This list proves imported project Python code comes from this checkout only.
    repo=Path(__file__).resolve().parents[1]
    imported={}
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if file:
            path=Path(file).resolve()
            if path.is_relative_to(Path.home()) and 'site-packages' not in path.parts and 'miniconda3' not in path.parts:
                assert path.is_relative_to(repo),f'External project import: {name}: {path}'
                imported[name]=str(path.relative_to(repo))
    dump_json(root/('imports_'+args.phase+'.json'),imported)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--phase',choices=['nulls','interrupt','resume','reference','geq','gmr','geq_full'])
    args=parser.parse_args()
    if args.phase:worker(args);return
    from soccergmr.config import load_config
    from soccergmr.io import read_jsonl,dump_json,dump_jsonl
    config=load_config(args.config);root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    def subset(rows,full=False):
        positive=[r for r in rows if r['relevant_windows']]
        negative=[r for r in rows if not r['relevant_windows']]
        chosen=[next((r for r in positive if len(r['relevant_windows'])>=2),positive[0]),negative[0]]
        if full:
            chosen.extend(sorted(rows,key=lambda r:r['duration'],reverse=True)[:2])
        else:
            chosen.extend(positive[1:4]);chosen.extend(negative[1:4])
        return list({r['qid']:r for r in chosen}.values())
    splits={}
    for split,path in config['data']['splits'].items():
        rows=subset(read_jsonl(path),split=='test_full')
        destination=root/(split+'.jsonl');dump_jsonl(destination,rows);splits[split]=str(destination)
    config['data']['splits']=splits
    config['data']['full_label']='Bounded real-data verification subset; NOT a benchmark result'
    config['protocol'].update(max_epochs=2,geq_max_epochs=2,batch_size=2,eval_batch_size=2,workers=0)
    config.pop('identity')
    # Resolved architecture is stored independently for relocatable config loading.
    architecture=root/'architecture.json';dump_json(architecture,config.pop('architecture'))
    config['architecture']=str(architecture)
    bounded=root/'smoke_config.json';dump_json(bounded,config)
    for phase in ('nulls','interrupt','resume','reference','geq','gmr','geq_full'):
        command=[sys.executable,str(Path(__file__).resolve()),'--config',str(bounded),'--output',str(root),'--device',args.device,'--phase',phase]
        with (root/(phase+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{phase} failed; see {root/(phase+".log")}')
        print(config['backbone'],phase,'PASS',flush=True)
    dump_json(root/'verification.json',dict(passed=True,backbone=config['backbone'],
        kind='bounded real-data integration check; NOT a benchmark result',
        training_rows=len(read_jsonl(splits['train'])),validation_rows=len(read_jsonl(splits['val'])),
        test_rows={k:len(read_jsonl(v)) for k,v in splits.items() if k.startswith('test_')},
        interrupted_and_resumed=True,exact_resume_compared_with_uninterrupted=True,
        stages=['gmr','geq_full'],all_imports_from_checkout=True,formal_retraining=False))

if __name__=='__main__':main()
