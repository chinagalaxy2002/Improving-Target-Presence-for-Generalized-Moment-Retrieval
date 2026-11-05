#!/usr/bin/env python3
"""Evaluate immutable historical Val-selected GMR / GEQ Full checkpoints.

This compatibility entry preserves historical input truncation and export.
It never trains or selects a checkpoint. It is separate from the new protocol.
"""
import argparse
import json
import os
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from soccergmr.backbones.registry import build,get_spec,forward
from soccergmr.geq import GEQFull
from soccergmr.config import Options
from soccergmr.checkpoint import run_lock,seed_all,source_identity
from soccergmr.data import normalize,move
from soccergmr.io import read_jsonl,sha256,dump_json,dump_jsonl,fingerprint
from soccergmr.evaluation import evaluate


class Inputs(Dataset):
    def __init__(self,labels,features,spec,opt):
        self.data=read_jsonl(labels);self.root=Path(features);self.spec=spec;self.opt=opt
        assert len({r['qid'] for r in self.data})==len(self.data)
    def __len__(self):return len(self.data)
    def paths(self,row):
        return [self.root/k/(Path(row['vid']).stem+'.npz') for k in self.spec.feature_order]+[self.root/'clip_text'/('qid'+str(row['qid'])+'.npz')]
    def __getitem__(self,index):
        row=self.data[index];paths=self.paths(row);arrays=[]
        for kind,path in zip(self.spec.feature_order,paths):
            with np.load(path) as z:a=z['features'][:self.opt.max_v_l].astype(np.float32)
            assert a.ndim==2 and a.shape[1]=={'clip':512,'slowfast':2304}[kind] and np.isfinite(a).all()
            arrays.append(normalize(a))
        n=min(map(len,arrays));video=torch.from_numpy(np.concatenate([a[:n] for a in arrays],-1))
        if self.spec.name in ('moment_detr','sim_detr'):
            start=np.arange(n,dtype=np.float32)/n
            tef=torch.from_numpy(np.stack([start,start+1/n],-1))
        elif self.spec.name=='eatr':
            start=torch.arange(n,dtype=torch.float32)/n;tef=torch.stack([start,start+1/n],-1)
        else:tef=torch.stack([torch.arange(n)/n,(torch.arange(n)+1)/n],-1)
        video=torch.cat([video,tef],-1)
        with np.load(paths[-1]) as z:
            query=z['last_hidden_state'][:self.opt.max_q_l].astype(np.float32)
            if self.spec.text_attention_mask:query=query[z['attention_mask'][:len(query)].astype(bool)]
        assert len(query)>0 and query.shape[1]==512 and np.isfinite(query).all()
        return row,video,torch.from_numpy(normalize(query))


def collate(items):
    metas,videos,texts=zip(*items)
    vm=pad_sequence([torch.ones(len(x)) for x in videos],batch_first=True)
    tm=pad_sequence([torch.ones(len(x)) for x in texts],batch_first=True)
    return list(metas),dict(src_vid=pad_sequence(videos,batch_first=True),src_vid_mask=vm,
                            src_txt=pad_sequence(texts,batch_first=True),src_txt_mask=tm)


def export(output,metas,name,opt):
    score=output['pred_logits'].softmax(-1)[...,0]
    logit=output.get('pred_exist_logits',output.get('pred_exist_logit')).reshape(-1)
    exist=logit.sigmoid()
    if name=='sim_detr':
        score=score*output['iou_scores'][...,0].sigmoid()
        score=score*torch.where(exist>=opt.exist_gate_thd,torch.ones_like(exist),exist)[:,None]
    center,width=output['pred_spans'].unbind(-1)
    spans=torch.stack([center-width/2,center+width/2],-1)
    assert all(torch.isfinite(x).all() for x in (spans,score,exist))
    result=[]
    for i,meta in enumerate(metas):
        duration=float(meta['duration']);windows=[]
        if name=='sim_detr':
            for (s,e),confidence in zip((spans[i]*duration).cpu().tolist(),score[i].cpu().tolist()):
                s=max(0.,min(duration,round(s/opt.clip_length)*opt.clip_length))
                e=max(0.,min(duration,round(e/opt.clip_length)*opt.clip_length))
                if e<=s:
                    if s<duration:e=min(duration,s+opt.clip_length)
                    else:s=max(0.,duration-opt.clip_length);e=duration
                windows.append([s,e,float(confidence)])
            windows.sort(key=lambda x:x[2],reverse=True)
        else:
            xy=(spans[i]*duration).clamp(0,duration)
            xy=(xy/2).round()*2;xy.clamp_(0,duration)
            candidates=torch.cat([xy,score[i,:,None]],-1)
            candidates=candidates[candidates[:,2].argsort(descending=True)]
            windows=candidates.cpu().tolist()
            if name!='cg_detr':windows=[[round(float(x),4) for x in w] for w in windows]
        result.append(dict(qid=meta['qid'],vid=meta['vid'],query=meta['query'],
                           pred_relevant_windows=windows[:opt.num_queries],pred_exist_score=float(exist[i])))
    return result


def run(args):
    manifest=json.loads(Path(args.manifest).read_text());entry=manifest['backbones'][args.backbone]
    spec=get_spec(args.backbone);torch.set_num_threads(2);seed_all(2023)
    baseline=torch.load(entry['gmr']['path'],map_location='cpu',weights_only=False)
    opt=baseline.get('opt',baseline.get('config'));opt=Options(opt if isinstance(opt,dict) else vars(opt))
    opt.binding_loss_coef=0.;opt.device=args.device
    config=dict(backbone=args.backbone,architecture=opt,data={'max_video_tokens':opt.max_v_l})
    output=Path(args.output)
    with run_lock(output):
        for split in ('standard','full'):
            data=Inputs(manifest['data'][split]['labels'],manifest['data'][split]['features'],spec,opt)
            expected=1036 if split=='standard' else 2970
            assert len(data)==expected
            assets={str(p):sha256(p) for row in data.data for p in data.paths(row)}
            for method in ('gmr','geq_full'):
                item=entry[method];path=Path(item['path']);before=sha256(path)
                assert before==item['sha256'],'Checkpoint changed'
                identity=fingerprint(dict(checkpoint=before,labels=sha256(manifest['data'][split]['labels']),
                     features=assets,source=source_identity(),runner=sha256(__file__),split=split))
                dest=output/split/method;dest.mkdir(parents=True,exist_ok=True)
                if (dest/'complete.json').exists():
                    old=json.loads((dest/'complete.json').read_text());assert old['identity']==identity
                    assert all(sha256(dest/f)==old[k] for f,k in [('predictions.jsonl','predictions_sha256'),('metrics.json','metrics_sha256')])
                    print('REUSE',args.backbone,split,method,flush=True);continue
                model,_=build(config,args.device)
                if method=='geq_full':model=GEQFull(model,spec).to(args.device)
                saved=torch.load(path,map_location='cpu',weights_only=False)
                model.load_state_dict(saved['model'],strict=True)
                model.eval().requires_grad_(False)
                batch=opt.eval_bsz if args.backbone=='sim_detr' else int(entry['batch_size'])
                loader=DataLoader(data,batch_size=batch,shuffle=False,num_workers=2,collate_fn=collate)
                predictions=[]
                with torch.inference_mode():
                    for metas,inputs in loader:
                        inputs=move(inputs,args.device)
                        result=forward(model,args.backbone,inputs,metas=metas)
                        predictions.extend(export(result,metas,args.backbone,opt))
                assert [p['qid'] for p in predictions]==[r['qid'] for r in data.data]
                metrics=evaluate(predictions,data.data)
                if split=='standard':
                    expected_report=json.loads(Path(item['standard_metrics']).read_text())
                    for tau in (.4,.5,.6):
                        original=expected_report[str(tau)]['brief']
                        for metric in ('mAP','mR@1','AUROC'):
                            assert abs(original[metric]-metrics['brief'][metric])<1e-9,(args.backbone,method,metric,original[metric],metrics['brief'][metric])
                        for k in (1,3):
                            key=f'G-mIoU@{k}'
                            assert abs(original[key]-metrics['gmiou'][str(tau)][key])<1e-9,(args.backbone,method,tau,key,original[key],metrics['gmiou'][str(tau)][key])
                assert sha256(path)==before
                dump_jsonl(dest/'predictions.jsonl',predictions);dump_json(dest/'metrics.json',metrics)
                dump_json(dest/'complete.json',dict(passed=True,identity=identity,backbone=args.backbone,
                    split=split,method=method,rows=len(predictions),positive=sum(bool(r['relevant_windows']) for r in data.data),
                    checkpoint_sha256=before,selected_epoch=saved['epoch'],selected_on='Standard Val',
                    strict_state_load=True,state_tensors=len(saved['model']),gradients_disabled=True,
                    training_called=False,test_selection=False,standard_exact_metric_reproduction=split=='standard',
                    max_video_tokens=opt.max_v_l,export_profile='historical Sim native' if args.backbone=='sim_detr' else 'historical E1 native',
                    predictions_sha256=sha256(dest/'predictions.jsonl'),metrics_sha256=sha256(dest/'metrics.json'),
                    label_sha256=sha256(manifest['data'][split]['labels']),feature_asset_identity=fingerprint(assets)))
                print('COMPLETE',args.backbone,split,method,metrics['brief']['mAP'],flush=True)
                if method=='geq_full':model.close()
                del model
                torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--backbone',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
