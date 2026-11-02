#!/usr/bin/env python3
"""Strict-load, gradient-free Full Test inference. No training entry point is called."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse
import copy
import fcntl
import random
import sys
import time
import numpy as np
if not hasattr(np,'trapz'): np.trapz=np.trapezoid
import torch
from torch.utils.data import Dataset
from common import *
sys.path.insert(0,str(KIT))
from native_adapter import Adapter
from geq_model import RouteGEQ

class FullSoccer(Dataset):
    """Native DAB loaders with Full paths, extension-safe lookup and max_v_l=75.

    Normalization, stream order, token truncation, TEF and return formats follow
    deploy.py / train_gmr.py. Labels are only carried for native collation; no
    target, saliency or auxiliary route is passed into the inference model.
    """
    def __init__(self, adapter, data):
        self.adapter=adapter; self.data=data; self.cache={}

    def __len__(self): return len(self.data)

    def __getitem__(self,i):
        d=self.data[i]; a=self.adapter; name=a.name; m=a.experiment.module
        key=Path(d['vid']).stem
        if key not in self.cache:
            vs=[]
            for kind in ('slowfast','clip'):
                with np.load(FEATURES/kind/f'{key}.npz') as z: v=z['features'][:75].astype(np.float32)
                vs.append(v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-5))
            n=min(map(len,vs))
            v=torch.from_numpy(np.concatenate([x[:n] for x in vs],-1))
            if name=='eatr':
                st=torch.arange(n,dtype=torch.float32)/n
                tef=torch.stack([st,st+1/n],-1)
            else: tef=torch.stack([torch.arange(n)/n,(torch.arange(n)+1)/n],-1)
            self.cache[key]=torch.cat([v,tef],-1)
        video=self.cache[key]; n=len(video)
        with np.load(FEATURES/'clip_text'/f'qid{d["qid"]}.npz') as z:
            q=z['last_hidden_state'][:32].astype(np.float32)
        query=torch.from_numpy(q/(np.linalg.norm(q,axis=-1,keepdims=True)+1e-5))
        windows=torch.tensor(d['relevant_windows'],dtype=torch.float32).reshape(-1,2)
        denominator=n*a.opt.clip_length if name=='eatr' else float(d['duration'])
        spans=m.span_xx_to_cxw(windows/denominator)
        if name=='eatr': return d,query,video,spans
        sal=torch.zeros(n)
        for start,end in windows.tolist():
            if name=='qd_detr':
                l=max(0,min(n-1,int(start/float(d['duration'])*n)))
                r=max(l+1,min(n,int(np.ceil(end/float(d['duration'])*n))))
            else:
                l=max(0,int(start/2));r=min(n,max(int(start/2)+1,int(np.ceil(end/2))))
            sal[l:r]=1
        pos=torch.where(sal>0)[0].tolist(); neg=torch.where(sal==0)[0].tolist()
        p=random.choices(pos,k=2) if pos else [0,0]
        ng=random.choices(neg,k=2) if neg else p
        if name=='qd_detr': return d,query,video,spans,sal,p,ng,float(bool(pos) and bool(neg))
        return d,video,query,spans,sal,p,ng

def make_data(adapter, gt):
    a=adapter
    order=('clip','slowfast') if a.name=='moment_detr' else ('slowfast','clip')
    a.opt.v_feat_dirs=[str(FEATURES/k) for k in order]
    a.opt.t_feat_dir=str(FEATURES/'clip_text')
    a.opt.eval_path=str(GT); a.opt.eval_split_name='test'
    a.opt.results_dir=str(HERE/'unused_native_output')
    if a.name=='moment_detr':
        # Avoid the upstream missing-feature skip/write fallback entirely.
        class StrictFullDataset(a.native.StartEndDataset):
            def load_data(self): return copy.deepcopy(gt)
        data=StrictFullDataset(**a.native.build_dataset_config(a.opt,str(GT),load_labels=True,keep_empty_gt=True))
        import evaluate
        # Official localization must consume raw proposal confidence. Existence
        # remains a separate exported field and is gated only by the evaluator.
        evaluate.apply_existence_gate=lambda window_scores,*args,**kwargs: window_scores
    else: data=FullSoccer(a,gt)
    assert data.data==gt
    return data

def load(entry):
    random.seed(2023); np.random.seed(2023); torch.manual_seed(2023); torch.cuda.manual_seed_all(2023)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    a=Adapter(entry['backbone'],'multi' if entry['method'].startswith('multi') else 'baseline')
    checkpoint=Path(entry['checkpoint'])
    assert sha(checkpoint)==entry['sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert len(state['model'])==entry['state_keys']
    mode='frozen' if entry['method'].endswith('_frozen') else 'full' if entry['method'].endswith('_full') else None
    model=RouteGEQ(a.base,a.name,mode).cuda() if mode else a.base
    loaded=model.load_state_dict(state['model'],strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    frozen={}
    if mode=='frozen':
        assert sha(entry['source_checkpoint'])==entry['source_sha256']
        parent=torch.load(entry['source_checkpoint'],map_location='cpu',weights_only=False)['model']
        native=model.base.state_dict()
        assert native.keys()==parent.keys()
        assert all(torch.equal(native[k].cpu(),parent[k]) for k in parent)
        frozen=dict(native_tensors_equal=True,native_tensors=len(parent),parent_checkpoint_sha256=entry['source_sha256'])
    model.eval().requires_grad_(False)
    assert all(not m.training for m in model.modules())
    return a,model,frozen

def validate_predictions(predictions, gt, nqueries):
    assert len(predictions)==len(gt)==len({p['qid'] for p in predictions})
    assert {p['qid'] for p in predictions}=={d['qid'] for d in gt}
    index={d['qid']:d for d in gt}
    counts=[]
    for p in predictions:
        d=index[p['qid']]
        assert (p['vid'],p['query'])==(d['vid'],d['query'])
        assert np.isfinite(p['pred_exist_score']) and 0<=p['pred_exist_score']<=1
        wins=p['pred_relevant_windows']; counts.append(len(wins))
        assert len(wins)==nqueries and np.isfinite(wins).all(), (p['qid'],len(wins),nqueries)
        assert all(0<=s<=e<=max(150,d['duration']) and 0<=c<=1 for s,e,c in wins),p
        assert all(wins[j][2]>=wins[j+1][2] for j in range(len(wins)-1))
    return dict(rows=len(gt),unique_qids=len(index),positive=sum(bool(d['relevant_windows']) for d in gt),
                negative=sum(not d['relevant_windows'] for d in gt),qid_coverage_exact=True,
                metadata_identity_exact=True,candidate_counts=sorted(set(counts)))

def evaluator():
    sys.path.insert(0,'/home/guoxiangyu/GMR_Unseen/experiments/table2_repro_20261022/shared')
    from official_eval.eval_main import evaluate_gmr
    return evaluate_gmr

def run(entry,phase):
    directory=run_dir(entry);directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/f'{phase}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    audit=json.loads((HERE/'INPUT_AUDIT.json').read_text())
    assert audit['input_sha256'][str(GT)]==sha(GT)
    assert audit['code_sha256']==code_files(), 'Code changed since input audit; rerun input audit before dispatch'
    identity=dict(checkpoint_sha256=entry['sha256'],gt_sha256=sha(GT),
                  feature_manifest_sha256=DATA['feature_manifest_sha256'],
                  feature_files_sha256=audit['feature_sha256'],input_audit_sha256=sha(HERE/'INPUT_AUDIT.json'),
                  code_sha256=audit['code_sha256'])
    marker=directory/('PREFLIGHT.json' if phase=='preflight' else 'COMPLETE.json')
    if marker.exists():
        old=json.loads(marker.read_text())
        assert old['identity']==identity
        assert old['predictions_sha256']==sha(directory/old['predictions_file'])
        if phase=='formal': assert old['metrics_sha256']==sha(directory/'metrics.json')
        print('REUSED_VALID_COMPLETION',entry['backbone'],entry['method'],phase,flush=True);return
    if phase=='formal':
        pre=json.loads((directory/'PREFLIGHT.json').read_text());assert pre['identity']==identity and pre['passed']
        assert all((run_dir(e)/'PREFLIGHT.json').is_file() for e in CHECKPOINTS['checkpoints'])
        if '_geq_' in entry['method']:
            assert all((run_dir(e)/'COMPLETE.json').is_file() for e in CHECKPOINTS['checkpoints'] if e['method'] in ('gmr','multi'))
    a,model,frozen=load(entry)
    gt=rows(GT)
    if phase=='preflight':
        ids=[next(i for i,d in enumerate(gt) if len(d['relevant_windows'])==n) for n in (0,1,2)]
        special={d['vid'] for d in audit['over_75_steps']+audit['temporal_mismatches_gt_vs_features']}
        ids+= [next(i for i,d in enumerate(gt) if d['vid']==v) for v in sorted(special)]
        gt=[gt[i] for i in dict.fromkeys(ids)]
    data=make_data(a,gt)
    seen=dict(batches=0,queries=0,max_video_steps=0,max_text_tokens=0)
    def check_inputs(module,args,kw):
        assert not torch.is_grad_enabled()
        assert 'targets' not in kw
        for key in ('src_vid','src_txt','src_vid_mask','src_txt_mask'): assert torch.isfinite(kw[key]).all()
        v,t=kw['src_vid'],kw['src_txt']; vm,tm=kw['src_vid_mask'],kw['src_txt_mask']
        assert v.shape[:2]==vm.shape and t.shape[:2]==tm.shape
        assert v.shape[-1]==2818 and t.shape[-1]==512 and v.shape[1]<=75 and t.shape[1]<=32
        assert vm.bool().any(dim=1).all() and tm.bool().any(dim=1).all()
        for mask in (vm,tm):
            assert ((mask==0)|(mask==1)).all()
            assert not ((~mask.bool()[:,:-1]) & mask.bool()[:,1:]).any()
        seen['batches']+=1;seen['queries']+=len(v)
        seen['max_video_steps']=max(seen['max_video_steps'],v.shape[1]);seen['max_text_tokens']=max(seen['max_text_tokens'],t.shape[1])
    def check_outputs(module,args,out):
        assert 'o2m_outputs' not in out
        for k in ('pred_logits','pred_spans'):
            assert torch.isfinite(out[k]).all()
    model.register_forward_pre_hook(check_inputs,with_kwargs=True)
    model.register_forward_hook(check_outputs)
    started=time.time()
    with torch.no_grad(): predictions=a.predict(model,data)
    assert seen['queries']==len(gt)
    coverage=validate_predictions(predictions,gt,a.opt.num_queries)
    if phase=='formal' and entry['method'].endswith('_frozen'):
        parent_method='multi' if entry['method'].startswith('multi') else 'gmr'
        parent_dir=HERE/'runs'/entry['backbone']/parent_method
        parent_marker=json.loads((parent_dir/'COMPLETE.json').read_text())
        assert parent_marker['identity']['gt_sha256']==identity['gt_sha256']
        assert parent_marker['identity']['checkpoint_sha256']==entry['source_sha256']
        assert sha(parent_dir/'predictions.jsonl')==parent_marker['predictions_sha256']
        parent=rows(parent_dir/'predictions.jsonl'); by_qid={p['qid']:p for p in predictions}
        # Export literally the parent's new Full candidates, including order,
        # confidence and all other fields; only existence is replaced.
        predictions=[dict(p,pred_exist_score=by_qid[p['qid']]['pred_exist_score']) for p in parent]
        assert all({k:v for k,v in p.items() if k!='pred_exist_score'}==
                   {k:v for k,v in q.items() if k!='pred_exist_score'} for p,q in zip(predictions,parent))
        frozen.update(parent_full_predictions_sha256=sha(parent_dir/'predictions.jsonl'),
                      full_candidates_exact=True,parent_method=parent_method)
    pred_name='preflight_predictions.jsonl' if phase=='preflight' else 'predictions.jsonl'
    jsonl(directory/pred_name,predictions)
    result=dict(passed=True,phase=phase,backbone=entry['backbone'],method=entry['method'],
                checkpoint=entry,identity=identity,strict_load=True,training=False,gradients=False,
                coverage=coverage,input_checks=seen,frozen=frozen,
                actual_video_feature_order=a.opt.v_feat_dirs,text_features=a.opt.t_feat_dir,
                max_v_l=75,max_q_l=32,num_queries=a.opt.num_queries,
                native_options=vars(a.opt),
                moment_detr_raw_localization_scores=True if a.name=='moment_detr' else None,
                predictions_file=pred_name,predictions_sha256=sha(directory/pred_name),
                seconds=time.time()-started,device=torch.cuda.get_device_name(),
                cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),torch_version=torch.__version__)
    if phase=='formal':
        evaluate=evaluator()
        reports={str(t):evaluate(predictions,gt,cls_thresholds=(.4,.5,.6),gmiou_cls_threshold=t,map_num_workers=0,verbose=False) for t in (.4,.5,.6)}
        dump(directory/'metrics.json',reports)
        if frozen:
            parent_reports=json.loads((parent_dir/'metrics.json').read_text())
            for t in reports:
                for key in ('mAP_detail','mR_detail','mR+_detail','mIoU_detail','mIoU+_detail'):
                    assert reports[t][key]==parent_reports[t][key],(t,key)
            frozen['all_localization_metrics_exact']=True
        result['metrics_sha256']=sha(directory/'metrics.json')
    dump(marker,result)
    print('PASSED',phase,entry['backbone'],entry['method'],len(predictions),'seconds',round(time.time()-started,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=('preflight','formal'),required=True)
    args=p.parse_args();torch.set_num_threads(2)
    run(CHECKPOINTS['checkpoints'][args.index],args.phase)
