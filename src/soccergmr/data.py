"""Common SoccerGMR assets and targets, with explicit backbone feature profiles."""
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .io import read_jsonl, sha256, dump_json


def normalize(array):
    return array/(np.linalg.norm(array, axis=-1, keepdims=True)+1e-5)


class SoccerDataset(Dataset):
    def __init__(self, config, split, spec):
        self.config, self.split, self.spec = config, split, spec
        self.path = Path(config['data']['splits'][split])
        self.data = read_jsonl(self.path)
        self.feature_root = Path(config['data']['features']['full' if split=='test_full' else 'standard'])
        ids = [r['qid'] for r in self.data]
        if len(ids)!=len(set(ids)):
            raise ValueError('Duplicate qid within '+split)
        for row in self.data:
            if not isinstance(row.get('duration'), (int,float)) or row['duration']<=0:
                raise ValueError('Invalid duration')
            for window in row['relevant_windows']:
                if len(window)!=2 or not (0<=window[0]<window[1]<=row['duration']+1e-5):
                    raise ValueError('Invalid GT window: '+str(row['qid']))

    def __len__(self):
        return len(self.data)

    def feature_paths(self, row):
        stem = Path(row['vid']).stem
        return [self.feature_root/kind/(stem+'.npz') for kind in self.spec.feature_order] + [self.feature_root/'clip_text'/('qid'+str(row['qid'])+'.npz')]

    def __getitem__(self, index):
        row = self.data[index]
        paths = self.feature_paths(row)
        arrays=[]
        for kind,path in zip(self.spec.feature_order, paths):
            with np.load(path) as archive:
                feature=archive['features'].astype(np.float32)
            if feature.ndim!=2 or feature.shape[1]!={'clip':512,'slowfast':2304}[kind] or not np.isfinite(feature).all():
                raise ValueError('Invalid video features: '+str(path))
            arrays.append(feature)
        length=min(len(a) for a in arrays)
        if not 0<length<=self.config['data']['max_video_tokens']:
            raise ValueError(f'Video length {length} exceeds configured limit; no silent truncation: {row["vid"]}')
        video=np.concatenate([normalize(a[:length]) for a in arrays],axis=-1)
        tef=np.arange(length,dtype=np.float32)/length
        video=np.concatenate([video,np.stack([tef,tef+1/length],axis=-1)],axis=-1)
        with np.load(paths[-1]) as archive:
            query=archive['last_hidden_state'].astype(np.float32)[:self.config['data']['max_text_tokens']]
            if self.spec.text_attention_mask:
                query=query[archive['attention_mask'][:len(query)].astype(bool)]
        if query.ndim!=2 or query.shape[1]!=512 or len(query)==0 or not np.isfinite(query).all():
            raise ValueError('Invalid text features: '+str(paths[-1]))
        query=normalize(query)
        windows=torch.tensor(row['relevant_windows'],dtype=torch.float32).reshape(-1,2)
        boundaries=windows/float(row['duration'])
        spans=torch.stack(((boundaries[:,0]+boundaries[:,1])/2,boundaries[:,1]-boundaries[:,0]),-1)
        # Targets and export both use actual video duration, including short and long videos.
        masks=torch.zeros(len(windows),length)
        for i,(start,end) in enumerate(windows.tolist()):
            a=max(0,min(length-1,math.floor(start/row['duration']*length)))
            b=max(a+1,min(length,math.ceil(end/row['duration']*length)))
            masks[i,a:b]=1
        saliency=masks.amax(0) if len(masks) else torch.zeros(length)
        pos=torch.where(saliency>0)[0].tolist()
        neg=torch.where(saliency==0)[0].tolist()
        # Deterministic GT-derived pair labels do not consume evaluation/training RNG.
        pi=([pos[0],pos[-1]] if pos else [0,0])
        ni=([neg[0],neg[-1]] if neg else pi)
        return dict(meta=row,video=torch.from_numpy(video),query=torch.from_numpy(query),
                    spans=spans,masks=masks,saliency=saliency,pos=pi,neg=ni)


def collate(items):
    metas=[x['meta'] for x in items]
    videos=[x['video'] for x in items];queries=[x['query'] for x in items]
    vm=pad_sequence([torch.ones(len(v),dtype=torch.bool) for v in videos],batch_first=True)
    tm=pad_sequence([torch.ones(len(q),dtype=torch.bool) for q in queries],batch_first=True)
    max_length=vm.shape[1]
    masks=[]
    for item in items:
        padded=torch.zeros(len(item['masks']),max_length)
        padded[:,:item['masks'].shape[1]]=item['masks']
        masks.append(padded)
    saliency=pad_sequence([x['saliency'] for x in items],batch_first=True)
    inputs=dict(src_vid=pad_sequence(videos,batch_first=True),src_vid_mask=vm.float(),
                src_txt=pad_sequence(queries,batch_first=True),src_txt_mask=tm.float())
    targets=dict(span_labels=[dict(spans=x['spans']) for x in items],
        exist_label=torch.tensor([bool(m['relevant_windows']) for m in metas],dtype=torch.float32),
        saliency_all_labels=saliency,relevant_clips=saliency,src_pos_mask=saliency,
        saliency_pos_labels=torch.tensor([x['pos'] for x in items]),
        saliency_neg_labels=torch.tensor([x['neg'] for x in items]),
        saliency_pair_mask=torch.tensor([bool(x['saliency'].any()) and bool((x['saliency']==0).any()) for x in items],dtype=torch.float32),
        mask_labels=masks)
    return metas,inputs,targets


def move(value, device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device) for v in value]
    return value


def audit_data(config, spec, destination, arrays=True):
    reports={};sets={};feature_hashes={}
    for split in ('train','val','test_standard','test_full'):
        dataset=SoccerDataset(config,split,spec)
        sets[split]={Path(r['vid']).stem for r in dataset.data}
        for i,row in enumerate(dataset.data):
            for path in dataset.feature_paths(row):
                if not path.is_file():raise FileNotFoundError(path)
                if str(path) not in feature_hashes:feature_hashes[str(path)]=sha256(path)
            if arrays:dataset[i]
        reports[split]=dict(path=str(dataset.path),sha256=sha256(dataset.path),rows=len(dataset),
            positive=sum(bool(r['relevant_windows']) for r in dataset.data),
            negative=sum(not r['relevant_windows'] for r in dataset.data),feature_root=str(dataset.feature_root))
    for left,right in [('train','val'),('train','test_standard'),('train','test_full'),('val','test_standard'),('val','test_full')]:
        overlap=sets[left]&sets[right]
        if overlap:raise ValueError(f'Video leakage between {left}/{right}: {sorted(overlap)[:3]}')
    result=dict(passed=True,dataset='SoccerGMR',train_split='Standard',splits=reports,
                full_label=config['data']['full_label'],feature_sha256=feature_hashes,
                full_array_validation=arrays,source='Labels and feature paths from explicit config; no automatic exclusions')
    dump_json(destination,result)
    return result
