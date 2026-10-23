"""Verbatim Phase 1 native data transforms, with no training or output side effects."""
import json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from qd_detr.span_utils import span_xx_to_cxw
ASSETS=Path('/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval/Soccergmr')
LABELS=ASSETS.parent/'data/label/Standard'
def norm(a):return a/(np.linalg.norm(a,axis=-1,keepdims=True)+1e-5)

class Soccer(Dataset):
 def __init__(self,split):
  self.data=[json.loads(l) for l in (LABELS/f'{split}.jsonl').read_text().splitlines() if l.strip()]
  self.cache={}
 def __len__(self):return len(self.data)
 def __getitem__(self,i):
  d=self.data[i];vid=Path(d['vid']).stem
  if vid not in self.cache:
   fs=[np.load(ASSETS/f/f'{vid}.npz')['features'].astype(np.float32) for f in ('slowfast','clip')]
   n=min(map(len,fs));assert n>0 and n<=75
   self.cache[vid]=torch.from_numpy(np.concatenate([norm(f[:n]) for f in fs],-1))
  video=self.cache[vid];n=len(video)
  tef=torch.stack([torch.arange(n)/n,(torch.arange(n)+1)/n],-1)
  video=torch.cat([video,tef],-1)
  q=norm(np.load(ASSETS/'clip_text'/f'qid{d["qid"]}.npz')['last_hidden_state'][:32].astype(np.float32))
  windows=torch.tensor(d['relevant_windows'],dtype=torch.float32).reshape(-1,2)
  spans=span_xx_to_cxw(windows/float(d['duration']))
  sal=torch.zeros(n)
  for start,end in windows:
   a=max(0,min(n-1,int(start/float(d['duration'])*n)));b=max(a+1,min(n,int(np.ceil(float(end)/float(d['duration'])*n))))
   sal[a:b]=1
  pos=torch.where(sal>0)[0].tolist();neg=torch.where(sal==0)[0].tolist()
  p=random.choices(pos,k=2) if pos else [0,0];ng=random.choices(neg,k=2) if neg else p
  return d,torch.from_numpy(q),video,spans,sal,p,ng,float(bool(pos) and bool(neg))

def collate(items):
 meta,q,v,spans,sal,pos,neg,pm=zip(*items)
 qm=pad_sequence([torch.ones(len(x)) for x in q],batch_first=True);vm=pad_sequence([torch.ones(len(x)) for x in v],batch_first=True)
 inputs={'src_txt':pad_sequence(q,batch_first=True),'src_vid':pad_sequence(v,batch_first=True),'src_txt_mask':qm,'src_vid_mask':vm}
 target={'span_labels':[{'spans':s} for s in spans],'saliency_all_labels':pad_sequence(sal,batch_first=True),'saliency_pos_labels':torch.tensor(pos),'saliency_neg_labels':torch.tensor(neg),'saliency_pair_mask':torch.tensor(pm)}
 return list(meta),inputs,target

def move(inputs,targets,device):
 inputs={k:v.to(device) for k,v in inputs.items()}
 target={k:([{'spans':x['spans'].to(device)} for x in v] if k=='span_labels' else v.to(device)) for k,v in targets.items()}
 return inputs,target
