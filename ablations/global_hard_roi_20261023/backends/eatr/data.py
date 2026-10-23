"""Isolated, augmentation-free exact Phase 1 EaTR SoccerDataset/collate definitions."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.span_utils import span_xx_to_cxw
SOURCE=Path('/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval')
class Soccer(Dataset):
 def __init__(self,split):
  assert split in ('train','val','test'),'Test curves are user-requested diagnostics'
  self.data=[json.loads(l) for l in (SOURCE/f'data/label/Standard/{split}.jsonl').read_text().splitlines() if l.strip()]
 def __len__(self):return len(self.data)
 def __getitem__(self,i):
  m=self.data[i];vs=[]
  for kind in ('slowfast','clip'):
   with np.load(SOURCE/f'Soccergmr/{kind}/{m["vid"]}.npz') as z:v=z['features'][:75].astype(np.float32)
   v=v/(np.linalg.norm(v,axis=-1,keepdims=True)+1e-5);vs.append(v)
  length=min(map(len,vs));v=torch.from_numpy(np.concatenate([a[:length] for a in vs],axis=-1))
  st=torch.arange(length,dtype=torch.float32)/length;v=torch.cat([v,torch.stack([st,st+1/length],-1)],-1)
  with np.load(SOURCE/f'Soccergmr/clip_text/qid{m["qid"]}.npz') as z:q=z['last_hidden_state'][:32].astype(np.float32)
  q=torch.from_numpy(q/(np.linalg.norm(q,axis=-1,keepdims=True)+1e-5))
  spans=span_xx_to_cxw(torch.tensor(m['relevant_windows'],dtype=torch.float32).reshape(-1,2)/(length*2))
  assert v.shape[-1]==2818 and q.shape[-1]==512 and len(spans)==len(m['relevant_windows'])
  return m,q,v,spans

def collate(batch):
 metas,qs,vs,spans=zip(*batch)
 qmask=pad_sequence([torch.ones(len(q),dtype=torch.bool) for q in qs],batch_first=True)
 vmask=pad_sequence([torch.ones(len(v),dtype=torch.bool) for v in vs],batch_first=True)
 inputs={'src_txt':pad_sequence(qs,batch_first=True),'src_txt_mask':qmask,'src_vid':pad_sequence(vs,batch_first=True),'src_vid_mask':vmask}
 return list(metas),inputs,{'span_labels':[{'spans':s} for s in spans],'exist_label':torch.tensor([bool(len(s)) for s in spans],dtype=torch.float32)}

def move(inputs,targets,device):
 return {k:v.to(device) for k,v in inputs.items()},{'span_labels':[{'spans':s['spans'].to(device)} for s in targets['span_labels']],'exist_label':targets['exist_label'].to(device)}
