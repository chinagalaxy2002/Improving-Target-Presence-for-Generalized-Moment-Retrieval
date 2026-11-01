"""FlashVTG + Multi-TQAux: train-only cross-architecture adaptation."""
import copy
import torch
from torch import nn
class Layer(nn.Module):
 def __init__(self):
  super().__init__();self.sa=nn.MultiheadAttention(256,8,dropout=.1,batch_first=True);self.ca=nn.MultiheadAttention(256,8,dropout=.1,batch_first=True)
  self.common=nn.Sequential(nn.Linear(256,1024),nn.ReLU(),nn.Dropout(.1),nn.Linear(1024,256));self.private=copy.deepcopy(self.common)
  self.norms=nn.ModuleList([nn.LayerNorm(256) for _ in range(3)]);self.drop=nn.Dropout(.1)
 def forward(self,x,memory,valid,instruction=None):
  q=x if instruction is None else torch.cat((instruction.expand(x.shape[0],-1,-1),x),1)
  x=self.norms[0](x+self.drop(self.sa(q,q,q,need_weights=False)[0][:,-24:]))
  x=self.norms[1](x+self.drop(self.ca(x,memory,memory,key_padding_mask=~valid,need_weights=False)[0]))
  return self.norms[2](x+self.drop((self.private if instruction is None else self.common)(x)))
class FlashAux(nn.Module):
 def __init__(self,base,enabled):
  super().__init__();self.base=base;self.routes=['sepffn','instruction'] if enabled else []
  self.queries=nn.ParameterDict({r:nn.Parameter(torch.randn(1,24,256)*.02) for r in self.routes})
  self.instructions=nn.Parameter(torch.randn(1,4,256)*.02) if enabled else None
  self.layers=nn.ModuleList([Layer() for _ in range(2)]) if enabled else nn.ModuleList()
  self.cls=nn.Linear(256,2) if enabled else None;self.span=nn.Linear(256,2) if enabled else None
  self.memory=None;self.executions=0
  base.transformer.register_forward_hook(self.capture)
 def capture(self,mod,args,out):
  if self.training and self.routes and self.memory is None:self.memory=out[0].transpose(0,1)
 def forward(self,**inputs):
  self.memory=None;out=self.base(**inputs)
  if self.training and self.routes:
   valid=inputs['src_vid_mask'].bool();memory=self.memory[:,:valid.shape[1]];result={}
   for r in self.routes:
    x=self.queries[r].expand(memory.shape[0],-1,-1);states=[]
    for layer in self.layers:
     x=layer(x,memory,valid,self.instructions if r=='instruction' else None);states.append(dict(pred_logits=self.cls(x),pred_spans=self.span(x).sigmoid()))
    result[r]=dict(states[-1],aux_outputs=states[:-1]);self.executions+=1
   out['o2m_outputs']=result
  self.memory=None;return out
