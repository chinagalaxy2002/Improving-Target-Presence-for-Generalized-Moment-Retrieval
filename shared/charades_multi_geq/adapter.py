import sys,copy,importlib.util
from paths import *
import torch
from torch import nn
from torch.nn import functional as F
from native_factory import TargetAdapter
from data import Features

def load_local(name,file):
 spec=importlib.util.spec_from_file_location(name,KIT/file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
class Adapter:
 def __init__(self,name,multi,device='cuda'):
  load_local("dab_routes_local","dab_routes.py")
  self.name=name;self.native=TargetAdapter(name,device,'charades',seed=2023,config={'enable_pat_select':False})
  if name=='eatr' and multi:
   self.native.cfg['multi_auxiliary_query_scope']='positive_only_v1'
  self.opt=self.native.opt;self.opt.num_workers=2;self.opt.seed=2023;self.native.cfg["seed"]=2023
  self.opt.max_v_l=json.loads((OUT/'data/audit.json').read_text())['max_v_l']
  self.bs={'moment_detr':8,'eatr':32,'qd_detr':32,'flashvtg':8}[name]
  self.base=self.native.base;self.native.criterion.eval()
  if name=='eatr':
   def presence(mod,args,out):
    # EaTR native factory supplies the head, while its forward lacks existence output.
    if 'pred_exist_logits' not in out:
     memory=out.pop('_presence_memory') if '_presence_memory' in out else self._memory
     valid=self._valid;pooled=(memory*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
     out['pred_exist_logits']=mod.exist_head(pooled).squeeze(-1)
   def memory(mod,args,out):self._memory=out[1]
   self.base.transformer.register_forward_hook(memory)
   self.base.register_forward_hook(presence)
  init=OUT/'runs'/name/'native_init.pt';init.parent.mkdir(parents=True,exist_ok=True)
  import fcntl
  with init.with_suffix('.lock').open('a') as guard:
   fcntl.flock(guard,fcntl.LOCK_EX)
   if init.exists():self.base.load_state_dict(torch.load(init,map_location='cpu',weights_only=False)['model'],strict=True)
   else:
    from engine_utils import save,state_hash
    save(init,dict(model=self.base.state_dict(),seed=2023,sha256=state_hash(self.base)))
  variant='multi' if multi else 'baseline'
  if name=='moment_detr':self.base=load_local('moment_routes_local','moment_routes.py').MultiRoute(self.base,variant)
  elif name=='flashvtg':self.base=load_local('flash_aux_local','flash_aux.py').FlashAux(self.base,multi)
  else:self.base=load_local('dab_routes_local','dab_routes.py').MultiRouteModel(self.base,variant,name)
  self.base.to(device);self.device=device
 def dataset(self,split):return Features(rows(OUT/'data'/f'{split}.jsonl'),self)
 def collate(self,items):
  metas,inputs,targets=self.native.collate(items);inputs.pop('durations',None)
  self._valid=inputs['src_vid_mask'].bool()
  return metas,inputs,targets
 def loss(self,out,targets,metas,epoch):
  native={k:v for k,v in out.items() if k!='o2m_outputs'}
  total,parts=self.native.native_loss(native,targets,metas,epoch-1)
  self.last_loss_components={'native_total':total.detach(),**{k:v.detach() for k,v in parts.items() if k.startswith('loss_') and isinstance(v,torch.Tensor)}}
  native_total=total
  # Fixed K=2 Hungarian matching shared by all train-only auxiliary routes.
  from scipy.optimize import linear_sum_assignment
  geometry=load_local("aux_geometry", "dab_routes.py")
  _generalized_iou,_span_xx=geometry._generalized_iou,geometry._span_xx
  cspan=float(self.native.cfg.get("span_loss_coef",10));cgiou=float(self.native.cfg.get("giou_loss_coef",1));cclass=float(self.native.cfg.get("label_loss_coef",4))
  mspan=float(self.native.cfg.get("set_cost_span",10));mgiou=float(self.native.cfg.get("set_cost_giou",1));mclass=float(self.native.cfg.get("set_cost_class",4))
  span_labels=targets['span_labels'];positive=None
  if self.native.cfg.get('multi_auxiliary_query_scope')=='positive_only_v1':
   positive=torch.tensor([len(t['spans'])>0 for t in span_labels],device=self.device,dtype=torch.bool)
   if not positive.any():
    self.last_loss_components['aux_total']=total.detach()*0
    return total
   span_labels=[t for t,keep in zip(span_labels,positive.tolist()) if keep]
  for route,pred in out.get('o2m_outputs',{}).items():
   route_total=total.detach()*0
   for p in pred.get('aux_outputs',[])+[pred]:
    z=p['pred_logits'];sp=p['pred_spans'];cls=torch.ones(z.shape[:2],dtype=torch.long,device=z.device);terms=[]
    if positive is not None:
     z=z[positive];sp=sp[positive];cls=cls[positive]
    for i,t in enumerate(span_labels):
     gt=t['spans'].repeat_interleave(2,dim=0)
     if len(gt):
      a=_span_xx(sp[i]);b=_span_xx(gt)
      giou=torch.stack([_generalized_iou(a,b[j:j+1].expand_as(a)) for j in range(len(b))],1)
      cost=-mclass*z[i].softmax(-1)[:,0:1]+mspan*torch.cdist(sp[i],gt,p=1)-mgiou*giou
      ii,jj=linear_sum_assignment(cost.detach().cpu().numpy());cls[i,ii]=0
      terms.append(cspan*F.l1_loss(sp[i,ii],gt[jj])+cgiou*(1-_generalized_iou(a[ii],b[jj])).mean())
    aux=cclass*F.cross_entropy(z.transpose(1,2),cls,weight=z.new_tensor([1.,.1]))+(torch.stack(terms).mean() if terms else sp.sum()*0)
    total=total+.15*aux
    route_total=route_total+.15*aux.detach()
   self.last_loss_components['aux_'+route]=route_total
  self.last_loss_components['aux_total']=(total-native_total).detach()
  return total
