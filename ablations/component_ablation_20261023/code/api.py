"""Only two supported public models: native E0 and S4 minus Fusion."""
from environment import ROOT,seed_all,read_json,options,E0_CHECKPOINTS,S4_CHECKPOINT,digest
import torch
from torch import nn
SHAS={'E0':'15f3054881d562801182efbc1b96b1f26de4c58d72466a8337022027b6485484','S4Fusion':'2e2ff2a728012bad5965e51b13b4ee4452b9ade09df25a571e1ba34a72dd4301'}
class BaselineModel(nn.Module):
 def __init__(self,base):super().__init__();self.base=base
 def forward(self,**inputs):return self.base(**inputs)
def job_for(name):
 assert name in SHAS
 return read_json(ROOT/'configs'/('baseline.json' if name=='E0' else 's4_fusion.json'))
def build_model(name,device='cpu',initialize=True):
 job=job_for(name)
 if name=='E0':
  from models.flash_vtg_gmr.model import build_model1
  seed_all(2026);opt=options(job,device);base,criterion=build_model1(opt)
  return BaselineModel(base).to(opt.device),opt,{},criterion
 from runtime import build
 if initialize:assert digest(E0_CHECKPOINTS[2026])==SHAS['E0'],'Wrong E0 initialization'
 assert job['presence_source']=='global_refine' and job['use_lora'] and job['lora_rank']==8 and job['use_boundary_refine']
 assert not job['evidence_gate'] and not job['balanced_safeguard'] and job['multi_weight']==0
 model,opt,checks=build(job,device,initialize=initialize)
 return model,opt,checks,None
def load_selected(name,device='cpu',checkpoint=None):
 path=checkpoint or (E0_CHECKPOINTS[2026] if name=='E0' else S4_CHECKPOINT)
 assert digest(path)==SHAS[name], 'Wrong checkpoint identity'
 model,opt,_,_=build_model(name,device,initialize=False)
 saved=torch.load(path,map_location='cpu',weights_only=False)
 model.load_state_dict(saved['model'],strict=True);model.eval()
 return model,opt,saved
